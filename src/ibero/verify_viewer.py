"""显式启动真实桌面窗口的交互冒烟测试，不在无头 pytest 中自动执行。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import mujoco.viewer

import ibero
from ibero.multiview import MultiViewDashboard
from ibero.review import run_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/viewer_smoke.json")
    )
    args = parser.parse_args()
    env = ibero.make("ibero/CableStretch-v0")
    report = {
        "initial_wait": False,
        "completed_episodes": 0,
        "pause": False,
        "single_step": False,
        "previous_frame": False,
        "replay": False,
        "persistent_finish": False,
    }
    shared = {"phase": "initial", "ticks": 0}
    started = time.monotonic()
    launch = mujoco.viewer.launch_passive

    def launch_checked(*a, **kw):
        handle = launch(*a, **kw)
        shared["handle"] = handle
        return handle

    def dashboard_factory(sim, run_requested, controls):
        dashboard = MultiViewDashboard(sim, run_requested, controls)
        present = dashboard.present

        def checked(info, status, *, force=False):
            alive = present(info, status, force=force)
            if time.monotonic() - started > 300:
                report["timeout_phase"] = shared["phase"]
                report["timeout_sim_time"] = float(sim.data.time)
                shared["handle"].close()
                raise TimeoutError("Viewer interaction smoke test exceeded 300 s")
            now = float(sim.data.time)
            phase = shared["phase"]
            if shared.get("logged_phase") != phase:
                print("Viewer check:", phase, "sim time:", now, flush=True)
                shared["logged_phase"] = phase
            if info.get("failure_reason"):
                raise AssertionError(f"Viewer episode failed: {info}")
            # 所有 Tk/MuJoCo 操作均在主事件循环；只注入真实 reviewer 的按键事件。
            if phase == "initial":
                assert now == 0
                report["initial_wait"] = True
                controls.key(13)
                shared["phase"] = "first_run"
            elif phase == "first_run" and info["is_success"]:
                report["completed_episodes"] += 1
                shared.update(phase="finish_wait", frozen=now, ticks=0)
            elif phase == "finish_wait":
                assert now == shared["frozen"] and shared["handle"].is_running()
                shared["ticks"] += 1
                if shared["ticks"] >= 5:
                    report["persistent_finish"] = True
                    controls.key(13)
                    shared["phase"] = "pause_run"
            elif phase == "pause_run" and now >= 0.15:
                controls.key(80)
                shared.update(phase="paused", frozen=now, ticks=0)
            elif phase == "paused":
                assert now == shared["frozen"]
                shared["ticks"] += 1
                if shared["ticks"] >= 5:
                    report["pause"] = True
                    controls.key(78)
                    shared["phase"] = "single"
            elif phase == "single":
                assert abs(now - shared["frozen"] - sim.control_dt) < 1e-8
                report["single_step"] = True
                controls.key(66)
                shared["phase"] = "back"
            elif phase == "back":
                assert abs(now - shared["frozen"]) < 1e-8
                report["previous_frame"] = True
                controls.key(82)
                shared["phase"] = "replay"
            elif phase == "replay":
                assert info.get("replay")
                report["replay"] = True
                controls.key(13)
                shared["phase"] = "last_run"
            elif phase == "last_run" and info["is_success"]:
                report["completed_episodes"] += 1
                # 直接保存面板自己的渲染缓冲，窗口被遮挡时也不截到其他应用。
                from PIL import Image

                args.output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(dashboard._views.render()).save(
                    args.output.with_suffix(".png")
                )
                shared["handle"].close()
            return alive

        dashboard.present = checked
        return dashboard

    try:
        with (
            patch("mujoco.viewer.launch_passive", launch_checked),
            patch("ibero.multiview.try_create_dashboard", dashboard_factory),
        ):
            run_review(env, 600, 7, True, 10.0)
        report["passed"] = (
            all(v for v in report.values()) and report["completed_episodes"] == 2
        )
    finally:
        env.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
