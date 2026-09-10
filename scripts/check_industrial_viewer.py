"""真实 Tk 工业窗口的短交互检查；完整成功两轮仍属于 G4/G8 长验收。"""

from datetime import datetime, timezone
from pathlib import Path
import json
import time
import argparse
from ibero.core.reproducibility import simulation_source_hash
from ibero.envs.latch_release import LatchReleaseEnv
from ibero.industrial_review import IndustrialReviewer


def progress_snapshot(app, state, now):
    """只读运行遥测；有窗口回调不等于物理成功，也不等于实测输入响应延迟。"""
    return {
        "scope": "viewer_check_progress_not_acceptance_evidence",
        "phase": state["phase"],
        "wall_seconds": now - state["started"],
        "simulation_seconds": float(app.env.data.time),
        "display_frame": app.index,
        "recorded_frames": len(app.trace.states),
        "completed_episodes": app.completed_episodes,
        "running": app.running and not app.closed,
        "finished": app.finished,
        "closed": app.closed,
        "worker_pending": app.future is not None and not app.future.done(),
        "max_check_callback_gap_s": state.get("max_callback_gap_s", 0.0),
        "phase_durations_s": dict(state.get("phase_durations_s", {})),
        "note": "Approximate concurrent counters; read final report.json for pass/fail",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also require two complete successful episodes",
    )
    parser.add_argument(
        "--scene",
        choices=["latch_release", "harness_unplug", "plate_milling"],
        default="latch_release",
    )
    parser.add_argument(
        "--shape", choices=["through_hole", "slot", "pocket"], default="through_hole"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or Path(
        "artifacts/industrial_core01",
        "stock" if args.scene == "plate_milling" else "latch",
        "viewer",
    ) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    source_hash = simulation_source_hash()
    reviewer_options = {}
    if args.scene == "plate_milling":
        from ibero.envs.plate_milling import PlateMillingEnv
        from ibero.milling_review import MillingReviewer

        env = PlateMillingEnv(shape=args.shape)
        reviewer = MillingReviewer
        reviewer_options["shape"] = args.shape
        env.control_dt = 1 / env.scene.config["physics"]["control_hz"]
        env.max_episode_steps = round(
            env.scene.constraints["task"]["max_seconds"] / env.control_dt
        )
    elif args.scene == "harness_unplug":
        from ibero.envs.harness_unplug import HarnessUnplugEnv

        env = HarnessUnplugEnv()
        reviewer = IndustrialReviewer
    else:
        env = LatchReleaseEnv()
        reviewer = IndustrialReviewer
    app = reviewer(
        env, seed=0, steps=env.max_episode_steps if args.full else 20, **reviewer_options
    )
    report = {
        "initial_wait": False,
        "pause": False,
        "single_step": False,
        "previous_frame": False,
        "history_does_not_integrate": False,
        "replay": False,
        "rerun": False,
    }
    # 提前关闭窗口也必须失败，不能因尚未创建成功字段而被 all() 当成通过。
    if args.full:
        report.update(
            successful_episode_1=False,
            successful_episode_2=False,
            persistent_finish=False,
        )
    state = {"phase": "initial", "started": time.monotonic()}

    def check():
        try:
            now = time.monotonic()
            previous = state.get("previous_callback", now)
            state["max_callback_gap_s"] = max(
                state.get("max_callback_gap_s", 0.0), now - previous
            )
            state["previous_callback"] = now
            phase = state["phase"]
            previous_phase = state.get("observed_phase")
            if phase != previous_phase:
                if previous_phase is not None:
                    state.setdefault("phase_durations_s", {})[previous_phase] = (
                        now - state["phase_started"]
                    )
                state.update(observed_phase=phase, phase_started=now)
            if now - state.get("last_progress_at", -float("inf")) >= 5:
                # 进度文件可更新，最终报告/轨迹仍只在唯一运行目录写入；不覆盖其他运行。
                (out / "progress.json").write_text(
                    json.dumps(progress_snapshot(app, state, now), indent=2),
                    encoding="utf-8",
                )
                state["last_progress_at"] = now
            if time.monotonic() - state["started"] > (14400 if args.full else 90):
                raise TimeoutError("Industrial viewer interaction check timed out")
            phase = state["phase"]
            if phase == "initial":
                assert env.data.time == 0 and not app.running
                report["initial_wait"] = True
                app.key("Return")
                state["phase"] = "run"
            elif phase == "run" and app.index >= 3:
                app.key("p")
                state["phase"] = "pause"
            elif phase == "pause" and app.future is None:
                report["pause"] = not app.running
                state.update(phase="single", index=app.index, time=float(env.data.time))
                app.key("n")
            elif (
                phase == "single" and app.future is None and app.index > state["index"]
            ):
                assert app.index == state["index"] + 1
                report["single_step"] = True
                state.update(phase="back", time=float(env.data.time))
                app.key("b")
            elif phase == "back":
                assert app.history and app.index == state["index"]
                assert env.data.time == state["time"]
                report["previous_frame"] = True
                report["history_does_not_integrate"] = True
                app.key("r")
                state["phase"] = "replay"
            elif phase == "replay" and not app.replaying:
                assert env.data.time == state["time"]
                report["replay"] = True
                app.key("Return")
                state["phase"] = "rerun"
            elif (
                phase == "rerun"
                and not app.pending_run
                and not app.resetting
                and env.data.time < state["time"]
            ):
                report["rerun"] = True
                if args.full:
                    state["phase"] = "full_first"
                    return app.root.after(100, check)
                app.last_image.save(out / "window.png")
                app.close()
                return
            elif (
                phase in {"full_first", "full_second"}
                and app.finished
                and app.future is None
            ):
                index = 1 if phase == "full_first" else 2
                # 失败回合也先留原始轨迹/截图，不能只留下 assertion 文本。
                app.trace.save(out / f"episode_{index}.npz")
                app.last_image.save(out / f"episode_{index}.png")
                assert app.trace.infos[-1]["success"], app.trace.infos[-1].get(
                    "failure_reason"
                )
                report[f"successful_episode_{index}"] = True
                print("GUI full episode", index, "passed", flush=True)
                if index == 1:
                    app.key("Return")
                    state["phase"] = "waiting_second_reset"
                else:
                    state.update(
                        phase="persistent_finish",
                        finished_at=time.monotonic(),
                        final_time=float(env.data.time),
                    )
            elif phase == "waiting_second_reset" and not app.finished:
                state["phase"] = "full_second"
            elif (
                phase == "persistent_finish"
                and time.monotonic() - state["finished_at"] > 1
            ):
                assert env.data.time == state["final_time"] and not app.closed
                report["persistent_finish"] = True
                app.close()
                return
        except Exception as error:
            report["error"] = str(error)
            app.close()
            return
        app.root.after(100, check)

    app.root.after(200, check)
    app.run()
    if hasattr(env, "close"):
        env.close()
    simulation_source_hash.cache_clear()
    report["source_unchanged_during_run"] = simulation_source_hash() == source_hash
    report["passed"] = all(report.values()) and "error" not in report
    report["source_hash"] = source_hash
    report["scene"] = args.scene
    report["shape"] = args.shape if args.scene == "plate_milling" else None
    report["full"] = args.full
    report["wall_seconds"] = time.monotonic() - state["started"]
    report["runtime_observability"] = progress_snapshot(app, state, time.monotonic())
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "progress.json").write_text(
        json.dumps(report["runtime_observability"], indent=2), encoding="utf-8"
    )
    print(out, report)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
