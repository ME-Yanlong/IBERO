"""从已完成的真实材料轨迹测量构建、提交和回放成本，不伪造物理实时因子。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from ibero.core.reproducibility import simulation_source_hash
from ibero.core.stock_trace import StockTrace
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.milling_review import MillingViews


def distribution(values):
    return {
        "count": len(values),
        **{
            name: float(np.percentile(values, q)) if values else None
            for name, q in (("median_s", 50), ("p95_s", 95), ("max_s", 100))
        },
    }


def windows_hardware():
    """只读本机设备说明；无窗口、无联网，不以查询失败影响物理验收。"""
    if platform.system() != "Windows":
        return {"status": "not_queried_on_this_platform"}
    result = {}
    for name, command in (
        (
            "cpu",
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress",
        ),
        (
            "gpu",
            "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress",
        ),
        (
            "memory",
            "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress",
        ),
    ):
        try:
            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
            )
            result[name] = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as error:
            result[name] = {"query_error": f"{type(error).__name__}: {error}"}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, default=Path("scenes/plate_milling"))
    p.add_argument("--shape", choices=["through_hole", "slot", "pocket"], required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrent-load-note", required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = simulation_source_hash()
    report = {
        "scope": "review_and_material_commit_cost_not_headless_physics_RTF",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_hash": source,
        "concurrent_load_note": args.concurrent_load_note,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "device_inventory": windows_hardware(),
            "numeric_thread_environment": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "passed": False,
    }
    views = None
    try:
        t = time.perf_counter()
        env = PlateMillingEnv(args.scene, shape=args.shape)
        report["build_and_reset_seconds"] = time.perf_counter() - t
        report["manifest"] = env.manifest()
        report["scene_config"] = env.scene.config
        report["capacity"] = dict(
            geoms=env.model.ngeom,
            stock_cells=len(env.stock.centers),
            occupancy_bytes=env.stock.occupied.nbytes,
        )
        t = time.perf_counter()
        trace = StockTrace(env).load(args.trace)
        report["trace_load_seconds"] = time.perf_counter() - t
        report["trace_file_bytes"] = args.trace.stat().st_size
        report["frames"] = len(trace.states)
        report["events"] = len(trace.events)
        report["uncompressed_state_bytes"] = sum(s.nbytes for s in trace.states)
        # 仅测材料事务：静止初态逐事件重建终态，不将这一步宣称为重新加工。
        env.reset(seed=env.seed_value)  # load 已严格验证并恢复归档的初始种子。
        timings = []
        for event in trace.events:
            t = time.perf_counter()
            env.binding.commit(event, env.data)
            timings.append(time.perf_counter() - t)
        env.binding.ensure_consistent()
        report["material_commit"] = distribution(timings)
        report["rebuilt_material_matches"] = (
            env.stock.state_hash() == trace.material_hashes[-1]
        )
        t = time.perf_counter()
        views = MillingViews(env)
        report["display_clone_and_renderer_seconds"] = time.perf_counter() - t
        restore, render = [], []
        frames = np.linspace(
            0, len(trace.states) - 1, min(30, len(trace.states)), dtype=int
        )
        live_qpos, live_hash = env.data.qpos.copy(), env.stock.state_hash()
        for index in frames:
            t = time.perf_counter()
            views.set_frame(trace, int(index))
            restore.append(time.perf_counter() - t)
            t = time.perf_counter()
            picture = views.render(trace.infos[: index + 1])
            render.append(time.perf_counter() - t)
        picture.save(args.output / "final_multiview.png")
        report["display_restore"] = distribution(restore)
        report["three_views_and_panel_render"] = distribution(render)
        report["live_isolated"] = bool(
            np.array_equal(live_qpos, env.data.qpos)
            and live_hash == env.stock.state_hash()
        )
        report["passed"] = (
            report["live_isolated"] and report["rebuilt_material_matches"]
        )
    except Exception as error:
        report["exception"] = f"{type(error).__name__}: {error}"
    finally:
        if views is not None:
            views.close()
        simulation_source_hash.cache_clear()
        report["frozen_source"] = simulation_source_hash() == source
        report["passed"] &= report["frozen_source"]
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(args.output, report["passed"], report.get("exception"), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
