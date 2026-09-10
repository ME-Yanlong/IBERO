"""独立新进程对比数值库线程：记录 Windows commit 与逐控制帧物理字节摘要。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def process_memory():
    if sys.platform != "win32":
        return {"status": "not_measured_on_this_platform"}
    # 只读当前子进程；不安装 psutil/threadpoolctl，不改 OS 配置。
    command = (
        f"$measuredProcess = Get-Process -Id {os.getpid()}\n"
        "[pscustomobject]@{PrivateBytes=$measuredProcess.PrivateMemorySize64; "
        "WorkingSetBytes=$measuredProcess.WorkingSet64; "
        "ThreadCount=$measuredProcess.Threads.Count} | ConvertTo-Json -Compress"
    )
    raw = subprocess.check_output(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=15,
    )
    return json.loads(raw)


def child(args):
    import mujoco
    import numpy as np
    from ibero.control.milling_bench import MillingBenchScript
    from ibero.core.reproducibility import simulation_source_hash
    from ibero.envs.plate_milling import PlateMillingEnv
    from ibero.review import Trace

    args.output.mkdir(parents=True, exist_ok=False)
    env = PlateMillingEnv(args.scene, shape="through_hole")
    source = simulation_source_hash()
    report = dict(
        scope="bounded_prefix_not_full_task_or_G8",
        requested_sim_seconds=args.seconds,
        source_hash=source,
        memory_after_build=process_memory(),
        passed=False,
        numpy_version=np.__version__,
        mujoco_version=mujoco.__version__,
        thread_environment={
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    )
    policy = MillingBenchScript(env, env.target)
    state = np.empty(mujoco.mj_stateSize(env.model, Trace.spec))
    state_digest, load_digest = hashlib.sha256(), hashlib.sha256()
    started = time.perf_counter()
    substeps = round(
        1 / env.scene.config["physics"]["control_hz"] / env.model.opt.timestep
    )
    for _ in range(round(args.seconds * env.scene.config["physics"]["control_hz"])):
        target, rpm = policy.command(env)
        info = env.step(target, rpm, substeps=substeps)
        mujoco.mj_getState(env.model, env.data, state, Trace.spec)
        state_digest.update(state.tobytes())
        load_digest.update(json.dumps(info, sort_keys=True, allow_nan=False).encode())
        if env._done:
            break
    report.update(
        wall_seconds=time.perf_counter() - started,
        sim_seconds=float(env.data.time),
        state_digest=state_digest.hexdigest(),
        info_digest=load_digest.hexdigest(),
        material_hash=env.stock.state_hash(),
        material_events=len(env.stock.events),
        invalid_reason=env.process.invalid_reason,
        memory_after_run=process_memory(),
    )
    simulation_source_hash.cache_clear()
    report["frozen_source"] = simulation_source_hash() == source
    report["passed"] = (
        report["frozen_source"] and not env._done and bool(env.stock.events)
    )
    if not env.stock.events:
        report["diagnostic_failure_reason"] = "prefix_did_not_reach_material_event"
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output, report["passed"], report["memory_after_build"], flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 30:
        parser.error("Prefix duration must be finite and in (0, 30] seconds")
    if args.child:
        raise SystemExit(child(args))
    args.output.mkdir(parents=True, exist_ok=False)
    from industrial_resources import require_worker_budget

    require_worker_budget(args.output, 2)
    children = []
    for count in (24, 1):
        env = os.environ.copy()
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = str(count)
        env["PYTHONPATH"] = str(args.source_root.resolve() / "src")
        output = args.output.resolve() / f"threads_{count}"
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            str(Path(__file__).resolve()),
            "--child",
            "--source-root",
            str(args.source_root.resolve()),
            "--scene",
            str(args.scene.resolve()),
            "--output",
            str(output),
            "--seconds",
            str(args.seconds),
        ]
        # 同时启动两份同源码前缀；只能改新子进程环境，不接触正在验收的进程。
        children.append((subprocess.Popen(cmd, cwd=args.source_root, env=env), output))
    exit_codes = [process.wait() for process, _ in children]
    rows = []
    for (_, output), exit_code in zip(children, exit_codes):
        path = output / "report.json"
        rows.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else dict(
                passed=False, exit_code=exit_code, exception="Child report missing"
            )
        )
    equal = {
        name: name in rows[0] and name in rows[1] and rows[0][name] == rows[1][name]
        for name in (
            "source_hash",
            "sim_seconds",
            "state_digest",
            "info_digest",
            "material_hash",
            "material_events",
        )
    }
    report = dict(
        scope="fixed_source_prefix_thread_comparison_not_full_G8",
        exit_codes=exit_codes,
        equalities=equal,
        results=rows,
        passed=all(code == 0 for code in exit_codes) and all(equal.values()),
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output, report["passed"], equal, flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
