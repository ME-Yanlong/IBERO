"""机器人切削短程剖析，记录真实失败；不是任务成功验收。"""

import argparse
import cProfile
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.control.milling_bench import MillingBenchScript
from ibero.core.reproducibility import simulation_source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or Path(
        "artifacts/industrial_core01/stock/g1_profile"
    ) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    env = PlateMillingEnv()
    manifest = env.manifest()
    policy = MillingBenchScript(env, env.target)
    profiler = cProfile.Profile()
    methods = {}
    started = time.perf_counter()
    profiler.enable()
    while env.data.time < args.seconds and not env._done:
        target, rpm = policy.command(env)
        info = env.step(target, rpm, substeps=100)
        name = env.process.coupling.last_diagnostics.get("method", "air")
        methods[name] = methods.get(name, 0) + 1
    profiler.disable()
    profiler.dump_stats(str(out / "profile.prof"))
    report = dict(
        scope="profile_diagnostic_not_G8",
        manifest=manifest,
        wall_seconds=time.perf_counter() - started,
        last_info=info,
        control_frame_method_samples=methods,
    )
    simulation_source_hash.cache_clear()
    report["source_unchanged"] = (
        report["manifest"]["source_hash"] == simulation_source_hash()
    )
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(out, report["wall_seconds"], methods, info["invalid_reason"], flush=True)


if __name__ == "__main__":
    main()
