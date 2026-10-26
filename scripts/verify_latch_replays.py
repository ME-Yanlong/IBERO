"""逐例加载当前冻结卡扣/线束轨迹并回看端点；只验证存档，不恢复策略续跑。"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import ibero
from ibero.core.reproducibility import simulation_source_hash
from ibero.envs.latch_release import LatchReleaseEnv
from ibero.envs.harness_unplug import HarnessUnplugEnv
from ibero.review import Trace

# 从实际加载的仿真包定位菜谱，支持主目录验收器审计独立冻结 worktree。
ROOT = Path(ibero.__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to replace evidence")
    source = simulation_source_hash()
    group = json.loads(args.group.read_text(encoding="utf-8"))
    started = time.perf_counter()
    rows, envs = [], {}
    try:
        for case in group["results"]:
            name, seed = case["scene"], case["seed"]
            row = dict(scene=name, seed=seed, case=case["case"], passed=False)
            try:
                if name not in envs:
                    constructor = (
                        HarnessUnplugEnv
                        if name.startswith("harness_unplug")
                        else LatchReleaseEnv
                    )
                    envs[name] = constructor(ROOT / "scenes" / name)
                env = envs[name]
                env.reset(seed=seed)
                path = Path(case["path"]) / "trace.npz"
                raw = json.loads(
                    (path.parent / "report.json").read_text(encoding="utf-8")
                )
                trace = Trace(env).load(path)
                checks = dict(
                    source=raw.get("manifest", {}).get("source_hash") == source,
                    final_info=trace.infos[-1] == raw["final"],
                    frames=len(trace.states) == len(raw["trace"]) + 1,
                )
                state = np.empty(mujoco.mj_stateSize(env.model, Trace.spec))
                for index in (len(trace.states) - 1, 0, len(trace.states) - 1):
                    trace.restore(index)
                    mujoco.mj_getState(env.model, env.data, state, Trace.spec)
                    checks[f"state_{index}"] = bool(
                        np.array_equal(state, trace.states[index])
                    )
                checks["readonly_requires_reset"] = env._replay_restored
                row.update(
                    checks=checks,
                    passed=all(checks.values()),
                    trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    path=str(path),
                )
            except Exception as error:
                row["exception"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            print(
                name,
                seed,
                case["case"],
                row["passed"],
                row.get("exception"),
                flush=True,
            )
    finally:
        for env in envs.values():
            env.close()
    frozen = source == simulation_source_hash() == group.get("source_hash")
    report = dict(
        scope="all_saved_latch_group_replays_not_policy_checkpoint",
        source_hash=source,
        frozen_source=frozen,
        results=rows,
        group=str(args.group.resolve()),
        passed=frozen and len(rows) == 34 and all(r["passed"] for r in rows),
        wall_seconds=time.perf_counter() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
