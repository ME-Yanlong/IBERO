"""S4 单场景诊断/检查记录，完整种子门槛由后续成组验收判定。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from ibero.envs.latch_release import LatchReleaseEnv
from ibero.control.latch import LatchReleaseScript


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--case", default="normal")
    a = p.parse_args()
    out = Path("artifacts/industrial_core01/latch/robot_task") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    env = LatchReleaseEnv()
    env.reset(seed=a.seed)
    policy = LatchReleaseScript(a.case)
    manifest = env.manifest()
    rows = []
    started = time.monotonic()
    for k in range(env.max_episode_steps):
        _, _, term, trunc, info = env.step(policy.action(env))
        info["controller_phase"] = policy.phase
        rows.append(info)
        if k % 50 == 0 or term or trunc:
            print(
                k,
                policy.phase,
                {
                    key: info.get(key)
                    for key in (
                        "time_s",
                        "clearance_m",
                        "withdrawal_m",
                        "contact_force_n",
                        "grasped",
                        "invalid_reason",
                        "success",
                        "failure_reason",
                    )
                },
                flush=True,
            )
        if term or trunc:
            break
    (out / "report.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "case": a.case,
                "wall_seconds": time.monotonic() - started,
                "trace": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env.close()
    print(out, flush=True)
    raise SystemExit(0 if rows[-1]["success"] else 1)


if __name__ == "__main__":
    main()
