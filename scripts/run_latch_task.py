"""S4 单场景诊断/检查记录，完整种子门槛由后续成组验收判定。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from ibero.envs.latch_release import LatchReleaseEnv
from ibero.control.latch import LatchReleaseScript


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--case", default="normal")
    p.add_argument(
        "--scene", default="latch_release", choices=["latch_release", "harness_unplug"]
    )
    a = p.parse_args()
    out = Path("artifacts/industrial_core01/latch/robot_task") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    if a.scene == "harness_unplug":
        from ibero.envs.harness_unplug import HarnessUnplugEnv
        from ibero.control.harness import HarnessUnplugScript

        env = HarnessUnplugEnv()
        policy = HarnessUnplugScript(a.case)
    else:
        env = LatchReleaseEnv()
        policy = LatchReleaseScript(a.case)
    env.reset(seed=a.seed)
    manifest = env.manifest()
    rows = []
    started = time.monotonic()
    error = None
    try:
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
                            "cable_tension_n",
                            "receiver_supported",
                        )
                    },
                    flush=True,
                )
            if term or trunc:
                break
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    (out / "report.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "case": a.case,
                "wall_seconds": time.monotonic() - started,
                "trace": rows,
                "exception": error,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env.close()
    print(out, flush=True)
    raise SystemExit(0 if not error and rows and rows[-1]["success"] else 1)


if __name__ == "__main__":
    main()
