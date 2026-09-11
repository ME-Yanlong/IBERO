"""比较两个源码检查点的真实切削前缀；不声称全回合等价或允许跨源码续跑。"""

import argparse
import json
from pathlib import Path

FIELDS = (
    "initial_model_arrays",
    "initial_state_digest",
    "rollout_state_digest",
    "rollout_info_digest",
    "material_hash",
    "material_events",
    "final_model_arrays",
    "timestep_s",
    "gravity_m_s2",
    "integrator",
    "sim_seconds",
    "requested_frames",
    "completed_frames",
    "robot",
)


def compare(before, after):
    checks = {
        name: name in before and name in after and before[name] == after[name]
        for name in FIELDS
    }
    checks["frozen_complete_material_prefixes"] = all(
        row.get("frozen_source") is True
        and row.get("prefix_complete") is True
        and row.get("material_events", 0) > 0
        and not row.get("invalid_reason")
        for row in (before, after)
    )
    return dict(
        checks=checks,
        passed=all(checks.values()),
        before_source=before.get("source_hash"),
        after_source=after.get("source_hash"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "before-fixture",
        "after-fixture",
        "before-robot",
        "after-robot",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace evidence")
    rows = {}
    for kind in ("fixture", "robot"):
        before, after = [
            json.loads(getattr(args, side + "_" + kind).read_text(encoding="utf-8"))
            for side in ("before", "after")
        ]
        rows[kind] = compare(before, after)
        rows[kind]["checks"]["correct_kind"] = before.get("robot") is (
            kind == "robot"
        ) and after.get("robot") is (kind == "robot")
        rows[kind]["passed"] = all(rows[kind]["checks"].values())
    report = dict(
        scope="two_concrete_cutting_prefixes_not_full_regression",
        results=rows,
        passed=all(row["passed"] for row in rows.values()),
        inputs={k: str(v.resolve()) for k, v in vars(args).items() if k != "output"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output, report["passed"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
