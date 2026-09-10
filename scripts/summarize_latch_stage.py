"""汇总 G4 分阶段冻结证据，保留被替代的错误动作报告；不是当前源码全量回归。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_groups(defaults, faults, scene):
    """严格核对分母和用例身份，不能把少跑、重复种子或失败重试改成更好分母。"""
    checks = {
        "normal_source_frozen": defaults.get("checks", {}).get("frozen_source") is True,
        "fault_source_frozen": faults.get("checks", {}).get("frozen_source") is True,
    }
    summaries = {}
    for variant, seeds, minimum in (
        (scene, range(20), 18),
        (scene + "_thinner", range(100, 105), 4),
        (scene + "_shallower", range(100, 105), 4),
    ):
        rows = [
            r
            for r in defaults["results"]
            if r["scene"] == variant and r["case"] == "normal"
        ]
        identities = [r["seed"] for r in rows]
        successes = sum(
            r.get("passed") is True
            and r.get("success") is True
            and not r.get("exception")
            for r in rows
        )
        checks[variant] = sorted(identities) == list(seeds) and successes >= minimum
        summaries[variant] = {
            "runs": len(rows),
            "successes": successes,
            "minimum": minimum,
            "seeds": identities,
        }
    expected = {"no_press", "partial_press", "offset_press", "early_release"}
    rows = faults["results"]
    checks["corrected_counterfactuals"] = (
        len(rows) == 4
        and {r["case"] for r in rows} == expected
        and all(
            r["scene"] == scene
            and r["seed"] == 0
            and r.get("passed") is True
            and r.get("success") is False
            and not r.get("exception")
            for r in rows
        )
    )
    return {"checks": checks, "groups": summaries, "passed": all(checks.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "latch-defaults",
        "harness-defaults",
        "latch-faults",
        "harness-faults",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    sources = {}
    for label in (
        "latch_defaults",
        "harness_defaults",
        "latch_faults",
        "harness_faults",
    ):
        path = getattr(args, label)
        sources[label] = {
            "path": str(path.resolve()),
            "report": json.loads(path.read_text(encoding="utf-8")),
        }
    report = {
        "scope": "G4_multiversion_milestone_not_final_current_source_regression",
        "replacement_reason": "Original early_release unloaded after the retention face; corrected trials unload before crossing it. All original reports remain intact (development log 18.10).",
        "sources": {
            k: {
                "path": v["path"],
                "source_hash": v["report"]["source_hash"],
                "original_checks": v["report"].get("checks", {}),
            }
            for k, v in sources.items()
        },
        "scenes": {},
    }
    for prefix, scene in (("latch", "latch_release"), ("harness", "harness_unplug")):
        report["scenes"][scene] = validate_groups(
            sources[prefix + "_defaults"]["report"],
            sources[prefix + "_faults"]["report"],
            scene,
        )
    report["passed"] = all(v["passed"] for v in report["scenes"].values())
    output = (
        args.output
        or ROOT
        / "artifacts/industrial_core01/latch/stage_summary"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        / "report.json"
    )
    if output.exists():
        raise ValueError("Refusing to replace existing evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(output, "passed", report["passed"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
