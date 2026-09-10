"""G8 汇总：三形状种子分母、真实 GUI、静态包络、G7 和实际回放成本。"""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(paths):
    inputs = {key: read(path) for key, path in paths.items()}
    group, viewer = inputs["robot_group"], inputs["viewer"]
    source = inputs["g7"]["source_hash"]
    checks = {
        "G7_same_source": inputs["g7"]["passed"] is True,
        "static_preflight": inputs["preflight"]["passed"] is True
        and inputs["preflight"]["source_hash"] == source
        and inputs["preflight"]["frozen_source"] is True,
        "viewer_two_complete_episodes": viewer["passed"] is True
        and viewer.get("full") is True
        and viewer["scene"] == "plate_milling"
        and viewer["source_hash"] == source
        and viewer.get("source_unchanged_during_run") is True
        and all(
            viewer.get(k) is True
            for k in (
                "successful_episode_1",
                "successful_episode_2",
                "persistent_finish",
                "initial_wait",
                "pause",
                "single_step",
                "previous_frame",
                "history_does_not_integrate",
                "replay",
                "rerun",
            )
        ),
        "performance_and_live_isolation": inputs["performance"]["passed"] is True
        and inputs["performance"]["source_hash"] == source
        and inputs["performance"]["frozen_source"] is True
        and inputs["performance"]["events"] > 0,
        "group_report_passed": group["passed"] is True,
    }
    rows = []
    for entry in group["results"]:
        row = read(entry["report_path"]) if entry.get("report_path") else entry
        passed = bool(
            row.get("passed") is True
            and row.get("frozen_source") is True
            and row.get("manifest", {}).get("source_hash") == source
            and row.get("manifest", {}).get("scene_kind") == "plate_milling"
            and row.get("task_check", {}).get("result", {}).get("success") is True
            and row.get("shape_check", {}).get("passed") is True
            and row.get("probe_check", {}).get("passed") is True
            and row.get("replay_check", {}).get("passed") is True
            and row.get("controller_finished") is True
            and not row.get("invalid_reason")
            and not row.get("exception")
        )
        rows.append(
            {
                "shape": row["shape"],
                "seed": row["seed"],
                "passed": passed,
                "report_path": entry.get("report_path"),
                "invalid_reason": row.get("invalid_reason"),
                "exception": row.get("exception"),
                "sim_seconds": row.get("sim_seconds"),
                "wall_seconds": row.get("wall_seconds"),
                "real_time_factor": row.get("real_time_factor"),
            }
        )
    counts = {}
    checks["exactly_thirty_cases"] = len(rows) == 30
    for shape in ("slot", "pocket", "through_hole"):
        selected = [r for r in rows if r["shape"] == shape]
        count = sum(r["passed"] for r in selected)
        counts[shape] = {"runs": len(selected), "successes": count, "required": 9}
        checks[shape] = (
            len(selected) == 10
            and {r["seed"] for r in selected} == set(range(10))
            and all(type(r["seed"]) is int for r in selected)
            and count >= 9
        )
    return {
        "scope": "G8_robot_milling_provisional_engineering_not_real_calibration",
        "gate_version": "C01-IND-GATES-1",
        "source_hash": source,
        "checks": checks,
        "passed": all(checks.values()),
        "counts": counts,
        "inputs": {k: str(Path(v).resolve()) for k, v in paths.items()},
        "results": rows,
        "performance_target_is_not_physics_gate": True,
        "excludes": [
            "real_material_calibration",
            "free_offcuts",
            "five_axis",
            "real_hardware",
            "G9",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("robot-group", "g7", "preflight", "viewer", "performance"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to replace existing evidence")
    report = summarize({k: v for k, v in vars(args).items() if k != "output"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output, report["passed"], report["checks"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
