"""G7 过程台架门槛汇总；显式排除机器人、实测标定和微观切屑。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(report):
    return {
        "removed_volume_m3": report["shape_check"]["removed_volume_m3"],
        "peak_force_n": max(r["peak_cutting_force_step_n"] for r in report["rows"]),
        "peak_torque_nm": max(r["peak_spindle_torque_step_nm"] for r in report["rows"]),
        "peak_power_w": max(r["peak_spindle_power_step_w"] for r in report["rows"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", type=Path, required=True)
    p.add_argument("--loads", type=Path, required=True)
    p.add_argument("--half-step-slot", type=Path, required=True)
    p.add_argument("--half-edge-slot", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    grouped, loads = read(args.shapes), read(args.loads)
    rows = grouped["results"]
    checks = {
        "three_shapes_present": len(rows) == 3
        and {r["shape"] for r in rows} == {"slot", "pocket", "through_hole"},
        "local_loads_and_counterfactuals": loads["passed"] is True
        and loads["frozen_source"] is True,
    }
    source = loads["source_hash"]
    raw = {}
    for r in rows:
        raw[r["shape"]] = read(args.shapes.parent / r["shape"] / "report.json")
        checks[r["shape"]] = (
            r["passed"] is True
            and r["frozen_source"] is True
            and r["manifest"]["source_hash"] == source
            and r["probe_check"]["passed"]
            and r["replay_check"]["passed"]
            and not r["invalid_reason"]
        )
    sensitivity = {}
    base = metrics(raw["slot"])
    for label, path in (
        ("half_timestep", args.half_step_slot),
        ("half_edge_transition", args.half_edge_slot),
    ):
        variant = read(path)
        vm = metrics(variant)
        differences = {k: abs(vm[k] - v) / max(abs(v), 1e-15) for k, v in base.items()}
        # 附加保守 5% 数值敏感性检查，不是钢材实测精度声明。
        sensitivity[label] = {
            "path": str(path.resolve()),
            "relative_changes": differences,
            "passed": variant["passed"] is True
            and variant["frozen_source"] is True
            and variant["manifest"]["source_hash"] == source
            and max(differences.values()) <= 0.05,
        }
        checks[label] = sensitivity[label]["passed"]
    report = {
        "scope": "G7_provisional_process_fixture_not_robot_or_measured",
        "source_hash": source,
        "gate_version": "C01-IND-GATES-1",
        "checks": checks,
        "passed": all(checks.values()),
        "shape_report": str(args.shapes.resolve()),
        "load_report": str(args.loads.resolve()),
        "shapes": {
            k: {
                "metrics": metrics(v),
                "shape_check": v["shape_check"],
                "probe_passed": v["probe_check"]["passed"],
                "replay_passed": v["replay_check"]["passed"],
                "sim_seconds": v["sim_seconds"],
                "wall_seconds": v["wall_seconds"],
            }
            for k, v in raw.items()
        },
        "sensitivity": sensitivity,
        "excludes": [
            "G1_milling",
            "measured_material_calibration",
            "micro_chips",
            "thermal",
            "chatter",
            "free_offcuts",
        ],
    }
    out = (
        args.output
        or ROOT
        / "artifacts/industrial_core01/stock/stage_summary"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        / "report.json"
    )
    if out.exists():
        raise ValueError("Refusing to replace existing evidence")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(out, "passed", report["passed"], checks, flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
