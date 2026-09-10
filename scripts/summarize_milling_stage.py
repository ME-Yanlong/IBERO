"""G7 过程台架门槛汇总；显式排除机器人、实测标定和微观切屑。"""

import argparse
import copy
import hashlib
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


def case_passed(row, source):
    return bool(
        row.get("passed") is True
        and row.get("frozen_source") is True
        and row.get("manifest", {}).get("source_hash") == source
        and row.get("probe_check", {}).get("passed") is True
        and row.get("replay_check", {}).get("passed") is True
        and row.get("shape_check", {}).get("passed") is True
        and row.get("controller_finished") is True
        and not row.get("invalid_reason")
        and not row.get("exception")
        and not row.get("artifact_errors")
    )


def summarize(args):
    attestations_path = getattr(args, "recipe_attestations", None)
    attestations = read(attestations_path)["results"] if attestations_path else []

    def read_case(path):
        row = read(path)
        if row.get("scene_config"):
            return row
        # 旧报告只有不可逆身份 hash；可另附原冻结环境的只读核验，不盲填当前菜谱。
        for evidence in attestations:
            if (
                evidence.get("passed") is True
                and evidence.get("report_path") == str(path.resolve())
                and evidence.get("report_sha256")
                == hashlib.sha256(path.read_bytes()).hexdigest()
                and evidence.get("manifest") == row.get("manifest")
                and evidence.get("original_result_unchanged") is True
                and evidence.get("physical_steps_executed") == 0
            ):
                return dict(
                    row,
                    scene_config=evidence["scene_config"],
                    constraints=evidence["constraints"],
                )
        return row

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
        raw[r["shape"]] = read_case(args.shapes.parent / r["shape"] / "report.json")
        # 上层仅给文件索引；计分以重新读取的原始形状/探针/回放证据为准。
        checks[r["shape"]] = case_passed(raw[r["shape"]], source)
    sensitivity = {}
    base = metrics(raw["slot"])
    for label, path in (
        ("half_timestep", args.half_step_slot),
        ("half_edge_transition", args.half_edge_slot),
    ):
        variant = read_case(path)
        vm = metrics(variant)
        differences = {k: abs(vm[k] - v) / max(abs(v), 1e-15) for k, v in base.items()}
        expected = copy.deepcopy(raw["slot"].get("scene_config"))
        recipe_ok = False
        if expected:
            section, field = (
                ("physics", "timestep_s")
                if label == "half_timestep"
                else ("numerics", "edge_transition_chip_m")
            )
            expected[section][field] /= 2
            recipe_ok = (
                variant.get("scene_config") == expected
                and variant.get("constraints") == raw["slot"].get("constraints")
                and variant.get("shape") == "slot"
                and variant.get("seed") == raw["slot"].get("seed")
            )
        # 附加保守 5% 数值敏感性检查，不是钢材实测精度声明。
        sensitivity[label] = {
            "path": str(path.resolve()),
            "relative_changes": differences,
            "recipe_changes_only_requested_numerical_parameter": recipe_ok,
            "passed": case_passed(variant, source)
            and recipe_ok
            and max(differences.values()) <= 0.05,
        }
        checks[label] = sensitivity[label]["passed"]
    return {
        "scope": "G7_provisional_process_fixture_not_robot_or_measured",
        "source_hash": source,
        "gate_version": "C01-IND-GATES-1",
        "checks": checks,
        "passed": all(checks.values()),
        "shape_report": str(args.shapes.resolve()),
        "load_report": str(args.loads.resolve()),
        "recipe_attestations": str(attestations_path.resolve())
        if attestations_path
        else None,
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", type=Path, required=True)
    p.add_argument("--loads", type=Path, required=True)
    p.add_argument("--half-step-slot", type=Path, required=True)
    p.add_argument("--half-edge-slot", type=Path, required=True)
    p.add_argument(
        "--recipe-attestations",
        type=Path,
        help="仅用于缺少完整菜谱的旧报告，需原冻结源码严格核验的附加证明",
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = summarize(args)
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
    print(out, "passed", report["passed"], report["checks"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
