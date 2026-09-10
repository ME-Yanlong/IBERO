"""G8 汇总：三形状种子分母、真实 GUI、静态包络、G7 和实际回放成本。"""

import argparse
import copy
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_passed(row, source):
    """只按逐例原始证据计数；不能信任上层复制出来的 passed 布尔值。"""
    return bool(
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
        and not row.get("artifact_errors")
    )


def raw_case(entry):
    """缺失原始文件仍保留一个失败分母，不用摘要替代原始检查。"""
    path = entry.get("report_path")
    if path and Path(path).is_file():
        return read(path)
    return dict(entry, passed=False, exception="Missing raw case report")


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
    rows, raw_defaults = [], []
    for entry in group["results"]:
        row = raw_case(entry)
        raw_defaults.append(row)
        passed = case_passed(row, source)
        rows.append(
            {
                "shape": row.get("shape"),
                "seed": row.get("seed"),
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
    base = next((r for r in raw_defaults if r.get("scene_config")), {})
    checks["default_recipe_and_failed_case_identity"] = bool(base) and all(
        r.get("scene_config") == base["scene_config"]
        and r.get("constraints") == base.get("constraints")
        and r.get("frozen_source") is True
        and r.get("manifest", {}).get("source_hash") == source
        and r.get("manifest", {}).get("seed") == r.get("seed")
        for r in raw_defaults
    )
    variants = inputs["variants"]
    checks["coefficient_variants"] = bool(
        variants["passed"] is True
        and variants["predefined_scales"] == [0.9, 1.1]
        and len(variants["results"]) == 10
        and all(
            r.get("frozen_source") is True
            and (r.get("manifest") or {}).get("source_hash") == source
            for r in variants["results"]
        )
    )
    variant_rows = []
    for entry in variants["results"]:
        row = raw_case(entry)
        label = entry.get("variant")
        scale = {"coefficients_090": 0.9, "coefficients_110": 1.1}.get(label)
        recipe_ok = False
        if base and scale is not None:
            expected = copy.deepcopy(base["scene_config"])
            coefficients = expected["process"]["coefficients"]
            for name in coefficients:
                if name not in {"source", "material_grade"}:
                    coefficients[name] *= scale
            expected["materials"]["stock"]["parameter_source"] += (
                f"; coefficient sensitivity x{scale}; not measured material data"
            )
            recipe_ok = (
                row.get("scene_config") == expected
                and row.get("constraints") == base.get("constraints")
                and row.get("manifest", {}).get("coefficients") == coefficients
                and row.get("manifest", {}).get("seed") == row.get("seed")
                and row.get("seed") == entry.get("seed")
                and row.get("shape") == "through_hole"
            )
        variant_rows.append(
            dict(
                variant=label,
                seed=row.get("seed"),
                passed=case_passed(row, source) and recipe_ok,
                recipe_matches_predefined_scale=recipe_ok,
                report_path=entry.get("report_path"),
                invalid_reason=row.get("invalid_reason"),
                exception=row.get("exception"),
            )
        )
    checks["variant_recipe_integrity"] = all(
        r["recipe_matches_predefined_scale"] for r in variant_rows
    )
    for label in ("coefficients_090", "coefficients_110"):
        selected = [r for r in variant_rows if r["variant"] == label]
        checks[label] = (
            len(selected) == 5
            and {r["seed"] for r in selected} == set(range(100, 105))
            and all(type(r["seed"]) is int for r in selected)
            and sum(r["passed"] is True for r in selected) >= 4
        )
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
        "variant_results": variant_rows,
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
    for name in ("robot-group", "g7", "preflight", "viewer", "performance", "variants"):
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
