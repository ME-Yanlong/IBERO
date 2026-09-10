"""三档实际加工台架的独立补核；不以纯几何或不同控制菜谱替代。"""

import argparse
import copy
import json
from pathlib import Path

from summarize_milling_stage import case_passed, metrics, read


def summarize(paths, *, medium_cell=0.00075):
    if medium_cell not in {0.00075, 0.002 / 3}:
        raise ValueError("Use a declared grid supplement, not an arbitrary replacement")
    levels = (("base", 0.001), ("medium", medium_cell), ("fine", 0.0005))
    shapes = ("slot", "pocket", "through_hole")
    raw, checks = {}, {}
    for label, _ in levels:
        group = read(paths[label])
        rows = group.get("results", [])
        checks[label + "_complete_group"] = (
            len(rows) == 3
            and {row.get("shape") for row in rows} == set(shapes)
            and group.get("passed") is True
        )
        for shape in shapes:
            path = paths[label].parent / shape / "report.json"
            raw[label, shape] = read(path) if path.is_file() else {}
    base = raw["base", "slot"]
    source = base.get("manifest", {}).get("source_hash")
    cfg = base.get("scene_config")
    checks["valid_base_recipe"] = bool(
        source
        and cfg
        and cfg.get("kind") == "milling_bench"
        and cfg.get("numerics", {}).get("cell_size_m") == 0.001
    )
    results = []
    for label, cell in levels:
        expected = copy.deepcopy(cfg)
        if expected:
            expected["numerics"]["cell_size_m"] = cell
        for shape in shapes:
            row = raw[label, shape]
            identity = bool(
                expected
                and row.get("scene_config") == expected
                and row.get("constraints") == base.get("constraints")
                and row.get("shape") == shape
                and row.get("seed") == base.get("seed")
                and row.get("manifest", {}).get("seed") == row.get("seed")
            )
            passed = identity and case_passed(row, source)
            checks[label + "_" + shape] = passed
            # 缺少指标也不虚构为零；原失败与不完整原始文件均明确留在分母。
            try:
                measured = metrics(row)
            except (KeyError, TypeError, ValueError):
                measured = None
                checks[label + "_" + shape] = False
            results.append(
                dict(
                    level=label,
                    cell_size_m=cell,
                    shape=shape,
                    passed=checks[label + "_" + shape],
                    recipe_changes_only_cell_size=identity,
                    metrics=measured,
                    shape_check=row.get("shape_check"),
                    invalid_reason=row.get("invalid_reason"),
                    exception=row.get("exception"),
                    report_path=str(
                        (paths[label].parent / shape / "report.json").resolve()
                    ),
                )
            )
    changes = []
    for shape in shapes:
        selected = [r for r in results if r["shape"] == shape]
        for a, b in zip(selected[:-1], selected[1:]):
            if a["metrics"] and b["metrics"]:
                changes.append(
                    dict(
                        shape=shape,
                        from_cell_size_m=a["cell_size_m"],
                        to_cell_size_m=b["cell_size_m"],
                        relative_to_finer={
                            k: abs(a["metrics"][k] - v) / max(abs(v), 1e-15)
                            for k, v in b["metrics"].items()
                        },
                    )
                )
    return dict(
        scope="three_resolution_physical_fixture_supplement_not_robot_or_calibration",
        grid_set="original075"
        if medium_cell == 0.00075
        else "additional_depth_aligned0667",
        source_hash=source,
        passed=all(checks.values()),
        checks=checks,
        results=results,
        adjacent_grid_changes=changes,
        note="Per-case shape/probe/replay gates apply. Force-grid differences are reported, not silently declared converged.",
        inputs={k: str(v.resolve()) for k, v in paths.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "medium", "fine", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--grid-set", choices=["original075", "additional0667"], default="original075"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace existing evidence")
    result = summarize(
        {k: getattr(args, k) for k in ("base", "medium", "fine")},
        medium_cell=0.00075 if args.grid_set == "original075" else 0.002 / 3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output, result["passed"], result["checks"], flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
