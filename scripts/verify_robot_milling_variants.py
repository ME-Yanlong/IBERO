"""额外留出：端铣系数 ±10%，各 seed 100—104 完整通孔；非实测材料泛化。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import shutil

import yaml
from ibero.core.scene_loader import SceneLoader
from verify_milling_process import run_shape


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, default=Path("scenes/plate_milling"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = p.parse_args()
    base = SceneLoader().validate(args.scene)
    if base.config["kind"] != "plate_milling":
        p.error("Robot material variants require plate_milling")
    args.output.mkdir(parents=True, exist_ok=False)
    recipes = {}
    for label, scale in (("coefficients_090", 0.9), ("coefficients_110", 1.1)):
        root = args.output / label / "resolved_scene"
        root.mkdir(parents=True, exist_ok=False)
        cfg = copy.deepcopy(base.config)
        coefficients = cfg["process"]["coefficients"]
        for name in coefficients:
            if name not in {"material_grade", "source"}:
                coefficients[name] *= scale
        cfg["materials"]["stock"]["parameter_source"] += (
            f"; coefficient sensitivity x{scale}; not measured material data"
        )
        (root / "scene_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        for name in ("constraints.yaml", "task_spec.py"):
            shutil.copyfile(args.scene / name, root / name)
        SceneLoader().validate(root)
        recipes[label] = root
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_shape,
                (
                    "through_hole",
                    str(args.output / label / f"seed_{seed}"),
                    str(root),
                    seed,
                ),
            ): (label, seed)
            for label, root in recipes.items()
            for seed in range(100, 105)
        }
        for future in as_completed(futures):
            label, seed = futures[future]
            try:
                raw = future.result()
                row = {
                    k: raw.get(k)
                    for k in (
                        "passed",
                        "invalid_reason",
                        "exception",
                        "manifest",
                        "frozen_source",
                        "sim_seconds",
                        "wall_seconds",
                        "real_time_factor",
                    )
                }
                row["passed"] = bool(
                    row["passed"] and raw["task_check"]["result"]["success"]
                )
            except Exception as error:
                row = dict(passed=False, exception=f"{type(error).__name__}: {error}")
            row.update(
                variant=label,
                seed=seed,
                report_path=str(
                    args.output / label / f"seed_{seed}" / "through_hole/report.json"
                ),
            )
            rows.append(row)
            (args.output / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                label,
                seed,
                row["passed"],
                row.get("invalid_reason"),
                row.get("exception"),
                flush=True,
            )
    checks = {
        label: sum(r["passed"] for r in rows if r["variant"] == label) >= 4
        for label in recipes
    }
    checks["ten_cases"] = len(rows) == 10
    checks["same_frozen_source"] = len(
        {(r.get("manifest") or {}).get("source_hash") for r in rows}
    ) == 1 and all(r.get("frozen_source") for r in rows)
    result = dict(
        scope="held_out_coefficient_sensitivity_not_real_material_generalization",
        predefined_scales=[0.9, 1.1],
        seeds=list(range(100, 105)),
        shape="through_hole",
        unchanged=[
            "robot",
            "control",
            "capacity_limits",
            "numerics",
            "geometry",
            "target",
        ],
        checks=checks,
        passed=all(checks.values()),
        results=rows,
    )
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
