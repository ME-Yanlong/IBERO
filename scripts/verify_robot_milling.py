"""G8 机器人加工三形状×十种子；每例独立原始证据，失败不删、不补种子。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from verify_milling_process import run_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("scenes/plate_milling"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or Path(
        "artifacts/industrial_core01/stock/robot_validation"
    ) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    shapes = ("slot", "pocket", "through_hole")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_shape, (shape, str(out / f"seed_{seed:02d}"), str(args.scene), seed)
            ): (shape, seed)
            for seed in range(10)
            for shape in shapes
        }
        for future in as_completed(futures):
            shape, seed = futures[future]
            try:
                raw = future.result()
                row = {
                    k: raw.get(k)
                    for k in (
                        "shape",
                        "seed",
                        "passed",
                        "manifest",
                        "frozen_source",
                        "invalid_reason",
                        "exception",
                        "wall_seconds",
                        "sim_seconds",
                        "real_time_factor",
                    )
                }
                row["report_path"] = str(
                    out / f"seed_{seed:02d}" / shape / "report.json"
                )
                row["task_passed"] = bool(
                    raw.get("task_check", {}).get("result", {}).get("success")
                )
            except Exception as error:
                row = dict(
                    shape=shape,
                    seed=seed,
                    passed=False,
                    exception=f"{type(error).__name__}: {error}",
                )
            rows.append(row)
            (out / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                shape,
                seed,
                row["passed"],
                row.get("invalid_reason"),
                row.get("exception"),
                flush=True,
            )
    checks = {
        shape: sum(
            bool(r["passed"] and r.get("task_passed"))
            for r in rows
            if r["shape"] == shape
        )
        >= 9
        for shape in shapes
    }
    checks["thirty_cases"] = len(rows) == 30
    checks["same_frozen_source"] = len(
        {(r.get("manifest") or {}).get("source_hash") for r in rows}
    ) == 1 and all(r.get("frozen_source") for r in rows)
    result = dict(
        scope="G8_robot_30_cases_only_viewer_and_other_gates_separate",
        checks=checks,
        passed=all(checks.values()),
        results=rows,
    )
    (out / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
