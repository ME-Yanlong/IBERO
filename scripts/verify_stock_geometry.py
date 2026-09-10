"""G5 几何台架入口；通过不代表 G6 碰撞或 G7 工艺已经通过。"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import yaml
from ibero.benches.stock_geometry import benchmark_shape, section_image
from ibero.core.reproducibility import simulation_source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (
        args.output
        or root
        / "artifacts/industrial_core01/stock/geometry"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    gates = yaml.safe_load(
        (root / "validation/industrial_gates.yaml").read_text(encoding="utf-8")
    )
    source = simulation_source_hash()
    rows = []
    for shape in ("slot", "pocket", "through_hole"):
        for cell in gates["stock"]["resolutions_m"]:
            started = time.monotonic()
            stock, row = benchmark_shape(shape, cell)
            row["wall_seconds"] = time.monotonic() - started
            row["passed"] = (
                row["volume_relative_error"] <= gates["stock"]["volume_relative_error"]
                and row["boundary_error_cells"]
                <= gates["stock"]["boundary_error_cells"]
            )
            section_image(stock, output / f"{shape}_{cell}.png")
            rows.append(row)
            print(row, flush=True)
    simulation_source_hash.cache_clear()
    frozen = source == simulation_source_hash()
    report = {
        "scope": "geometry_only",
        "excludes": ["collision", "cutting_loads", "robot_milling"],
        "source_hash": source,
        "frozen_source": frozen,
        "gate_version": gates["version"],
        "results": rows,
        "passed": frozen and all(r["passed"] for r in rows),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output, "passed", report["passed"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
