"""G7 局部载荷/反事实/回放正式报告；不代替三形状实际加工。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from ibero.benches.milling_validation import synthetic_load_report
from ibero.core.scene_loader import SceneLoader
from ibero.core.reproducibility import simulation_source_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    out = (
        args.output
        or ROOT
        / "artifacts/industrial_core01/stock/milling_loads"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = simulation_source_hash()
    report = {
        "scope": "G7_local_load_and_counterfactual_subset",
        "source_hash": source,
        "gate_version": "C01-IND-GATES-1",
        "passed": False,
    }
    try:
        scene = SceneLoader().validate(ROOT / "scenes/milling_bench")
        report["recipe_hash"] = scene.scene_hash
        report["loads"] = synthetic_load_report(scene)
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "pytest",
            "-q",
            "tests/test_milling_forces.py",
            "tests/test_milling_dynamics.py",
            "tests/test_milling_recipe.py",
            "tests/test_stock_collision.py",
            f"--junitxml={out / 'tests.xml'}",
        ]
        with (out / "tests.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        report["test_exit_code"] = result.returncode
        report["passed"] = report["loads"]["passed"] and result.returncode == 0
    except Exception as error:
        report["exception"] = f"{type(error).__name__}: {error}"
    simulation_source_hash.cache_clear()
    report["frozen_source"] = source == simulation_source_hash()
    report["passed"] &= report["frozen_source"]
    report["wall_seconds"] = time.perf_counter() - started
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(out, "passed", report["passed"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
