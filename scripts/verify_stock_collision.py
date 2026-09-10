"""G6 真实动态探针＋事务/回放测试；原始输出与失败均保留。"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
import time
from ibero.benches.stock_probe import probe_benchmark
from ibero.core.reproducibility import simulation_source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    out = (
        args.output
        or args.project_root
        / "artifacts/industrial_core01/stock/collision"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = simulation_source_hash()
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "pytest",
        "-q",
        "tests/test_stock_collision.py",
        "tests/test_stock_geometry.py",
    ]
    with (out / "tests.log").open("w", encoding="utf-8") as log:
        tests = subprocess.run(
            command,
            cwd=args.project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print("Transaction/replay tests exit", tests.returncode, flush=True)
    probe = probe_benchmark()
    simulation_source_hash.cache_clear()
    report = {
        "source_hash": source,
        "frozen_source": source == simulation_source_hash(),
        "gate_version": "C01-IND-GATES-1",
        "scope": "stock_collision_and_material_replay_only",
        "test_command": command,
        "test_exit_code": tests.returncode,
        "probe": probe,
        "wall_seconds": time.monotonic() - started,
    }
    report["passed"] = (
        report["frozen_source"] and tests.returncode == 0 and probe["passed"]
    )
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(out, "passed", report["passed"], probe["checks"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
