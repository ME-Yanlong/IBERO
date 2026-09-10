"""S7 切削因果诊断；临时台架参数明确传入，不冒称完整 G7/真实材料标定。"""

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import time
from ibero.benches.milling import MillingFixture
from ibero.materials.parameters import StockParameters
from ibero.processes.tools import EndMillGeometry
from ibero.processes.milling_forces import MillingCoefficients, MillingLimits
from ibero.core.reproducibility import simulation_source_hash


def main():
    c = MillingCoefficients(
        1.8e9,
        0.6e9,
        0.2e9,
        100,
        50,
        20,
        1.8e9,
        0.4e9,
        "AISI1045-provisional",
        "provisional",
    )
    limits = MillingLimits(1000, 12000, 0.0001, 0.006, 50, 0.15, 100, 2, True)
    env = MillingFixture(
        StockParameters(size_m=(0.02, 0.016, 0.004)),
        0.001,
        EndMillGeometry(0.004, 0.012, 0.005, 0.02),
        c,
        limits,
        start=(0, 0, 0.0023),
    )
    out = Path("artifacts/industrial_core01/stock/milling_diagnostics") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    manifest = env.manifest()
    target = env.start.copy()
    started = time.perf_counter()
    for i in range(3000):
        if i > 20:
            target[2] = max(-0.0025, target[2] - 0.0002 * 0.01)
        info = env.step(target, 6000)
        rows.append(info)
        if i % 100 == 0 or env._done:
            print(
                i,
                {
                    k: info[k]
                    for k in (
                        "time_s",
                        "rpm",
                        "mode",
                        "force_world_n",
                        "total_removed_volume_m3",
                        "tip_position_m",
                        "invalid_reason",
                    )
                },
                flush=True,
            )
        if env._done:
            break
    expected = math.pi * 0.004**2 * 0.004
    simulation_source_hash.cache_clear()
    report = {
        "manifest": manifest,
        "frozen_source": manifest["source_hash"] == simulation_source_hash(),
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
        "expected_volume_m3": expected,
        "relative_volume_error": abs(
            rows[-1]["total_removed_volume_m3"] / expected - 1
        ),
        "invalid_reason": env.process.invalid_reason,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(out, "volume error", report["relative_volume_error"], flush=True)


if __name__ == "__main__":
    main()
