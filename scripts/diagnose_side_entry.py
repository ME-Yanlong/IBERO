"""低速侧向入刀最小复现：显式预制孔初态，不冒称完整加工/回放验收。"""

from datetime import datetime, timezone
from pathlib import Path
import json
import mujoco
import numpy as np
from ibero.benches.milling import MillingFixture
from ibero.core.scene_loader import SceneLoader
from ibero.materials.parameters import StockParameters, strict_parameters
from ibero.processes.tools import EndMillGeometry, ToolPose, swept_cells
from ibero.processes.milling_forces import MillingCoefficients, MillingLimits
from ibero.core.reproducibility import simulation_source_hash


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = SceneLoader().validate(root / "scenes/milling_bench").config
    e = MillingFixture(
        strict_parameters(StockParameters, cfg["materials"]["stock"]),
        0.001,
        strict_parameters(EndMillGeometry, cfg["tool"]),
        strict_parameters(MillingCoefficients, cfg["process"]["coefficients"]),
        strict_parameters(MillingLimits, cfg["process"]["limits"]),
        start=(-0.004, 0, 0.0001),
        axis_stiffness_n_m=cfg["machine"]["axis_stiffness_n_m"],
        axis_damping_ns_m=cfg["machine"]["axis_damping_ns_m"],
        axis_force_limit_n=cfg["machine"]["axis_force_limit_n"],
    )
    pose = ToolPose((-0.004, 0, 0))
    e.binding.commit(
        e.stock.prepare_removal(
            swept_cells(e.stock, e.tool, pose, pose), "diagnostic-precut-initial-state"
        ),
        e.data,
    )
    e.data.qvel[2] = -3e-5
    e.data.joint("mill_spindle").qvel[0] = 5978 * np.pi / 30
    mujoco.mj_forward(e.model, e.data)
    manifest = e.manifest()
    rows = []
    for i in range(200):
        target = (-0.004 + (i + 1) * 0.001 * 0.01, 0, 0)
        info = e.step(target, 6000)
        rows.append(info)
        if i % 20 == 0 or e._done:
            print(
                i,
                info["invalid_reason"],
                info["tip_position_m"],
                info["force_world_n"],
                flush=True,
            )
        if e._done:
            break
    out = (
        root
        / "artifacts/industrial_core01/stock/side_entry"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    out.mkdir(parents=True, exist_ok=False)
    simulation_source_hash.cache_clear()
    report = {
        "scope": "precut_diagnostic_only",
        "manifest": manifest,
        "rows": rows,
        "frozen_source": manifest["source_hash"] == simulation_source_hash(),
        "passed": not e._done,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(out, "passed", report["passed"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
