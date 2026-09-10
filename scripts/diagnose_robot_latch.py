"""自由插头接触研发记录；诊断命令不等于 G4 验收。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from PIL import Image
import mujoco
from ibero.benches.robot_latch import RobotLatchDiagnostic
from ibero.core.reproducibility import simulation_source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-press", action="store_true")
    args = parser.parse_args()
    out = Path("artifacts/industrial_core01/latch/robot_diagnostics") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    sim = RobotLatchDiagnostic()
    rows = []
    started = time.monotonic()
    press_start = sim.left_target.copy()
    pull_start = sim.right_target.copy()
    for k in range(450):
        t = k * 0.01
        if t > 1.6 and not args.no_press:
            sim.left_target[2] = press_start[2] - 0.0165 * min(1, (t - 1.6) / 1)
        if t > 3:
            sim.right_target[1] = pull_start[1] - 0.020 * min(1, (t - 3) / 1.2)
        row = sim.step(grip=-1 if t < 0.7 else 1)
        row["contacts"] = [
            (sim.model.geom(c.geom1).name, sim.model.geom(c.geom2).name, float(c.dist))
            for c in sim.data.contact
        ]
        rows.append(row)
        if k % 50 == 0:
            print(k, row, "right_tcp", sim.data.site("right_pinch").xpos, flush=True)
        if row["released"] or row["invalid_reason"]:
            break
    renderer = mujoco.Renderer(sim.model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = sim.origin
    camera.distance = 0.35
    camera.azimuth = 135
    camera.elevation = -25
    renderer.update_scene(sim.data, camera=camera)
    Image.fromarray(renderer.render()).save(out / "final.png")
    renderer.close()
    (out / "report.json").write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "no_press": args.no_press,
                "source_hash": simulation_source_hash(),
                "wall_seconds": time.monotonic() - started,
                "trace": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out, rows[-1], flush=True)


if __name__ == "__main__":
    main()
