"""独立自由动态探针：去除前阻挡、孔中通过、邻近剩余实体阻挡。"""

import time
import mujoco
import numpy as np
from ibero.materials.parameters import StockParameters
from ibero.materials.stock import VoxelStock
from ibero.materials.stock_collision import add_stock_geoms, StockCollisionBinding
from ibero.processes.tools import EndMillGeometry, ToolPose, swept_cells


def probe_spec(stock, *, active_only=False):
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.0002
    spec.option.gravity = [0, 0, 0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    add_stock_geoms(spec, stock, active_only=active_only)
    probe = spec.worldbody.add_body(name="probe", pos=[0, 0, 0.035])
    probe.add_freejoint(name="probe_free")
    probe.add_geom(
        name="probe_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.002],
        mass=0.02,
        rgba=[0.9, 0.3, 0.1, 1],
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
    )
    spec.worldbody.add_light(pos=[0.1, -0.1, 0.2])
    return spec


def run_probe(model, data, x=0):
    """只在 reset 设置初速度，之后无 qpos 写入、无恒定速度运动学作弊。"""
    mujoco.mj_resetData(model, data)
    data.joint("probe_free").qpos[0] = x
    data.joint("probe_free").qvel[2] = -0.15
    mujoco.mj_forward(model, data)
    rows = []
    start = time.monotonic()
    for _ in range(100):
        mujoco.mj_step(model, data, nstep=20)
        mujoco.mj_forward(model, data)
        rows.append(
            {
                "time_s": float(data.time),
                "z_m": float(data.body("probe").xpos[2]),
                "velocity_z_m_s": float(data.joint("probe_free").qvel[2]),
                "contacts": data.ncon,
            }
        )
    return {
        "x_m": x,
        "final_z_m": rows[-1]["z_m"],
        "minimum_z_m": min(r["z_m"] for r in rows),
        "ever_contact": any(r["contacts"] for r in rows),
        "wall_seconds": time.monotonic() - start,
        "trace": rows,
    }


def probe_benchmark():
    stock = VoxelStock(StockParameters(), 0.001)
    start = time.monotonic()
    model = probe_spec(stock).compile()
    preallocated_compile_s = time.monotonic() - start
    data = mujoco.MjData(model)
    binding = StockCollisionBinding(model, stock)
    before = run_probe(model, data)
    tool = EndMillGeometry(0.008, 0.02, 0.01, 0.03)
    pose = ToolPose((0, 0, -0.005))
    event = stock.prepare_removal(
        swept_cells(stock, tool, pose, pose), "geometry-only-hole"
    )
    start = time.monotonic()
    binding.commit(event, data)
    commit_s = time.monotonic() - start
    through = run_probe(model, data)
    neighbor = run_probe(model, data, x=0.016)
    start = time.monotonic()
    rebuilt = probe_spec(stock, active_only=True).compile()
    recompile_s = time.monotonic() - start
    alternative = run_probe(rebuilt, mujoco.MjData(rebuilt))
    binding.reset(data)
    reset = run_probe(model, data)
    checks = {
        "before_blocked": before["minimum_z_m"] > 0.0065 and before["ever_contact"],
        "hole_passed": through["final_z_m"] < -0.009 and not through["ever_contact"],
        "neighbor_blocked": neighbor["minimum_z_m"] > 0.0065
        and neighbor["ever_contact"],
        "reset_blocked": reset["minimum_z_m"] > 0.0065 and reset["ever_contact"],
        "recompile_agrees": np.isclose(
            alternative["final_z_m"], through["final_z_m"], atol=1e-10
        ),
        "probe_clearance": (0.008 - 0.002) >= 3 * stock.cell_size_m,
    }
    return {
        "checks": {k: bool(v) for k, v in checks.items()},
        "passed": bool(all(checks.values())),
        "preallocated_compile_s": preallocated_compile_s,
        "commit_s": commit_s,
        "controlled_recompile_s": recompile_s,
        "preallocated_geoms": model.ngeom,
        "recompiled_geoms": rebuilt.ngeom,
        "cases": {
            "before": before,
            "through": through,
            "neighbor": neighbor,
            "reset": reset,
            "recompiled": alternative,
        },
    }
