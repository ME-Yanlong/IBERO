"""独立自由动态探针：去除前阻挡、孔中通过、邻近剩余实体阻挡。"""

import time
import mujoco
import numpy as np
from ibero.materials.parameters import StockParameters
from ibero.materials.parameters import finite_number
from ibero.materials.stock import VoxelStock
from ibero.materials.stock_collision import add_stock_geoms, StockCollisionBinding
from ibero.processes.tools import EndMillGeometry, ToolPose, swept_cells


def probe_spec(stock, *, active_only=False, radius_m=0.002):
    finite_number(radius_m, "probe radius_m")
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
        size=[radius_m],
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
    start = time.perf_counter()
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
        "wall_seconds": time.perf_counter() - start,
        "trace": rows,
    }


def probe_benchmark():
    stock = VoxelStock(StockParameters(), 0.001)
    start = time.perf_counter()
    model = probe_spec(stock).compile()
    preallocated_compile_s = time.perf_counter() - start
    data = mujoco.MjData(model)
    binding = StockCollisionBinding(model, stock)
    before = run_probe(model, data)
    tool = EndMillGeometry(0.008, 0.02, 0.01, 0.03)
    pose = ToolPose((0, 0, -0.005))
    event = stock.prepare_removal(
        swept_cells(stock, tool, pose, pose), "geometry-only-hole"
    )
    start = time.perf_counter()
    binding.commit(event, data)
    commit_s = time.perf_counter() - start
    through = run_probe(model, data)
    neighbor = run_probe(model, data, x=0.016)
    start = time.perf_counter()
    rebuilt = probe_spec(stock, active_only=True).compile()
    recompile_s = time.perf_counter() - start
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


def inspect_machined_stock(stock, target):
    """将实际加工占据传给独立动力学探针，不从目标形状重建理想孔。

    这是检查台架的受控模型重建，不是原机器人场景中的在线探针操作。
    毛坯坐标/姿态受限，避免把世界 Z 探针误用于旋转工件。
    """
    if (
        not np.allclose(stock.origin, 0)
        or not np.allclose(stock.rotation, np.eye(3))
        or target.center_xy_m != (0, 0)
    ):
        raise ValueError("Probe inspection supports the declared centered fixture only")
    radius = 0.00075
    start = time.perf_counter()
    model = probe_spec(stock, radius_m=radius).compile()
    data = mujoco.MjData(model)
    binding = StockCollisionBinding(model, stock)
    binding.ensure_consistent()
    center = run_probe(model, data)
    neighbor = run_probe(model, data, x=stock.params.size_m[0] / 2 - 2 * radius)
    bottom = stock.params.size_m[2] / 2 - target.depth_m
    center_ok = (
        (
            not center["ever_contact"]
            and center["final_z_m"] < -stock.params.size_m[2] / 2 - radius
        )
        if target.shape == "through_hole"
        else (center["ever_contact"] and center["minimum_z_m"] > bottom + radius * 0.7)
    )
    checks = {
        "actual_hole_or_floor": bool(center_ok),
        "remaining_neighbor_blocks": bool(
            neighbor["ever_contact"]
            and neighbor["minimum_z_m"] > stock.params.size_m[2] / 2 + radius * 0.7
        ),
        "radial_clearance": target.corner_radius_m - radius >= 3 * stock.cell_size_m,
    }
    return {
        "method": "independent_dynamic_probe_from_actual_occupancy",
        "stock_state_hash": stock.state_hash(),
        "radius_m": radius,
        "clearance_cells": (target.corner_radius_m - radius) / stock.cell_size_m,
        "checks": checks,
        "passed": all(checks.values()),
        "cases": {"center": center, "neighbor": neighbor},
        "wall_seconds": time.perf_counter() - start,
    }
