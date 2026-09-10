"""G1 左主轴/右安全停放构型；沿用原资产关节范围与执行器能力。"""

import mujoco
import numpy as np
from ibero.robots.g1_industrial import robot_spec, IndustrialRobotHandles
from ibero.robots.g1_upperbody import ARM_JOINTS
from ibero.benches.milling import add_spindle


def milling_robot_spec(
    tool,
    limits,
    *,
    timestep=0.0001,
    tip_offset_m=(0.16, 0, -0.06),
    bracket_mass_kg=0.08,
):
    spec = robot_spec(timestep=timestep)
    # 复用已验收的 G1 裁剪与右夹爪；左按压工具整件卸下，不能叠装主轴。
    spec.delete(spec.body("left_press_tool"))
    offset = np.asarray(tip_offset_m, dtype=float)
    if (
        offset.shape != (3,)
        or not np.isfinite(offset).all()
        or not np.isfinite(bracket_mass_kg)
        or bracket_mass_kg <= 0
    ):
        raise ValueError("Finite spindle mount geometry/mass required")
    wrist = spec.body("left_wrist_yaw_link")
    bracket = wrist.add_body(name="left_spindle_bracket")
    top = offset + [0, 0, tool.cutting_length_m + tool.shank_length_m]
    elbow = np.array([offset[0], offset[1], 0])
    for name, start, end in (("forward", [0.045, 0, 0], elbow), ("down", elbow, top)):
        if np.linalg.norm(np.asarray(end) - start) < 0.001:
            raise ValueError("Spindle bracket link is degenerate")
        bracket.add_geom(
            name="left_spindle_bracket_" + name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=np.r_[start, end],
            size=[0.006],
            mass=bracket_mass_kg / 2,
            rgba=[0.25, 0.5, 0.6, 1],
        )
    frame = bracket.add_body(name="left_spindle_tip_frame", pos=offset)
    add_spindle(spec, frame, tool, torque_limit_nm=limits.max_torque_nm)
    return spec


def milling_robot_handles(model):
    arms = np.array([model.actuator(n).id for n in ARM_JOINTS], dtype=int)
    grips = np.array([model.actuator("right_fingers_actuator").id], dtype=int)
    spindle = model.actuator("mill_motor").id
    held = np.array(
        [i for i in range(model.nu) if i not in set(arms) | set(grips) | {spindle}],
        dtype=int,
    )
    return IndustrialRobotHandles(
        arms, held, grips, model.site("mill_tip").id, model.site("right_pinch").id
    )


def add_plate_support(spec, *, origin, half_size):
    """固定毛坯边缘支撑，中间留空，贯通后刀具不能碰到一块隐藏实心垫板。"""
    origin, half = np.asarray(origin), np.asarray(half_size)
    # 支撑只占两侧外缘 1 mm，供首版 24 mm 工件；实际工件仍是声明的世界固定毛坯。
    for side in (-1, 1):
        spec.worldbody.add_geom(
            name="plate_edge_support_" + str(side),
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=origin + [side * (half[0] - 0.0005), 0, -half[2] - 0.01],
            size=[0.0005, half[1], 0.01],
            rgba=[0.28, 0.32, 0.36, 1],
        )
    spec.worldbody.add_geom(
        name="plate_support_base",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=origin + [0, 0, -half[2] - 0.024],
        size=[half[0] + 0.008, half[1] + 0.008, 0.004],
        rgba=[0.18, 0.2, 0.24, 1],
    )
