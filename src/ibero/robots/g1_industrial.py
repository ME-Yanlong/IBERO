"""工业场景的最小 G1 工具组合；保持原资产关节能力，不复制线束环境。"""

from dataclasses import dataclass
import mujoco
import numpy as np

from ibero.robots.g1_upperbody import ARM_JOINTS, G1_XML, _spec_from_snapshot
from ibero.robots.g1_upperbody_2f85 import (
    _remove_native_hand_visuals,
    _add_robotiq_gripper,
    _add_wrist_interfaces,
)


@dataclass(frozen=True)
class IndustrialRobotHandles:
    arm_actuator_ids: np.ndarray
    hold_actuator_ids: np.ndarray
    gripper_actuator_ids: np.ndarray
    left_anchor_site_id: int
    right_anchor_site_id: int


def robot_spec(
    *,
    timestep=0.0000025,
    press_tcp_offset_m=(0.21, -0.06, 0),
    press_stem_height_m=0.035,
):
    """固定基座、保留腰部，左按压工具、右 2F-85；工具无隐藏运动自由度。"""
    spec = _spec_from_snapshot(G1_XML)
    spec.delete(spec.joint("floating_base_joint"))
    for key in list(spec.keys):
        spec.delete(key)
    for name in ("left_hip_pitch_link", "right_hip_pitch_link"):
        spec.delete(spec.body(name))
    _remove_native_hand_visuals(spec)
    _add_robotiq_gripper(spec, "right")
    _add_wrist_interfaces(spec, "left")
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 100
    wrist = spec.body("left_wrist_yaw_link")
    tcp = np.asarray(press_tcp_offset_m, dtype=float)
    if (
        tcp.shape != (3,)
        or not np.isfinite(tcp).all()
        or not np.isfinite(press_stem_height_m)
        or press_stem_height_m <= 0
    ):
        raise ValueError("Finite tool dimensions and positive stem height required")
    top = tcp + [0, 0, press_stem_height_m]
    # L 形压头：腕前伸 210 mm、向内偏置 60 mm；肩部保持外展，上方横杆保留净空。
    tool = wrist.add_body(name="left_press_tool")
    for name, ends, radius, mass in (
        ("left_press_shank", np.r_[[0.065, 0, press_stem_height_m], top], 0.006, 0.025),
        (
            "left_press_tip_stem",
            np.r_[top, tcp + [0, 0, 0.003]],
            0.0015,
            0.010,
        ),
    ):
        tool.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=ends,
            size=[radius],
            mass=mass,
            rgba=[0.25, 0.7, 0.45, 1],
        )
    tool.add_geom(
        name="left_press_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=tcp,
        size=[0.004, 0.0008, 0.003],
        mass=0.005,
        rgba=[0.3, 0.9, 0.5, 1],
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
    )
    tool.add_site(name="left_press_tcp", pos=tcp, size=[0.0015])
    spec.worldbody.add_light(pos=[0.3, -0.4, 1.5], dir=[0, 0, -1])
    spec.worldbody.add_geom(
        name="industrial_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2, 2, 0.1],
        pos=[0, 0, -0.74],
        rgba=[0.15, 0.17, 0.2, 1],
    )
    spec.worldbody.add_camera(
        name="front", pos=[1.4, -1.3, 1.25], euler=[65, 0, 45], fovy=48
    )
    return spec


def robot_handles(model):
    arms = np.array([model.actuator(n).id for n in ARM_JOINTS], dtype=int)
    grips = np.array([model.actuator("right_fingers_actuator").id], dtype=int)
    held = np.array(
        [i for i in range(model.nu) if i not in set(arms) | set(grips)], dtype=int
    )
    return IndustrialRobotHandles(
        arms, held, grips, model.site("left_press_tcp").id, model.site("right_pinch").id
    )


def solve_reset_pose(model, data, side, site_name, target, rotation, *, iterations=500):
    """仅用于 reset 的运动学对齐；运行阶段禁止调用此函数移动机器人/对象。

    采用关节范围内阻尼最小二乘，返回真实残差；不把不可达目标静默算作成功。
    """
    joints = np.array(
        [model.joint(n).id for n in ARM_JOINTS if n.startswith(side + "_")]
    )
    adr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    site = model.site(site_name).id
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        current = data.site_xmat[site].reshape(3, 3)
        dp = np.asarray(target) - data.site_xpos[site]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, (rotation @ current.T).ravel())
        dr = np.empty(3)
        mujoco.mju_quat2Vel(dr, quaternion, 1.0)
        if np.linalg.norm(dp) < 1e-5 and np.linalg.norm(dr) < 1e-4:
            return float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jac = np.vstack((jp[:, dofs], 0.2 * jr[:, dofs]))
        delta = jac.T @ np.linalg.solve(
            jac @ jac.T + 1e-4 * np.eye(6), np.r_[dp, 0.2 * dr]
        )
        data.qpos[adr] = np.clip(
            data.qpos[adr] + 0.3 * np.clip(delta, -0.15, 0.15),
            model.jnt_range[joints, 0],
            model.jnt_range[joints, 1],
        )
    raise ValueError(
        f"Unreachable {side} reset pose: position {np.linalg.norm(dp):.6g} m, orientation {np.linalg.norm(dr):.6g} rad"
    )
