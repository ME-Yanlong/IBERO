"""将立式板、自由 Flex 线束、弹性侧向线夹与固定基座 G1 装成一个模型。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ibero.materials.routing_cable import RoutingCableParameters, add_routing_cable, free_cable_points
from ibero.mechanisms.cable_clip import CableClipParameters, add_cable_clip
from ibero.robots.g1_upperbody import ARM_JOINTS, G1_XML, _spec_from_snapshot
from ibero.robots.g1_upperbody_2f85 import (
    _add_robotiq_gripper,
    _pad_geom_ids,
    _remove_native_hand_visuals,
    _robot_side_geoms,
)


@dataclass(frozen=True)
class RoutingRobotHandles:
    arm_actuator_ids: np.ndarray
    hold_actuator_ids: np.ndarray
    gripper_actuator_ids: np.ndarray
    left_anchor_site_id: int
    right_anchor_site_id: int
    left_pad_geom_ids: tuple[int, ...]
    right_pad_geom_ids: tuple[int, ...]
    left_robot_geom_ids: frozenset[int]
    right_robot_geom_ids: frozenset[int]


@dataclass(frozen=True)
class CompiledRouting:
    model: mujoco.MjModel
    robot: RoutingRobotHandles
    cable: RoutingCableParameters
    clip_names: tuple
    clip_parameters: CableClipParameters


def _clip_quaternion(tangent="horizontal"):
    # 局部 x=板上向上，y=线束切向，z=板外朝机器人；压入方向为世界 +X。
    if tangent == "horizontal":
        rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    elif tangent == "vertical":
        # 局部 y 沿世界竖直；保持同一个指向机器人（-X）的夹口法向。
        rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    else:
        raise ValueError(f"Unsupported clip tangent: {tangent}")
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, rotation.ravel())
    return result


def compile_routing(config):
    """保留旧 G1 资产和现有材料力律；新组件只在布线 schema 下组合。"""
    spec = _spec_from_snapshot(G1_XML)
    spec.modelname = config["id"]
    spec.delete(spec.joint("floating_base_joint"))
    for key in list(spec.keys):
        spec.delete(key)
    for hip in ("left_hip_pitch_link", "right_hip_pitch_link"):
        spec.delete(spec.body(hip))
    _remove_native_hand_visuals(spec)
    for side in ("left", "right"):
        _add_robotiq_gripper(spec, side)
    spec.option.timestep = float(config["physics"]["timestep_s"])
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 100
    spec.option.ls_iterations = 50
    board = config["workcell"]["board"]
    spec.worldbody.add_geom(
        name="routing_board",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=board["center_m"],
        size=board["half_size_m"],
        friction=[0.7, 0.005, 0.0001],
        rgba=[0.22, 0.28, 0.33, 1],
    )
    # 三个实体支点留下两段夹爪可进入的空档，防止下指垫被供料板挡住。
    # 机器人须先在空档夹住材料，再把线移离供料架。
    for index, y in enumerate((0.145, 0.0, -0.145)):
        spec.worldbody.add_geom(
            name=f"feed_support_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.405, y, 0.850],
            size=[0.022, 0.012, 0.009],
            rgba=[0.26, 0.38, 0.50, 1],
            friction=[0.65, 0.005, 0.0001],
        )
    spec.worldbody.add_geom(
        name="routing_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, -0.74],
        size=[2, 2, 0.1],
        rgba=[0.14, 0.15, 0.17, 1],
    )
    spec.worldbody.add_light(pos=[0.3, -0.4, 1.5], dir=[0, 0, -1])
    spec.worldbody.add_camera(name="front", pos=[1.25, -1.05, 1.24], euler=[60, 0, 45], fovy=55)
    spec.worldbody.add_camera(name="side", pos=[0.72, -0.45, 0.91], euler=[90, 0, 65], fovy=45)
    cable = RoutingCableParameters.from_mapping(config["materials"]["cable"])
    initialization = config["initialization"]
    points = free_cable_points(
        initialization["cable_start_m"],
        initialization["cable_end_m"],
        length_m=cable.length_m,
        segments=cable.segments,
    )
    add_routing_cable(spec, cable, points)
    clip_params = CableClipParameters(**config["mechanism"])
    names = []
    for clip in config["workcell"]["clips"]:
        # 线夹通过实体支架离开板面，给夹持线束的指垫留下法向净空。
        # 支架位于夹座下方，不作为隐藏导线轨道。
        clip_origin = np.asarray(clip["origin_m"], dtype=float)
        board_front_x = float(board["center_m"][0] - board["half_size_m"][0])
        span = board_front_x - clip_origin[0]
        if span > 0.002:
            spec.worldbody.add_geom(
                name=f"{clip['id']}_mount_bracket",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[clip_origin[0] + span / 2, clip_origin[1], clip_origin[2] - 0.014],
                size=[span / 2, 0.006, 0.003],
                rgba=[0.38, 0.43, 0.48, 1],
            )
        names.append(
            add_cable_clip(
                spec,
                clip_params,
                name=str(clip["id"]),
                origin=clip_origin,
                quaternion=_clip_quaternion(clip.get("tangent", "horizontal")),
                segments=6,
            )
        )
    model = spec.compile()
    arms = np.array([model.actuator(n).id for n in ARM_JOINTS], dtype=np.int32)
    grips = np.array(
        [model.actuator(f"{side}_fingers_actuator").id for side in ("left", "right")],
        dtype=np.int32,
    )
    held = np.array(
        [i for i in range(model.nu) if i not in set(arms) | set(grips)], dtype=np.int32
    )
    handles = RoutingRobotHandles(
        arms,
        held,
        grips,
        model.site("left_pinch").id,
        model.site("right_pinch").id,
        _pad_geom_ids(model, "left"),
        _pad_geom_ids(model, "right"),
        _robot_side_geoms(model, "left"),
        _robot_side_geoms(model, "right"),
    )
    return CompiledRouting(model, handles, cable, tuple(names), clip_params)
