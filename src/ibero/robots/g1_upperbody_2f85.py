"""P2 runtime preset: fixed-base G1 upper body with two real 2F-85 grippers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from ibero.materials.cable import (
    CableHandles,
    CableParameters,
    add_cable,
    cable_handles,
)
from ibero.robots.g1_upperbody import (
    ARM_JOINTS,
    G1_XML,
    GRIPPER_XML,
    _spec_from_snapshot,
)


READY_ARM_QPOS = np.asarray(
    [
        -0.80,
        0.45,
        0.00,
        1.00,
        0.00,
        -0.20,
        0.00,
        -0.80,
        -0.45,
        0.00,
        1.00,
        0.00,
        -0.20,
        0.00,
    ],
    dtype=np.float64,
)

# The dedicated handover fixture is aligned to the Menagerie model's neutral
# arm pose.  Keeping this separate from the Core-0 force-test posture makes
# the physical initial grasp an explicit preset property.
HANDOVER_ARM_QPOS = np.zeros(14, dtype=np.float64)


# 当前 vendored G1 MJCF 的“手”是固定 rubber-hand 外观几何，并非带手指
# 关节和致动器的灵巧手模型。把这一事实编码为独立模式，防止配置误导用户。
SUPPORTED_END_EFFECTORS = frozenset({"robotiq_2f85", "g1_native_hand"})
_NATIVE_HAND_MESHES = frozenset({"left_rubber_hand", "right_rubber_hand"})


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise RuntimeError(f"Expected {kind.name} named {name!r} in G1 preset")
    return result


@lru_cache(maxsize=1)
def vendored_robot_asset_hash() -> str:
    """Hash the XML and mesh snapshots that determine the assembled preset."""

    digest = hashlib.sha256()
    for root in (G1_XML.parent, GRIPPER_XML.parent):
        for path in sorted(Path(root).rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class HandoverModelHandles:
    """Stable P2/P3 names resolved to ids after model compilation."""

    arm_actuator_ids: np.ndarray
    hold_actuator_ids: np.ndarray
    gripper_actuator_ids: np.ndarray
    # The resolved-rate controller retains these historical aliases.  They are
    # actual 2F-85 pinch sites in this preset, not wrist cable anchors.
    left_anchor_site_id: int
    right_anchor_site_id: int
    left_pinch_site_id: int
    right_pinch_site_id: int
    left_gripper_base_body_id: int
    right_gripper_base_body_id: int
    left_ft_force_sensor_id: int
    left_ft_torque_sensor_id: int
    right_ft_force_sensor_id: int
    right_ft_torque_sensor_id: int
    left_pad_geom_ids: tuple[int, ...]
    right_pad_geom_ids: tuple[int, ...]
    left_robot_geom_ids: frozenset[int]
    right_robot_geom_ids: frozenset[int]
    target_receiver_body_id: int
    target_region_site_id: int
    cable: CableHandles
    end_effector_type: str


def _add_wrist_interfaces(spec: mujoco.MjSpec, side: str) -> None:
    """Attach sensors/camera shared by both selectable end-effector modes."""

    wrist = spec.body(f"{side}_wrist_yaw_link")
    # F/T site 统一留在 G1 手腕，故即便未来换成真实灵巧手，传感器坐标系
    # 与观测字段也无需改变。
    ft_site = wrist.add_site(
        name=f"{side}_wrist_ft",
        pos=[0.065, 0.0, 0.0],
        size=[0.004],
        rgba=[0.85, 0.20, 0.95, 0.8],
    )
    spec.add_sensor(
        name=f"{side}_wrist_ft_force",
        type=mujoco.mjtSensor.mjSENS_FORCE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=ft_site.name,
    )
    spec.add_sensor(
        name=f"{side}_wrist_ft_torque",
        type=mujoco.mjtSensor.mjSENS_TORQUE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=ft_site.name,
    )
    wrist.add_camera(
        name=f"{side}_wrist",
        pos=[0.075, 0.0, 0.040],
        euler=[0.0, -28.0, 0.0],
        fovy=65.0,
    )


def _remove_native_hand_visuals(spec: mujoco.MjSpec) -> None:
    """Remove G1's non-actuated hand meshes before an external gripper mounts."""

    # 这些 geom 在源 MJCF 中没有 name，必须按 meshname 定位；只删 visual
    # 外观，不会删除手腕 link、关节或碰撞几何。
    native_geoms = [geom for geom in spec.geoms if geom.meshname in _NATIVE_HAND_MESHES]
    if len(native_geoms) != 2:
        raise RuntimeError("Expected exactly two fixed G1 rubber-hand visual geoms")
    for geom in native_geoms:
        spec.delete(geom)


def _add_robotiq_gripper(spec: mujoco.MjSpec, side: str) -> None:
    """Mount one 2F-85 after the incompatible G1 hand visual has been removed."""

    wrist = spec.body(f"{side}_wrist_yaw_link")
    mount = wrist.add_site(
        name=f"{side}_gripper_mount",
        pos=[0.065, 0.0, 0.0],
        # Robotiq 资产沿 +Z 前伸；G1 腕部沿 +X 前伸。绕腕部 Y 转 90°。
        # 安装旋转与机械臂关节姿态必须分开，不能靠扭腕掩盖错误转接坐标系。
        quat=[2**-0.5, 0.0, 2**-0.5, 0.0],
        size=[0.006],
        rgba=[0.15, 0.85, 0.95, 0.8],
    )
    spec.attach(_spec_from_snapshot(GRIPPER_XML), site=mount, prefix=f"{side}_")
    _add_wrist_interfaces(spec, side)


def _add_native_hand_interfaces(spec: mujoco.MjSpec, side: str) -> None:
    """Expose inspection sites for the stock visual-only G1 hand option."""

    wrist = spec.body(f"{side}_wrist_yaw_link")
    wrist.add_site(
        name=f"{side}_pinch",
        pos=[0.090, 0.0, 0.0],
        size=[0.006],
        rgba=[0.95, 0.70, 0.12, 0.8],
    )
    _add_wrist_interfaces(spec, side)


def _add_workcell(
    spec: mujoco.MjSpec,
    target_region: np.ndarray,
    target_radius: float,
    receiver_config=None,
) -> None:
    spec.worldbody.add_geom(
        name="workcell_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.1],
        pos=[0.0, 0.0, -0.74],
        rgba=[0.12, 0.13, 0.16, 1.0],
        contype=1,
        conaffinity=1,
    )
    spec.worldbody.add_light(
        name="workcell_key_light",
        pos=[0.3, -0.5, 1.6],
        dir=[-0.2, 0.35, -1.0],
        diffuse=[0.9, 0.9, 0.9],
    )
    spec.worldbody.add_camera(
        name="front",
        pos=[1.55, -1.55, 1.35],
        euler=[62.0, 0.0, 45.0],
        fovy=52.0,
    )
    receiver = spec.worldbody.add_body(
        name="target_receiver", pos=target_region.tolist()
    )
    if receiver_config:
        receiver.add_geom(
            name="receiver_support",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=receiver_config["offset_m"],
            size=receiver_config["half_size_m"],
            rgba=[0.18, 0.45, 0.25, 1],
            friction=[1, 0.005, 0.0001],
        )
    receiver.add_geom(
        name="target_receiver_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[target_radius, 0.002],
        rgba=[0.10, 0.92, 0.36, 0.26],
        contype=0,
        conaffinity=0,
    )
    receiver.add_site(
        name="target_region",
        # The logical radius lives in constraints.yaml.  Keep this site tiny
        # so its debug rendering does not hide the physical handover itself.
        size=[0.004],
        rgba=[0.10, 0.92, 0.36, 0.55],
    )


def _robot_side_geoms(model: mujoco.MjModel, side: str) -> frozenset[int]:
    ids: set[int] = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        while body_id > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and name.startswith(f"{side}_"):
                ids.add(geom_id)
                break
            body_id = int(model.body_parentid[body_id])
    return frozenset(ids)


def _pad_geom_ids(model: mujoco.MjModel, side: str) -> tuple[int, ...]:
    names = (
        f"{side}_right_pad1",
        f"{side}_right_pad2",
        f"{side}_left_pad1",
        f"{side}_left_pad2",
    )
    return tuple(_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names)


def _end_effector_type(scene_config: Mapping[str, Any]) -> str:
    """Read the validated recipe choice close to the P2 assembly boundary."""

    end_effector = str(scene_config["robot"]["end_effector"]["type"])
    if end_effector not in SUPPORTED_END_EFFECTORS:
        raise ValueError(f"Unsupported end effector {end_effector!r}")
    return end_effector


def build_g1_handover_model(
    scene_config: Mapping[str, Any], constraints: Mapping[str, Any]
) -> tuple[mujoco.MjModel, HandoverModelHandles, CableParameters]:
    """Compile the P2 robot, selected P3 cable and static P4 workcell.

    The original Menagerie G1 assets are composed at runtime to keep their
    licences intact.  Removing the floating root and both hip subtrees creates
    a fixed-base upper-body workcell.  The remaining waist joints are held at
    their explicit reset targets and are excluded from the public 14-DoF arm
    action interface.
    """

    if not G1_XML.exists() or not GRIPPER_XML.exists():
        raise FileNotFoundError(f"Missing vendored assets: {G1_XML} / {GRIPPER_XML}")
    params = CableParameters.from_scene(scene_config["materials"]["cable"])
    initialization = scene_config["initialization"]
    terminal_origin = np.asarray(initialization["terminal_pose"], dtype=np.float64)
    tail_direction = np.asarray(initialization["tail_direction"], dtype=np.float64)
    target_region = np.asarray(initialization["target_region"], dtype=np.float64)
    target_radius = float(constraints["task"]["target_radius_m"])

    end_effector_type = _end_effector_type(scene_config)
    spec = _spec_from_snapshot(G1_XML)
    spec.modelname = f"ibero_g1_upperbody_{end_effector_type}_handover"
    spec.delete(spec.joint("floating_base_joint"))
    # A keyframe stores a full qpos vector.  Once the leg subtrees are removed
    # it is intentionally invalid, so drop source locomotion keyframes rather
    # than silently retaining a malformed upper-body preset.
    for key in list(spec.keys):
        spec.delete(key)
    for hip_root in ("left_hip_pitch_link", "right_hip_pitch_link"):
        spec.delete(spec.body(hip_root))
    # 接触 + Flex 的稳定性优先于实时速度；这是当前回归使用的工程设置，
    # 不是已证明的最小迭代次数。改动后须重新执行 calibration 和基线。
    spec.option.timestep = float(scene_config["physics"]["timestep_s"])
    # 弹性边的阻尼由隐式速度积分处理；弯曲外力在每个物理子步更新。
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 100
    spec.option.ls_iterations = 50
    _add_workcell(
        spec,
        target_region,
        target_radius,
        scene_config.get("workcell", {}).get("receiver"),
    )
    if end_effector_type == "robotiq_2f85":
        _remove_native_hand_visuals(spec)
        _add_robotiq_gripper(spec, "left")
        _add_robotiq_gripper(spec, "right")
    else:
        # 保留原始 G1 手部外观用于构型审阅；它没有可控手指，不能假装成
        # 已支持的灵巧手 handover。环境会在创建任务时给出明确提示。
        _add_native_hand_interfaces(spec, "left")
        _add_native_hand_interfaces(spec, "right")
    add_cable(
        spec,
        params=params,
        terminal_origin=terminal_origin,
        tail_direction=tail_direction,
    )
    model = spec.compile()

    arm_actuators = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINTS],
        dtype=np.int32,
    )
    gripper_actuators = (
        np.asarray(
            [
                _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_fingers_actuator"),
                _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_fingers_actuator"),
            ],
            dtype=np.int32,
        )
        if end_effector_type == "robotiq_2f85"
        else np.empty(0, dtype=np.int32)
    )
    held = np.asarray(
        [index for index in range(model.nu) if index not in set(arm_actuators)],
        dtype=np.int32,
    )
    held = held[~np.isin(held, gripper_actuators)]
    cable = cable_handles(model, params.representation)
    handles = HandoverModelHandles(
        arm_actuator_ids=arm_actuators,
        hold_actuator_ids=held,
        gripper_actuator_ids=gripper_actuators,
        left_anchor_site_id=_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_pinch"),
        right_anchor_site_id=_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_pinch"),
        left_pinch_site_id=_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_pinch"),
        right_pinch_site_id=_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_pinch"),
        left_gripper_base_body_id=_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "left_base"
            if end_effector_type == "robotiq_2f85"
            else "left_wrist_yaw_link",
        ),
        right_gripper_base_body_id=_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_base"
            if end_effector_type == "robotiq_2f85"
            else "right_wrist_yaw_link",
        ),
        left_ft_force_sensor_id=_id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wrist_ft_force"
        ),
        left_ft_torque_sensor_id=_id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wrist_ft_torque"
        ),
        right_ft_force_sensor_id=_id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wrist_ft_force"
        ),
        right_ft_torque_sensor_id=_id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wrist_ft_torque"
        ),
        left_pad_geom_ids=(
            _pad_geom_ids(model, "left") if end_effector_type == "robotiq_2f85" else ()
        ),
        right_pad_geom_ids=(
            _pad_geom_ids(model, "right") if end_effector_type == "robotiq_2f85" else ()
        ),
        left_robot_geom_ids=_robot_side_geoms(model, "left"),
        right_robot_geom_ids=_robot_side_geoms(model, "right"),
        target_receiver_body_id=_id(model, mujoco.mjtObj.mjOBJ_BODY, "target_receiver"),
        target_region_site_id=_id(model, mujoco.mjtObj.mjOBJ_SITE, "target_region"),
        cable=cable,
        end_effector_type=end_effector_type,
    )
    return model, handles, params
