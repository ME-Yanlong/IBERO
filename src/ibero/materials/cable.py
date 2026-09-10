"""Contact-ready cable representations for the Core-0.1 harness scene.

The material values belong to ``scene_config.yaml``.  This module only turns
those values into a named MuJoCo representation; it deliberately owns no task
thresholds or scene-specific success logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np


@dataclass(frozen=True)
class CableParameters:
    """Scene-level cable parameters, all in SI units."""

    representation: str
    length_m: float
    outer_diameter_m: float
    linear_density_kg_m: float
    minimum_bend_radius_m: float
    terminal_length_m: float
    terminal_width_m: float
    terminal_height_m: float
    terminal_mass_kg: float
    axial_stiffness_n_m: float
    bending_stiffness_nm2: float
    damping: float
    friction: float
    parameter_source: Mapping[str, str]
    segments: int
    end_layout: str = "handover"
    initial_span_m: float | None = None

    @classmethod
    def from_scene(cls, cable: Mapping[str, Any]) -> "CableParameters":
        # P3 不在这里补默认材料常量：所有会改变线束响应的值必须来自已校验
        # 的菜谱 YAML，才能随 seed/config hash 一同写进 manifest。
        terminal = cable["terminal"]
        return cls(
            representation=str(cable["representation"]),
            length_m=float(cable["length_m"]),
            outer_diameter_m=float(cable["outer_diameter_m"]),
            linear_density_kg_m=float(cable["linear_density_kg_m"]),
            minimum_bend_radius_m=float(cable["minimum_bend_radius_m"]),
            terminal_length_m=float(terminal["length_m"]),
            terminal_width_m=float(terminal["width_m"]),
            terminal_height_m=float(terminal["height_m"]),
            terminal_mass_kg=float(terminal["mass_kg"]),
            axial_stiffness_n_m=float(cable["axial_stiffness_n_m"]),
            bending_stiffness_nm2=float(cable["bending_stiffness_nm2"]),
            damping=float(cable["damping"]),
            friction=float(cable["friction"]),
            parameter_source=dict(cable["parameter_source"]),
            segments=int(cable["segments"]),
            end_layout=str(cable.get("end_layout", "handover")),
            initial_span_m=cable.get("initial_span_m"),
        )


@dataclass(frozen=True)
class CableHandles:
    """Stable model names/ids exposed by a cable representation."""

    representation: str
    terminal_body_id: int
    tail_body_id: int
    terminal_left_grasp_site_id: int
    terminal_right_grasp_site_id: int
    tendon_id: int
    terminal_geom_id: int
    flex_id: int | None = None


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    identifier = mujoco.mj_name2id(model, kind, name)
    if identifier < 0:
        raise RuntimeError(f"Expected {kind.name} named {name!r}")
    return identifier


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Cable tail_direction must not be zero")
    return vector / norm


def _quat_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return MuJoCo's scalar-first quaternion for an orthonormal matrix."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    index = int(np.argmax(np.diag(rotation)))
    if index == 0:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ]
        )
    elif index == 1:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.asarray(
            [
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ]
        )
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.asarray(
            [
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    return quat / np.linalg.norm(quat)


def _flex_subspec(params: CableParameters, tail_direction: np.ndarray) -> mujoco.MjSpec:
    """Build a 1-D MuJoCo Flex assembly with physical terminal connections.

    MuJoCo's flexcomp creates the nodal bodies itself.  The terminal bodies use
    three slides plus a ball joint (rather than a kinematic attachment), and
    equality *connect* constraints join only the material endpoints to their
    corresponding Flex nodes.  This is a real material attachment, not a
    grasp shortcut: grippers still contact the terminal geoms normally.
    """

    # 以下 Flex 超参数均直接映射 YAML：节点数量影响数值分辨率，轴向刚度和
    # 阻尼影响拉力峰值。不要为某一个 task seed 在代码中悄悄调参。
    total_mass = params.linear_density_kg_m * params.length_m
    point_count = params.segments + 1
    half_length = params.terminal_length_m / 2.0
    grasp_offset = params.terminal_length_m * 0.375
    # 端头长轴是局部 Y；默认尾向 -Y 时保持单位姿态，其他方向同步旋转
    # 几何和夹持标记。只旋转 geom，不改变平移关节及连接锚点的坐标系。
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(tail_direction @ up)) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0])
    local_y = -tail_direction
    local_x = _unit(np.cross(local_y, up))
    local_z = _unit(np.cross(local_x, local_y))
    quat = _quat_from_rotation(np.column_stack((local_x, local_y, local_z)))
    quat_text = " ".join(f"{value:.12g}" for value in quat)
    span = params.initial_span_m or params.length_m
    tail = tail_direction * span
    start_site = tail_direction * half_length
    end_site = (
        -tail_direction * half_length
        if params.end_layout == "dual_end"
        else np.zeros(3)
    )
    # 单端 handover 的尾端是轻质配重；双端张力任务才使用同尺寸第二连接器。
    tail_geometry = (
        f'type="box" quat="{quat_text}" size="{params.terminal_width_m / 2} {half_length} {params.terminal_height_m / 2}" mass="{params.terminal_mass_kg}"'
        if params.end_layout == "dual_end"
        else f'type="sphere" size="{max(params.outer_diameter_m, 0.014)}" mass="{max(params.terminal_mass_kg * 0.35, 0.01)}"'
    )
    # 弧长而非端点距离定义材料原长。初始化可通过垂向正弦弧表示松弛，
    # 二分求弧幅后把原长交给每段弹簧；不会把松弛误当成负拉力。
    t = np.linspace(0, 1, point_count)
    sag_direction = -local_z

    def points(amplitude):
        return (
            start_site
            + t[:, None] * (tail + end_site - start_site)
            + np.sin(np.pi * t)[:, None] * sag_direction * amplitude
        )

    low, high = 0.0, params.length_m
    for _ in range(50):
        mid = (low + high) / 2
        if np.linalg.norm(np.diff(points(mid), axis=0), axis=1).sum() > params.length_m:
            high = mid
        else:
            low = mid
    vertices = points((low + high) / 2)
    point_text = " ".join(str(float(v)) for v in vertices.ravel())
    elements = " ".join(f"{i} {i + 1}" for i in range(params.segments))
    if params.end_layout == "dual_end":
        grasp_offset = 0.0
    left_grasp = " ".join(str(float(v)) for v in -tail_direction * grasp_offset)
    right_grasp = " ".join(str(float(v)) for v in tail_direction * grasp_offset)
    xml = f"""
<mujoco model="ibero_flex_cable">
  <worldbody>
    <body name="assembly">
      <body name="cable_terminal" pos="0 0 0">
        <joint name="cable_terminal_x" type="slide" axis="1 0 0"/>
        <joint name="cable_terminal_y" type="slide" axis="0 1 0"/>
        <joint name="cable_terminal_z" type="slide" axis="0 0 1"/>
        <joint name="cable_terminal_ball" type="ball"/>
        <geom name="cable_terminal_geom" type="box"
              quat="{quat_text}"
              contype="4" conaffinity="1"
              size="{params.terminal_width_m / 2} {half_length} {params.terminal_height_m / 2}"
              mass="{params.terminal_mass_kg}" friction="{params.friction} 0.005 0.0001"
              rgba="0.15 0.72 0.95 1"/>
        <site name="cable_terminal_left_grasp" pos="{left_grasp}" size="0.006"/>
        <site name="cable_terminal_right_grasp" pos="{right_grasp}" size="0.006"/>
        <site name="cable_start_site" pos="{start_site[0]} {start_site[1]} {start_site[2]}" size="0.004"/>
      </body>
      <body name="cable_tail_terminal" pos="{tail[0]} {tail[1]} {tail[2]}">
        <joint name="cable_tail_x" type="slide" axis="1 0 0"/>
        <joint name="cable_tail_y" type="slide" axis="0 1 0"/>
        <joint name="cable_tail_z" type="slide" axis="0 0 1"/>
        <joint name="cable_tail_ball" type="ball"/>
        <geom name="cable_tail_terminal_geom" {tail_geometry}
              contype="4" conaffinity="1"
              friction="{params.friction} 0.005 0.0001" rgba="0.25 0.28 0.34 1"/>
        <site name="cable_end_site" pos="{end_site[0]} {end_site[1]} {end_site[2]}" size="0.004"/>
        <site name="cable_tail_grasp" size="0.004"/>
      </body>
      <flexcomp name="cable_flex" type="direct" dim="1"
                point="{point_text}" element="{elements}"
                radius="{params.outer_diameter_m / 2}" mass="{total_mass}"
                rgba="0.95 0.20 0.04 1">
        <!-- 端头边缘与 Flex 节点是材料连接；屏蔽两者自身装配接触，
             保留线束/夹爪、线束/工装及端头/夹爪接触。 -->
        <contact contype="2" conaffinity="1" selfcollide="none" friction="{params.friction} 0.005 0.0001"/>
        <edge equality="false" stiffness="{params.axial_stiffness_n_m * params.segments}" damping="{params.damping * params.segments}"/>
      </flexcomp>
    </body>
  </worldbody>
  <equality>
    <connect name="terminal_to_flex" body1="cable_terminal" body2="cable_flex_0"
             anchor="{start_site[0]} {start_site[1]} {start_site[2]}" solref="0.001 1" solimp="0.999 0.999 0.001"/>
    <connect name="tail_to_flex" body1="cable_tail_terminal" body2="cable_flex_{params.segments}"
             anchor="{end_site[0]} {end_site[1]} {end_site[2]}" solref="0.001 1" solimp="0.999 0.999 0.001"/>
  </equality>
</mujoco>
"""
    # 插入 XML 的值均来自已校验数值字段，避免把未经检查的用户字符串写入
    # MuJoCo 子模型；这是 scene compiler 的安全边界。
    return mujoco.MjSpec.from_string(xml)


def _add_flex(
    spec: mujoco.MjSpec,
    *,
    params: CableParameters,
    terminal_origin: np.ndarray,
    tail_direction: np.ndarray,
) -> None:
    mount = spec.worldbody.add_site(
        name="cable_flex_mount",
        pos=terminal_origin.tolist(),
        size=[0.001],
    )
    flex_spec = _flex_subspec(params, tail_direction)
    spec.attach(flex_spec, site=mount, prefix="")
    tendon = spec.add_tendon(
        name="cable_axial_tendon",
        # 仅保留旧具名 tendon 作为端点距离测量；绝不再施力或用它估计拉力。
        stiffness=0.0,
        damping=0.0,
        springlength=[0.0, params.length_m],
        width=0.001,
        rgba=[1.0, 1.0, 1.0, 0.0],
    )
    tendon.wrap_site("cable_start_site")
    tendon.wrap_site("cable_end_site")


def add_cable(
    spec: mujoco.MjSpec,
    *,
    params: CableParameters,
    terminal_origin: np.ndarray,
    tail_direction: np.ndarray,
) -> None:
    """Add the scene-selected physical cable representation to ``spec``."""

    direction = _unit(np.asarray(tail_direction, dtype=np.float64))
    if params.segments < 4:
        raise ValueError("Cable needs at least four segments")
    if params.representation == "flex":
        # MuJoCo 的 1-D Flex 渲染/接触为一串 capsule，但不是独立刚体链后端；
        # 不再对同一个模型提供两个名称。选择 Flex 不等于完成真实材料标定。
        _add_flex(
            spec,
            params=params,
            terminal_origin=np.asarray(terminal_origin, dtype=np.float64),
            tail_direction=direction,
        )
    else:
        raise ValueError(f"Unsupported cable representation {params.representation!r}")


def cable_handles(model: mujoco.MjModel, representation: str) -> CableHandles:
    """Resolve stable names after an assembled cable model is compiled."""

    flex_candidate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "cable_flex")
    flex_id = flex_candidate if flex_candidate >= 0 else None
    return CableHandles(
        representation=representation,
        terminal_body_id=_id(model, mujoco.mjtObj.mjOBJ_BODY, "cable_terminal"),
        tail_body_id=_id(model, mujoco.mjtObj.mjOBJ_BODY, "cable_tail_terminal"),
        terminal_left_grasp_site_id=_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "cable_terminal_left_grasp"
        ),
        terminal_right_grasp_site_id=_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "cable_terminal_right_grasp"
        ),
        tendon_id=_id(model, mujoco.mjtObj.mjOBJ_TENDON, "cable_axial_tendon"),
        terminal_geom_id=_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cable_terminal_geom"),
        flex_id=flex_id,
    )
