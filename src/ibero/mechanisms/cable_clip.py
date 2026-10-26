"""侧向弹性线夹：线束沿板面切向经过，沿板面法线压入双唇夹座。

局部坐标 x 横跨开口、y 沿线束、z 从板面指向机器人。夹臂只允许
x-z 平面弯曲；夹持、保持和退出均由 MuJoCo 接触及被动梁决定。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import mujoco
import numpy as np

from ibero.materials.elastic_beam import add_elastic_beam
from ibero.materials.parameters import BeamParameters, finite_number


@dataclass(frozen=True)
class CableClipParameters:
    """单根圆线的示例线夹；所有尺寸为米，力学参数仍是 provisional。"""

    wire_radius_m: float = 0.005
    seat_half_width_m: float = 0.0056
    mouth_half_width_m: float = 0.0038
    seat_depth_m: float = 0.008
    jaw_length_m: float = 0.025
    jaw_width_m: float = 0.020
    jaw_thickness_m: float = 0.0015
    young_pa: float = 2.0e9
    density_kg_m3: float = 1200.0
    relaxation_time_s: float = 0.003
    lip_depth_m: float = 0.002
    friction: float = 0.4
    parameter_source: str = "provisional side-entry cable clip; not measured"

    def __post_init__(self):
        for entry in fields(self):
            if entry.name != "parameter_source":
                finite_number(
                    getattr(self, entry.name),
                    entry.name,
                    positive=entry.name != "friction",
                )
        if not isinstance(self.parameter_source, str) or not self.parameter_source.strip():
            raise ValueError("parameter_source must identify the example or measurement")
        if not (
            self.mouth_half_width_m
            < self.wire_radius_m
            < self.seat_half_width_m
        ):
            raise ValueError("The mouth must be narrower and the seat wider than the wire")
        if self.jaw_width_m <= 2 * self.wire_radius_m:
            raise ValueError("The cable must fit within the finite clip width")
        if not (
            self.wire_radius_m
            <= self.seat_depth_m
            < self.jaw_length_m - self.wire_radius_m
        ):
            raise ValueError("The full wire cross-section must fit behind the lip")
        if self.lip_depth_m >= self.jaw_length_m / 2:
            raise ValueError("The lip must be shorter than half the jaw")
        if self.jaw_thickness_m >= self.jaw_length_m / 5:
            raise ValueError("The jaw must remain a slender beam")

    @property
    def lip_height_m(self) -> float:
        return self.seat_half_width_m - self.mouth_half_width_m

    @property
    def seat_z_min_m(self) -> float:
        return self.wire_radius_m

    @property
    def seat_z_max_m(self) -> float:
        # 线心可以靠近唇边；全截面都在唇边背面会错误拒绝实际被卡住的线。
        # 此边界仅用于候选入座，保持还须检查夹口回弹、净宽与小拉力。
        return self.lip_z_m - self.lip_depth_m / 2

    @property
    def seat_half_length_m(self) -> float:
        return self.jaw_width_m / 2

    @property
    def lip_z_m(self) -> float:
        """唇边背面到板面的距离，正向指向机器人。"""
        return self.jaw_length_m

    def beam(self) -> BeamParameters:
        return BeamParameters(
            length_m=self.jaw_length_m,
            width_m=self.jaw_width_m,
            thickness_m=self.jaw_thickness_m,
            young_pa=self.young_pa,
            density_kg_m3=self.density_kg_m3,
            relaxation_time_s=self.relaxation_time_s,
            parameter_source=self.parameter_source,
        )


@dataclass(frozen=True)
class CableClipNames:
    name: str
    mount_body: str
    seat_site: str
    tip_sites: tuple[str, str]
    beam_joints: tuple[str, ...]
    contact_geoms: tuple[str, ...]


def add_cable_clip(
    spec: mujoco.MjSpec,
    params: CableClipParameters,
    *,
    name: str,
    origin=(0, 0, 0),
    quaternion=(1, 0, 0, 0),
    segments: int = 6,
) -> CableClipNames:
    """把夹具安装在板面；座和两只悬臂始终参与碰撞，不切换约束。"""
    if not isinstance(name, str) or not name or any(c.isspace() for c in name):
        raise ValueError("Clip name must be a nonempty identifier without spaces")
    origin = np.asarray(origin, dtype=float)
    quaternion = np.asarray(quaternion, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Clip origin must be a finite 3D position")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("Clip quaternion must be a finite wxyz quaternion")
    if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-6):
        raise ValueError("Clip quaternion must be normalized")
    beam = params.beam()
    mount = spec.worldbody.add_body(
        name=f"{name}_mount", pos=origin, quat=quaternion
    )
    # 固定板面及背座阻止线束从后方穿过；薄板的正面位于局部 z=0。
    mount.add_geom(
        name=f"{name}_base",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, -0.001],
        size=[params.seat_half_width_m + params.jaw_thickness_m + 0.004,
              params.jaw_width_m / 2 + 0.003, 0.001],
        rgba=[0.22, 0.28, 0.32, 1],
        friction=[params.friction, 0.005, 0.0001],
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
    )
    seat_site = mount.add_site(
        name=f"{name}_seat",
        pos=[0, 0, params.seat_depth_m],
        size=[0.001],
        rgba=[0.15, 0.9, 0.3, 1],
    )
    tips, joints = [], []
    contact_geoms = [f"{name}_base"]
    # 将已有梁的局部 +X 转到线夹局部 +Z；梁绕局部 Y 的关节因接触自然向两侧弯曲。
    turn_to_normal = [math.sqrt(0.5), 0, -math.sqrt(0.5), 0]
    h = beam.length_m / segments
    for side, sign in (("plus", 1), ("minus", -1)):
        root = mount.add_body(
            name=f"{name}_{side}_root",
            pos=[sign * (params.seat_half_width_m + params.jaw_thickness_m / 2), 0, 0],
            quat=turn_to_normal,
        )
        last, beam_names = add_elastic_beam(
            root, beam, segments=segments, prefix=f"{name}_{side}_"
        )
        # 梁末端一只向内凸起的实体唇边；局部 +Z 映射到线夹 -X。
        lip = last.add_geom(
            name=f"{name}_{side}_lip",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[h + params.lip_depth_m / 2,
                 0,
                 sign * (params.jaw_thickness_m + params.lip_height_m) / 2],
            size=[params.lip_depth_m / 2,
                  params.jaw_width_m / 2,
                  params.lip_height_m / 2],
            mass=max(1e-7, beam.mass_kg * params.lip_depth_m / beam.length_m),
            rgba=[0.9, 0.55, 0.15, 1],
            friction=[params.friction, 0.005, 0.0001],
            solref=[0.001, 1],
            solimp=[0.99, 0.999, 0.00001, 0.5, 2],
        )
        tips.append(beam_names.tip_site)
        joints.extend(beam_names.joints)
        contact_geoms.extend(f"{name}_{side}_beam_{i}" for i in range(segments))
        contact_geoms.append(lip.name)
    return CableClipNames(
        name=name,
        mount_body=mount.name,
        seat_site=seat_site.name,
        tip_sites=tuple(tips),
        beam_joints=tuple(joints),
        contact_geoms=tuple(contact_geoms),
    )


class CableClipObserver:
    """只读观察夹臂回弹和接触穿透；不改变物理、也不单独宣布线束捕获。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        names: CableClipNames,
        params: CableClipParameters,
        *,
        recovery_tolerance_m: float = 0.0006,
        penetration_limit_m: float = 0.001,
    ):
        self.model = model
        self.names = names
        self.params = params
        self.recovery_tolerance_m = finite_number(
            recovery_tolerance_m, "recovery_tolerance_m"
        )
        self.penetration_limit_m = finite_number(
            penetration_limit_m, "penetration_limit_m"
        )
        self.mount_id = model.body(names.mount_body).id
        self.tip_ids = tuple(model.site(site).id for site in names.tip_sites)
        self.contact_geom_ids = frozenset(model.geom(geom).id for geom in names.contact_geoms)

    def observe(self, data: mujoco.MjData) -> dict[str, object]:
        rotation = data.xmat[self.mount_id].reshape(3, 3).copy()
        origin = data.xpos[self.mount_id].copy()
        tip_local = [
            rotation.T @ (data.site_xpos[site] - origin) for site in self.tip_ids
        ]
        nominal_half_span = (
            self.params.seat_half_width_m + self.params.jaw_thickness_m / 2
        )
        offsets = (
            abs(float(tip_local[0][0]) - nominal_half_span),
            abs(float(tip_local[1][0]) + nominal_half_span),
        )
        # gap 是无转角时的等效唇口净宽；弯曲较大时以真实几何碰撞为准。
        gap = (
            float(tip_local[0][0] - tip_local[1][0])
            - self.params.jaw_thickness_m
            - 2 * self.params.lip_height_m
        )
        penetration = 0.0
        for contact in data.contact[: data.ncon]:
            if (
                int(contact.geom1) in self.contact_geom_ids
                or int(contact.geom2) in self.contact_geom_ids
            ):
                penetration = max(penetration, -float(contact.dist))
        return {
            "id": self.names.name,
            "origin_m": origin,
            "rotation_world": rotation,
            "seat_z_min_m": self.params.seat_z_min_m,
            "seat_z_max_m": self.params.seat_z_max_m,
            "seat_half_width_m": self.params.seat_half_width_m,
            "seat_half_length_m": self.params.seat_half_length_m,
            "lip_z_m": self.params.lip_z_m,
            "wire_radius_m": self.params.wire_radius_m,
            "jaw_gap_m": gap,
            "jaw_deflection_m": max(offsets),
            "jaw_recovered": max(offsets) <= self.recovery_tolerance_m,
            "max_penetration_m": max(0.0, penetration),
            "penetration_limit_m": self.penetration_limit_m,
        }
