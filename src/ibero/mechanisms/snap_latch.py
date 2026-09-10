"""弹性舌片卡扣：扣齿—肩部接触阻挡退出，按压使实际几何让开。"""

from dataclasses import dataclass

import mujoco
import numpy as np

from ibero.materials.elastic_beam import add_elastic_beam
from ibero.materials.parameters import BeamParameters, LatchParameters


@dataclass(frozen=True)
class LatchNames:
    tip_site: str
    hook_geom: str
    shoulder_geom: str
    plug_body: str
    beam_joints: tuple[str, ...]


def add_latch(
    spec,
    beam: BeamParameters,
    latch: LatchParameters,
    segments=8,
    *,
    fixture=True,
    origin=(0, 0, 0),
):
    """同一机构用于台架和自由机器人操作；fixture 才添加轴向滑台。

    插座固定在世界，插头壳体、舌片和扣齿组成动态对象。没有 weld，
    没有运行中关闭锁止约束；锁止由两个 box 的真实接触产生。
    """
    origin = np.asarray(origin, dtype=float)
    plug = spec.worldbody.add_body(name="latch_plug", pos=origin)
    if fixture:
        plug.add_joint(
            name="plug_slide",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            damping=0.8,
            frictionloss=0.05,
        )
    else:
        plug.add_freejoint(name="plug_free")
    plug.add_geom(
        name="plug_shell",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.012, 0, -0.008],
        size=[0.018, 0.009, 0.006],
        mass=latch.plug_mass_kg,
        rgba=[0.22, 0.42, 0.65, 1],
        friction=[1, 0.001, 0.0001],
    )
    last, names = add_elastic_beam(plug, beam, segments)
    h = beam.length_m / segments
    last.add_geom(
        name="latch_hook",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[h, 0, latch.hook_height_m / 2],
        size=[latch.hook_length_m / 2, beam.width_m / 2, latch.hook_height_m / 2],
        mass=0.0001,
        rgba=[0.95, 0.4, 0.1, 1],
        friction=[latch.friction, 0.001, 0.0001],
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
    )
    # 肩部在扣齿退出方向 (-X) 上，初始留出 axial_gap，底面形成 overlap。
    shoulder_right = beam.length_m - latch.hook_length_m / 2 - latch.axial_gap_m
    bottom = latch.hook_height_m - latch.overlap_m
    socket = spec.worldbody.add_body(name="latch_socket", pos=origin)
    socket.add_geom(
        name="latch_shoulder",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[shoulder_right - 0.002, 0, bottom + 0.006],
        size=[0.002, beam.width_m, 0.006],
        rgba=[0.4, 0.48, 0.55, 1],
        friction=[latch.friction, 0.001, 0.0001],
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
    )
    # 下部导向板与侧壁只靠真实接触支撑；台架的单轴约束明确不等同自由抓取。
    socket.add_geom(
        name="socket_base",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.025, 0, -0.025],
        size=[0.045, 0.025, 0.003],
        rgba=[0.25, 0.3, 0.35, 1],
    )
    for sign in (-1, 1):
        socket.add_geom(
            name=f"socket_side_{sign}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.036, sign * 0.015, -0.003],
            size=[0.032, 0.003, 0.019],
            rgba=[0.35, 0.4, 0.45, 0.45],
        )
    return LatchNames(
        names.tip_site, "latch_hook", "latch_shoulder", plug.name, names.joints
    )


class LatchObserver:
    """状态标签只读几何和接触；损伤能力尚未实现，超限只标模型失效。"""

    def __init__(
        self, model, names, *, origin_z=0.0, max_force_n=5.0, max_deflection_m=0.012
    ):
        self.model, self.names = model, names
        self.tip = model.site(names.tip_site).id
        self.hook = model.geom(names.hook_geom).id
        self.shoulder = model.geom(names.shoulder_geom).id
        self.origin_z = origin_z
        self.max_force_n = max_force_n
        self.max_deflection_m = max_deflection_m
        self.reset()

    def reset(self):
        self.invalid_reason = None
        self.events = []
        self.previous = None
        self.peak_contact_n = 0.0
        self.peak_penetration_m = 0.0

    def bounds(self, data, geom):
        half = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ self.model.geom_size[geom]
        return data.geom_xpos[geom] - half, data.geom_xpos[geom] + half

    def observe(self, data):
        lo, hi = self.bounds(data, self.hook)
        shoulder_lo, shoulder_hi = self.bounds(data, self.shoulder)
        clearance = float(shoulder_lo[2] - hi[2])
        released = bool(hi[0] < shoulder_lo[0] - 0.001)
        deflection = float(self.origin_z - data.site_xpos[self.tip, 2])
        contact_n, penetration = 0.0, 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            # 台架导向/按压接触也需要审计，不能只测扣齿这一个接触对。
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, force)
            contact_n = max(contact_n, float(np.linalg.norm(force[:3])))
            penetration = max(penetration, -float(contact.dist))
        self.peak_contact_n = max(self.peak_contact_n, contact_n)
        self.peak_penetration_m = max(self.peak_penetration_m, penetration)
        if abs(deflection) > self.max_deflection_m:
            self.invalid_reason = "deflection_out_of_model_range"
        if contact_n > self.max_force_n:
            self.invalid_reason = "force_out_of_model_range"
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or any(
                data.warning[i].number
                for i in (
                    mujoco.mjtWarning.mjWARN_BADQPOS,
                    mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC,
                )
            )
        ):
            self.invalid_reason = "numerical_instability"
        # 净空 20 um 的观察滞回仅防标签抖动，绝不用于切换物理约束。
        clear = clearance > (0 if self.previous == "clearance_open" else 0.00002)
        state = "released" if released else "clearance_open" if clear else "locked"
        if self.invalid_reason:
            state = "invalid"
        if state != self.previous:
            self.events.append(
                {"time_s": float(data.time), "state": state, "clearance_m": clearance}
            )
            self.previous = state
        return {
            "time_s": float(data.time),
            "latch_state": state,
            "released": released,
            "clearance_m": clearance,
            "deflection_m": deflection,
            "contact_force_n": contact_n,
            "penetration_m": penetration,
            "peak_contact_force_n": self.peak_contact_n,
            "peak_penetration_m": self.peak_penetration_m,
            "invalid_reason": self.invalid_reason,
        }
