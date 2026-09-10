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
    quaternion=(1, 0, 0, 0),
):
    """同一机构用于台架和自由机器人操作；fixture 才添加轴向滑台。

    插座固定在世界，插头壳体、舌片和扣齿组成动态对象。没有 weld，
    没有运行中关闭锁止约束；锁止由两个 box 的真实接触产生。
    """
    origin = np.asarray(origin, dtype=float)
    plug = spec.worldbody.add_body(name="latch_plug", pos=origin, quat=quaternion)
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
        mass=latch.plug_mass_kg * (1 if fixture else 0.9),
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
    socket = spec.worldbody.add_body(name="latch_socket", pos=origin, quat=quaternion)
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
        pos=[0.025 if fixture else 0.039, 0, -0.025],
        size=[0.045 if fixture else 0.031, 0.025, 0.003],
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
    if not fixture:
        # 自由插头的真实导向滑靴；不使用单轴关节限制六自由度。
        # 滑靴质量从壳体总质量中分配，避免引入隐形配重。
        plug.add_geom(
            name="plug_guide_keel",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.03, 0, -0.0185],
            size=[0.035, 0.009, 0.0035],
            mass=latch.plug_mass_kg * 0.1,
            rgba=[0.2, 0.4, 0.6, 1],
        )
        for sign in (-1, 1):
            socket.add_geom(
                name=f"socket_guide_lip_{sign}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.037, sign * 0.009, -0.012],
                size=[0.027, 0.003, 0.0025],
                rgba=[0.4, 0.45, 0.5, 1],
                friction=[latch.friction, 0.001, 0.0001],
            )
        for geom in spec.geoms:
            if geom.name.startswith(("plug_", "socket_")):
                geom.solref = [0.001, 1]
                geom.solimp = [0.99, 0.999, 0.00001, 0.5, 2]
    return LatchNames(
        names.tip_site, "latch_hook", "latch_shoulder", plug.name, names.joints
    )


class LatchObserver:
    """状态标签只读几何和接触；损伤能力尚未实现，超限只标模型失效。"""

    def __init__(
        self,
        model,
        names,
        *,
        origin_z=0.0,
        max_force_n=5.0,
        max_deflection_m=0.012,
        audited_geom_ids=None,
    ):
        self.model, self.names = model, names
        self.tip = model.site(names.tip_site).id
        self.hook = model.geom(names.hook_geom).id
        self.shoulder = model.geom(names.shoulder_geom).id
        self.socket = model.body("latch_socket").id
        self.plug = model.body(names.plug_body).id
        self.origin_z = origin_z
        self.max_force_n = max_force_n
        self.max_deflection_m = max_deflection_m
        self.audited_geom_ids = (
            None if audited_geom_ids is None else frozenset(audited_geom_ids)
        )
        self.reset()

    def reset(self):
        self.invalid_reason = None
        self.events = []
        self.previous = None
        self.peak_contact_n = 0.0
        self.peak_penetration_m = 0.0

    def bounds(self, data, geom):
        # 所有净空在固定插座局部坐标测量；改变工作台朝向不改变锁止含义。
        frame = data.xmat[self.socket].reshape(3, 3).T
        half = (
            np.abs(frame @ data.geom_xmat[geom].reshape(3, 3))
            @ self.model.geom_size[geom]
        )
        center = frame @ (data.geom_xpos[geom] - data.xpos[self.socket])
        return center - half, center + half

    def observe(self, data, *, sample_time_s=None):
        # mj_step 留下的是积分前的几何/接触；调用者可显式传入该采样时刻。
        # 已执行 mj_forward 的完整当前帧则沿用 data.time。
        sample_time = float(data.time if sample_time_s is None else sample_time_s)
        lo, hi = self.bounds(data, self.hook)
        shoulder_lo, shoulder_hi = self.bounds(data, self.shoulder)
        clearance = float(shoulder_lo[2] - hi[2])
        released = bool(hi[0] < shoulder_lo[0] - 0.001)
        deflection = -float(
            data.xmat[self.plug].reshape(3, 3)[:, 2]
            @ (data.site_xpos[self.tip] - data.xpos[self.plug])
        )
        contact_n, penetration = 0.0, 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            if self.audited_geom_ids is not None and not (
                set((contact.geom1, contact.geom2)) & self.audited_geom_ids
            ):
                continue
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
                {"time_s": sample_time, "state": state, "clearance_m": clearance}
            )
            self.previous = state
        return {
            "time_s": sample_time,
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
