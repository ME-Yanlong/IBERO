"""固定毛坯、受限三轴端铣：几何候选→平均载荷/能力→原子提交→动力学。"""

from dataclasses import dataclass
import math
import mujoco
import numpy as np
from ibero.processes.tools import ToolPose, swept_cells
from ibero.processes.implicit_wrench import ImplicitWrenchCoupling
from ibero.processes.milling_forces import (
    mean_side_wrench,
    mean_face_wrench,
    capacity_reason,
)


@dataclass(frozen=True)
class MillingState:
    mode: str
    invalid_reason: str | None
    rpm: float
    force_world_n: tuple
    torque_world_nm: tuple
    application_point_world_m: tuple
    spindle_power_w: float
    removed_volume_m3: float
    material_version: int
    axial_depth_m: float
    evaluated_velocity_world_m_s: tuple
    evaluated_rpm: float
    coupling_velocity_residual: float


class MillingProcess:
    """本模型不是微观断裂：体素更新离散，载荷是周向平均。

    刃区正常切削时，工具—毛坯硬接触由过程载荷替代，避免同一剪切阻力
    计算两次。刀柄和夹具始终保持原生接触。停转恢复刃区实体接触；过载
    不提交候选，也不继续积分。调用环境必须把 invalid 锁存为终止。

    体素格内表面未知：用不超过一格的前视估计未切啮合，平滑格子逐个
    消失带来的力脉冲。几何仍只按实际当前/下一子步预测运动扫掠提交，
    不删前视区。瞬时 ΔV/Δt 不冒称连续瞬时切屑流率，功率核对用平均量。
    """

    def __init__(
        self,
        model,
        binding,
        tool,
        coefficients,
        limits,
        *,
        tool_body,
        tool_tip_site,
        blade_geom,
        spindle_joint,
        angular_samples=128,
    ):
        if type(angular_samples) is not int or not 32 <= angular_samples <= 512:
            raise ValueError("Bounded engagement quadrature required")
        self.model, self.binding, self.stock = model, binding, binding.stock
        self.tool, self.coefficients, self.limits = tool, coefficients, limits
        self.tool_body = model.body(tool_body).id
        self.tip_site = model.site(tool_tip_site).id
        self.blade_geom = model.geom(blade_geom).id
        self.spindle_joint = model.joint(spindle_joint).id
        self.stock_body = model.body("machining_stock").id
        if not np.all(binding._initial_type == 8) or not np.all(
            binding._initial_affinity == 1
        ):
            raise ValueError(
                "Milling stock requires dedicated contype=8/conaffinity=1; cannot disable global contact"
            )
        self.nphi = angular_samples
        phi = (np.arange(angular_samples) + 0.5) * (2 * np.pi / angular_samples)
        self.radial = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(len(phi))))
        self.invalid_reason = None
        self.sequence = 0
        self.coupling = ImplicitWrenchCoupling(
            model,
            tool_body=self.tool_body,
            tip_site=self.tip_site,
            spindle_dof=int(model.jnt_dofadr[self.spindle_joint]),
        )

    def reset(self):
        self.invalid_reason = None
        self.sequence = 0
        self.model.geom_contype[self.blade_geom] = 1
        self.model.geom_conaffinity[self.blade_geom] = 1

    def kinematics(self, data):
        rotation = data.site_xmat[self.tip_site].reshape(3, 3)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        pose = ToolPose(tuple(data.site_xpos[self.tip_site]), tuple(quat))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_SITE, self.tip_site, velocity, 0
        )
        rpm = (
            float(data.qvel[self.model.jnt_dofadr[self.spindle_joint]])
            * 60
            / (2 * np.pi)
        )
        # 刀尖 site 固定在不旋转安装座，不能把转子转速误当三轴工作站姿态变化。
        return pose, velocity[3:], velocity[:3], rpm

    def _engagement(self, pose, velocity_world):
        cell = self.stock.cell_size_m
        rotation = pose.rotation
        speed = np.linalg.norm(velocity_world)
        ahead = np.zeros(3) if speed <= 1e-12 else velocity_world / speed * cell
        future = ToolPose(tuple(np.asarray(pose.position) + ahead), pose.quaternion)
        future_ids = swept_cells(self.stock, self.tool, pose, future)
        pending = future_ids[self.stock._occupied[future_ids]]
        if not len(pending):
            return np.zeros(self.nphi), np.zeros(self.nphi), 0.0, False
        # 三轴工况的刀轴平行工件 Z，沿每根体素列精确裁剪轴向交集，
        # 不用粗糙 Z 采样把 1 mm 切深误估成 1.125 mm。
        ring = self.stock.world_to_local(
            self.radial * self.tool.radius_m @ rotation.T + pose.position + ahead
        )
        size = np.asarray(self.stock.params.size_m)
        ij = np.floor((ring[:, :2] + size[:2] / 2) / cell).astype(int)
        valid = np.all(
            (ring[:, :2] >= -size[:2] / 2) & (ring[:, :2] < size[:2] / 2), axis=1
        )
        ij = np.clip(ij, 0, np.asarray(self.stock.shape[:2]) - 1)
        occupied = (
            self.stock._occupied.reshape(self.stock.shape)[ij[:, 0], ij[:, 1], :]
            & valid[:, None]
        )
        z = self.stock.centers.reshape((*self.stock.shape, 3))[0, 0, :, 2]
        half = self.stock.half_sizes.reshape((*self.stock.shape, 3))[0, 0, :, 2]
        tip_z = self.stock.world_to_local(pose.position)[2]
        future_z = self.stock.world_to_local(np.asarray(pose.position) + ahead)[2]
        lo = np.maximum(z - half, future_z)
        hi = np.minimum(z + half, future_z + self.tool.cutting_length_m)
        width = np.maximum(hi - lo, 0)
        weight = occupied * width
        lengths = weight.sum(axis=1)
        centroids = np.divide(
            (weight * ((lo + hi) / 2 - tip_z)).sum(axis=1),
            lengths,
            out=np.zeros(self.nphi),
            where=lengths > 0,
        )
        # 面刃用等面积圆盘采样，与轴向侧刃啮合分开。
        radii = self.tool.radius_m * np.sqrt((np.arange(16) + 0.5) / 16)
        face = self.radial[:, None, :] * radii[None, :, None]
        local_face = self.stock.world_to_local(face @ rotation.T + pose.position)
        face_ij = np.floor((local_face[..., :2] + size[:2] / 2) / cell).astype(int)
        face_valid = np.all(
            (local_face[..., :2] >= -size[:2] / 2)
            & (local_face[..., :2] < size[:2] / 2),
            axis=-1,
        )
        face_ij = np.clip(face_ij, 0, np.asarray(self.stock.shape[:2]) - 1)
        columns = self.stock._occupied.reshape(self.stock.shape)[
            face_ij[..., 0], face_ij[..., 1], :
        ]
        # 看一格内最近未去除的中心，不把前视采样点穿过板底误认为末层已经切空。
        below = (z <= tip_z + 1e-12) & (z >= tip_z - cell - 1e-12)
        fraction = float((np.any(columns & below, axis=-1) & face_valid).mean())
        return lengths, centroids, fraction, True

    def advance(self, data, loads, *, fault_at=None):
        """在当前状态 forward 后、物理积分前调用。外力累加器由环境统一 begin/commit。"""
        if self.invalid_reason:
            raise RuntimeError("Invalid machining episode must be reset")
        self.binding.ensure_consistent()
        pose, velocity, angular_velocity, rpm = self.kinematics(data)
        dt = self.model.opt.timestep
        local_velocity = pose.rotation.T @ velocity
        zero = np.zeros(3)
        evaluated_velocity = velocity.copy()
        evaluated_rpm = rpm
        coupling_residual = 0.0

        def state(mode, reason=None, force=zero, torque=zero, volume=0.0, depth=0.0):
            return MillingState(
                mode,
                reason,
                rpm,
                tuple(float(v) for v in force),
                tuple(float(v) for v in torque),
                tuple(pose.position),
                abs(
                    float(pose.rotation[:, 2] @ torque)
                    * evaluated_rpm
                    * 2
                    * math.pi
                    / 60
                ),
                volume,
                self.stock.version,
                depth,
                tuple(float(v) for v in evaluated_velocity),
                evaluated_rpm,
                coupling_residual,
            )

        if rpm < self.limits.min_rpm:
            # 不到可切转速：实体刀具仍可碰撞/受阻，没有清料授权。
            self.model.geom_contype[self.blade_geom] = 1
            return state("solid_contact")
        reason = None
        if rpm > self.limits.max_rpm:
            reason = "spindle_out_of_range"
        elif (
            np.linalg.norm(angular_velocity) > 1e-3
            or abs(pose.rotation[:, 2] @ self.stock.rotation[:, 2] - 1) > 1e-6
        ):
            reason = "only_fixed_axis_three_axis_milling_supported"
        elif np.linalg.norm(velocity) * dt > self.stock.cell_size_m / 4:
            reason = "motion_exceeds_geometry_substep_bound"
        if reason:
            self.invalid_reason = reason
            return state("invalid", reason)
        lengths, centroids, fraction, pending = self._engagement(pose, velocity)

        def wrench_at(u):
            v = pose.rotation.T @ u[:3]
            rev = float(u[3] * 30 / np.pi)
            f, t = mean_side_wrench(
                self.coefficients,
                radius_m=self.tool.radius_m,
                rpm=rev,
                teeth=self.limits.teeth,
                velocity_tool=v,
                axial_lengths_m=lengths,
                axial_centroids_m=centroids,
            )
            if v[2] < -1e-12 and fraction > 0:
                if not self.limits.center_cutting:
                    raise ValueError("center_cutting_not_supported")
                ff, tf = mean_face_wrench(
                    self.coefficients,
                    radius_m=self.tool.radius_m,
                    rpm=rev,
                    teeth=self.limits.teeth,
                    axial_velocity_m_s=v[2],
                    engagement_fraction=fraction,
                )
                f += ff
                t += tf
            return np.r_[pose.rotation @ f, pose.rotation @ t]

        previous_type = int(self.model.geom_contype[self.blade_geom])
        if pending:
            # 预测只在独立 data；临时工作面位掩码在 finally 恢复，尚未提交材料。
            self.model.geom_contype[self.blade_geom] = 2
            try:
                wrench, response, coupling_residual = self.coupling.solve(
                    data,
                    loads.generalized,
                    loads.external_body + loads.body_wrenches,
                    pose.position,
                    wrench_at,
                )
                evaluated_velocity = response[:3]
                evaluated_rpm = float(response[3] * 30 / np.pi)
                local_velocity = pose.rotation.T @ evaluated_velocity
            except (ValueError, np.linalg.LinAlgError) as error:
                self.invalid_reason = str(error)
                return state("invalid", self.invalid_reason)
            finally:
                self.model.geom_contype[self.blade_geom] = previous_type
            world_force, world_torque = wrench[:3], wrench[3:]
            force, torque = (
                pose.rotation.T @ world_force,
                pose.rotation.T @ world_torque,
            )
        else:
            world_force, world_torque = zero.copy(), zero.copy()
            force, torque = zero.copy(), zero.copy()
        depth = (
            float(max(lengths)) if np.linalg.norm(local_velocity[:2]) > 1e-12 else 0.0
        )
        if pending:
            reason = capacity_reason(
                self.limits,
                rpm=evaluated_rpm,
                velocity_tool=local_velocity,
                axial_depth_m=depth,
                force_tool=force,
                torque_tool=torque,
            )
        if reason:
            # 失败保留当前物理态/材料，拒绝本子步积分，不以截断力换取继续穿行。
            self.invalid_reason = reason
            self.model.geom_contype[self.blade_geom] = 1
            mujoco.mj_forward(self.model, data)
            return state("invalid", reason, world_force, world_torque, depth=depth)
        predicted = ToolPose(
            tuple(np.asarray(pose.position) + evaluated_velocity * dt), pose.quaternion
        )
        ids = (
            swept_cells(self.stock, self.tool, pose, predicted)
            if pending
            else np.empty(0, dtype=int)
        )
        # 停止进给、只有转动且仍嵌入材料的首次加工不在本三轴平均模型范围。
        actual = ids[self.stock._occupied[ids]]
        if (
            len(actual)
            and np.linalg.norm(world_force) + np.linalg.norm(world_torque) <= 1e-15
        ):
            self.invalid_reason = "removal_without_modeled_engagement"
            return state("invalid", self.invalid_reason)
        if len(actual) and np.linalg.norm(velocity) <= 1e-12:
            self.invalid_reason = "stationary_initial_overlap_unsupported"
            return state("invalid", self.invalid_reason)
        self.model.geom_contype[self.blade_geom] = 2
        volume = 0.0
        try:
            if len(actual):
                event = self.stock.prepare_removal(ids, f"cut-{self.sequence}")
                volume = self.binding.commit(event, data, fault_at=fault_at)
                self.sequence += 1
        except Exception:
            self.model.geom_contype[self.blade_geom] = previous_type
            mujoco.mj_forward(self.model, data)
            raise
        loads.add_wrench(data, self.tool_body, world_force, world_torque, pose.position)
        # 固定夹具的 Jacobian 为零，反力不会凭空移动世界；反作用仍作为明确账目记录。
        loads.add_wrench(
            data, self.stock_body, -world_force, -world_torque, pose.position
        )
        return state(
            "cutting" if pending else "air_cut",
            force=world_force,
            torque=world_torque,
            volume=volume,
            depth=depth,
        )
