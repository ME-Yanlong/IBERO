"""具体加工运行时：材料事务、载荷提交、物理时钟与审计的唯一步进入口。"""

from dataclasses import asdict
import hashlib
import json
import mujoco
import numpy as np
from ibero.core.loads import PhysicalLoads
from ibero.core.reproducibility import simulation_source_hash
from ibero.materials.parameters import finite_number
from ibero.processes.milling import MillingProcess, MillingState


class MillingRuntime:
    """台架与机器人共享积分事务；不共享执行器映射，不自动 reset 子类。"""

    def _initialize_runtime(self, compiled):
        self.scene = compiled.scene
        self.model, self.stock, self.binding = (
            compiled.model,
            compiled.stock,
            compiled.binding,
        )
        self.tool, self.coefficients, self.limits = (
            compiled.tool,
            compiled.coefficients,
            compiled.limits,
        )
        self.start = compiled.start.copy()
        self.axis_parameters = dict(compiled.axis_parameters)
        self.data = mujoco.MjData(self.model)
        self.loads = PhysicalLoads(self.model)
        self.process = MillingProcess(
            self.model,
            self.binding,
            self.tool,
            self.coefficients,
            self.limits,
            tool_body="mill_rotor",
            tool_tip_site="mill_tip",
            blade_geom="mill_blade",
            spindle_joint="mill_spindle",
            **compiled.process_options,
        )

    def reset(self, *, seed=0):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Fixture seed must be an integer in [0, 2**32)")
        self.binding.reset(self.data)
        self.process.reset()
        self.loads.reset()
        self.seed_value = seed
        self._replay_restored = False
        self._done = False
        self.last_info = {
            "time_s": 0,
            "material_version": 0,
            "removed_volume_m3": 0,
            "mode": "initial",
        }

    def manifest(self):
        # 回放身份覆盖全部台架物理配置，不能把同一毛坯上的不同刀具/系数当成同场景。
        recipe = {
            "stock_hash": self.stock.stock_hash,
            "tool": asdict(self.tool),
            "coefficients": asdict(self.coefficients),
            "limits": asdict(self.limits),
            "timestep_s": float(self.model.opt.timestep),
            "start_m": self.start.tolist(),
            "angular_samples": self.process.nphi,
            "edge_transition_chip_m": self.process.edge_transition_chip_m,
            "ft_quaternion": self.model.site("mill_ft_site").quat.tolist(),
            "axis_parameters": self.axis_parameters,
            "scene_recipe_hash": None if self.scene is None else self.scene.scene_hash,
        }
        return {
            "scene_hash": hashlib.sha256(
                json.dumps(recipe, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "source_hash": simulation_source_hash(),
            "mujoco_version": mujoco.__version__,
            "asset_hash": None,
            "tool": asdict(self.tool),
            "coefficients": asdict(self.coefficients),
            "limits": asdict(self.limits),
            "seed": self.seed_value,
        }

    def inspect_target(self, target):
        """只读裁判使用用户菜谱的收紧阈值；不能校验了 YAML 却固定按缺省门槛评分。"""
        from ibero.processes.shape_check import inspect_shape

        criteria = (
            {}
            if self.scene is None
            else {
                name: self.scene.constraints["task"][name]
                for name in ("volume_error_fraction", "boundary_error_cells")
            }
        )
        return inspect_shape(self.stock, target, **criteria)

    def step(self, target, rpm, *, substeps=50, fault_at=None):
        if self._done or self._replay_restored:
            raise RuntimeError("Finished/replayed milling must reset")
        target = np.asarray(target, dtype=float)
        finite_number(rpm, "spindle command rpm", positive=False)
        if (
            target.shape != (3,)
            or not np.isfinite(target).all()
            or not np.isfinite(rpm)
            or rpm < 0
        ):
            raise ValueError("Finite target and nonnegative spindle command required")
        if type(substeps) is not int or not 1 <= substeps <= 5000:
            raise ValueError("Invalid milling substep count")
        horizon_hit = False
        if self.scene is not None:
            dt = float(self.model.opt.timestep)
            remaining = self.scene.constraints["task"]["max_seconds"] - self.data.time
            # 只在靠近截止时刻时做除法，避免巨大合法时长 / 极小步长溢出。
            # 1e-8 个物理步仅补偿时钟累计舍入，不允许多积分一个完整子步。
            if remaining <= 0:
                substeps, horizon_hit = 0, True
            elif remaining < (substeps + 1) * dt:
                budget_steps = max(0, int(np.floor(remaining / dt + 1e-8)))
                horizon_hit = budget_steps <= substeps
                substeps = min(substeps, budget_steps)
        load_evaluation_time = float(self.data.time)
        if substeps:
            self._set_motion_command(target)
            self.data.ctrl[self.model.actuator("mill_motor").id] = rpm * np.pi / 30
        else:
            # 不足一个物理步时立即锁存超时；不写执行器、载荷、材料或引擎状态。
            self._done = True
            self.process.invalid_reason = "episode_time_limit"
            actual_rpm = float(self.data.joint("mill_spindle").qvel[0]) * 30 / np.pi
            state = MillingState(
                mode="timeout",
                invalid_reason="episode_time_limit",
                rpm=actual_rpm,
                force_world_n=(0.0, 0.0, 0.0),
                torque_world_nm=(0.0, 0.0, 0.0),
                application_point_world_m=tuple(self.data.site("mill_tip").xpos),
                spindle_power_w=0.0,
                removed_volume_m3=0.0,
                material_version=self.stock.version,
                axial_depth_m=0.0,
                evaluated_velocity_world_m_s=(0.0, 0.0, 0.0),
                evaluated_rpm=actual_rpm,
                coupling_velocity_residual=0.0,
            )
        total_removed = 0.0
        peak_force, peak_torque, peak_power, peak_shank_penetration = 0.0, 0.0, 0.0, 0.0
        peak_applied_force = 0.0
        integrated_substeps = 0
        housing = self.model.geom("mill_housing").id
        for _ in range(substeps):
            self._before_physics_step()
            load_evaluation_time = float(self.data.time)
            self.loads.begin(self.data)
            try:
                state = self.process.advance(self.data, self.loads, fault_at=fault_at)
            except Exception:
                # 材料事务异常已由 binding 回滚；本环境锁存终止，禁止带半开载荷继续。
                self._done = True
                self.process.invalid_reason = "material_or_load_transaction_failed"
                self.loads.generalized.fill(0)
                self.loads.body_wrenches.fill(0)
                self.loads.commit(self.data)
                raise
            self.loads.commit(self.data)
            peak_force = max(peak_force, float(np.linalg.norm(state.force_world_n)))
            spindle_axis = self.data.site_xmat[self.process.tip_site].reshape(3, 3)[
                :, 2
            ]
            peak_torque = max(
                peak_torque, abs(float(spindle_axis @ state.torque_world_nm))
            )
            peak_power = max(peak_power, state.spindle_power_w)
            if state.invalid_reason:
                self._done = True
                break
            total_removed += state.removed_volume_m3
            peak_applied_force = max(
                peak_applied_force, float(np.linalg.norm(state.force_world_n))
            )
            mujoco.mj_step(self.model, self.data)
            integrated_substeps += 1
            mujoco.mj_forward(self.model, self.data)
            robot_reason = self._after_physics_step()
            if robot_reason:
                self._done = True
                self.process.invalid_reason = robot_reason
                break
            for contact in self.data.contact:
                if housing in (contact.geom1, contact.geom2):
                    peak_shank_penetration = max(
                        peak_shank_penetration, -float(contact.dist)
                    )
            if (
                self.scene is not None
                and peak_shank_penetration
                > self.scene.constraints["safety"]["max_shank_penetration_m"]
            ):
                self._done = True
                self.process.invalid_reason = "shank_contact_limit"
                break
            if (
                not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
                or any(
                    self.data.warning[i].number
                    for i in (
                        mujoco.mjtWarning.mjWARN_BADQPOS,
                        mujoco.mjtWarning.mjWARN_BADQVEL,
                        mujoco.mjtWarning.mjWARN_BADQACC,
                    )
                )
            ):
                self._done = True
                self.process.invalid_reason = "numerical_instability"
                break
        if horizon_hit and not self._done:
            self._done = True
            self.process.invalid_reason = "episode_time_limit"
        info = asdict(state)
        if substeps == 0:
            info["load_evaluation_performed"] = False
        actual_omega = float(self.data.joint("mill_spindle").qvel[0])
        motor_torque = float(
            self.data.actuator_force[self.model.actuator("mill_motor").id]
        )
        info.update(
            time_s=float(self.data.time),
            load_evaluation_time_s=load_evaluation_time,
            actual_rpm=actual_omega * 30 / np.pi,
            motor_torque_nm=motor_torque,
            motor_mechanical_power_w=motor_torque * actual_omega,
            removed_volume_step_m3=total_removed,
            total_removed_volume_m3=self.stock.initial_volume_m3 - self.stock.volume_m3,
            tip_position_m=self.data.site("mill_tip").xpos.copy().tolist(),
            invalid_reason=self.process.invalid_reason,
            fixture_reaction_force_n=(-np.asarray(state.force_world_n)).tolist(),
            fixture_reaction_torque_about_tip_nm=(
                -np.asarray(state.torque_world_nm)
            ).tolist(),
            fixture_reaction_torque_about_stock_origin_nm=(
                -np.asarray(state.torque_world_nm)
                - np.cross(
                    np.asarray(state.application_point_world_m) - self.stock.origin,
                    np.asarray(state.force_world_n),
                )
            ).tolist(),
            ft_force_n=self.data.sensor("mill_force").data.copy().tolist(),
            ft_torque_nm=self.data.sensor("mill_torque").data.copy().tolist(),
            peak_cutting_force_step_n=peak_force,
            # 过载候选被拒绝时并未施加该载荷；预测峰值与真实已积分载荷必须分开。
            last_substep_load_applied=state.invalid_reason is None,
            applied_force_world_n=list(state.force_world_n)
            if state.invalid_reason is None
            else [0.0, 0.0, 0.0],
            applied_torque_world_nm=list(state.torque_world_nm)
            if state.invalid_reason is None
            else [0.0, 0.0, 0.0],
            peak_applied_cutting_force_step_n=peak_applied_force,
            integrated_substeps=integrated_substeps,
            peak_spindle_torque_step_nm=peak_torque,
            peak_spindle_power_step_w=peak_power,
            peak_shank_penetration_step_m=peak_shank_penetration,
            coupling_diagnostics=self.process.coupling.last_diagnostics
            if self.process.invalid_reason
            else None,
        )
        self.last_info = info
        return info

    def _set_motion_command(self, target):
        raise NotImplementedError("Concrete milling actuator mapping required")

    def _before_physics_step(self):
        pass

    def _after_physics_step(self):
        return None
