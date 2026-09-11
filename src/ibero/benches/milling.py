"""三轴加工台架：三平移伺服＋真实主轴关节，刀具不是运动学橡皮擦。"""

from dataclasses import asdict
import hashlib
import json
import mujoco
import numpy as np
from ibero.materials.stock import VoxelStock
from ibero.materials.stock_collision import add_stock_geoms, StockCollisionBinding
from ibero.core.loads import PhysicalLoads
from ibero.core.reproducibility import simulation_source_hash
from ibero.processes.milling import MillingProcess, MillingState
from ibero.materials.parameters import finite_number
from ibero.processes.tools import quaternion_matrix


def add_spindle(
    spec,
    parent,
    tool,
    *,
    prefix="mill_",
    rotor_mass_kg=0.02,
    housing_mass_kg=0.08,
    torque_limit_nm=0.15,
    ft_quaternion=(1, 0, 0, 0),
):
    """可复用工具：端面/侧刃圆柱与非切削刀柄分开，转速来自实际转子关节。"""
    quaternion_matrix(ft_quaternion)
    for name, value in (
        ("rotor_mass_kg", rotor_mass_kg),
        ("housing_mass_kg", housing_mass_kg),
        ("torque_limit_nm", torque_limit_nm),
    ):
        finite_number(value, name)
    mount = parent.add_body(name=prefix + "mount")
    mount.add_geom(
        name=prefix + "housing",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[tool.shank_radius_m, tool.shank_length_m / 2],
        pos=[0, 0, tool.cutting_length_m + tool.shank_length_m / 2],
        mass=housing_mass_kg,
        rgba=[0.3, 0.35, 0.4, 1],
    )
    mount.add_site(
        name=prefix + "ft_site",
        pos=[0, 0, tool.cutting_length_m + tool.shank_length_m],
        size=[0.001],
        quat=ft_quaternion,
    )
    mount.add_site(name=prefix + "tip", pos=[0, 0, 0], size=[0.001])
    rotor = mount.add_body(name=prefix + "rotor")
    rotor.add_joint(
        name=prefix + "spindle",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        limited=False,
        # 工具转子不是 G1 手臂电机，不能继承父级 g1 的 0.3 N·m 摩擦和 0.01 附加惯量。
        # 真实惯量来自已声明的转子几何/质量；此原型未标定轴承损耗。
        frictionloss=0,
        armature=0,
        damping=0,
    )
    rotor.add_geom(
        name=prefix + "blade",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, tool.cutting_length_m / 2],
        size=[tool.radius_m, tool.cutting_length_m / 2],
        mass=rotor_mass_kg,
        contype=1,
        conaffinity=1,
        rgba=[0.8, 0.8, 0.85, 1],
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
    )
    # 转子有真实惯量与限矩速度伺服；绘制转动标记不代替 qvel 转速。
    rotor.add_site(
        name=prefix + "rotation_mark",
        pos=[tool.radius_m, 0, tool.cutting_length_m],
        size=[0.0006],
        rgba=[1, 0.2, 0.1, 1],
    )
    actuator = spec.add_actuator(
        name=prefix + "motor",
        target=prefix + "spindle",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
    )
    actuator.set_to_velocity(kv=0.002)
    actuator.forcelimited = True
    actuator.forcerange = [-torque_limit_nm, torque_limit_nm]
    for name, sensor in (
        ("force", mujoco.mjtSensor.mjSENS_FORCE),
        ("torque", mujoco.mjtSensor.mjSENS_TORQUE),
    ):
        spec.add_sensor(
            name=prefix + name,
            type=sensor,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=prefix + "ft_site",
        )


class MillingFixture:
    """参数由调用菜谱/台架提供；reset 以外只下发执行器和过程载荷。"""

    def __init__(
        self,
        stock_parameters,
        cell_size_m,
        tool,
        coefficients,
        limits,
        *,
        timestep=0.0002,
        start=(-0.025, 0, 0.025),
        max_cells=60000,
        angular_samples=128,
        ft_quaternion=(1, 0, 0, 0),
        edge_transition_chip_m=1e-8,
        axis_stiffness_n_m=5000.0,
        axis_damping_ns_m=80.0,
        axis_force_limit_n=50.0,
    ):
        finite_number(timestep, "timestep")
        start = np.asarray(start, dtype=float)
        if start.shape != (3,) or not np.isfinite(start).all():
            raise ValueError("Finite initial tool position required")
        self.axis_parameters = {
            "stiffness_n_m": finite_number(axis_stiffness_n_m, "axis_stiffness_n_m"),
            "damping_ns_m": finite_number(axis_damping_ns_m, "axis_damping_ns_m"),
            "force_limit_n": finite_number(axis_force_limit_n, "axis_force_limit_n"),
        }
        self.stock = VoxelStock(stock_parameters, cell_size_m, max_cells=max_cells)
        self.tool, self.coefficients, self.limits = tool, coefficients, limits
        spec = mujoco.MjSpec()
        spec.option.timestep = timestep
        spec.option.gravity = [0, 0, 0]  # 辨识台架零重力；机器人场景必须恢复真实重力。
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        add_stock_geoms(spec, self.stock, contype=8)
        self.start = np.asarray(start, dtype=float)
        carriage = spec.worldbody.add_body(name="mill_carriage", pos=self.start)
        carriage.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.002],
            mass=0.5,
            contype=0,
            conaffinity=0,
            rgba=[0, 0, 0, 0],
        )
        for i, axis in enumerate(np.eye(3)):
            name = f"mill_axis_{i}"
            carriage.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
            actuator = spec.add_actuator(
                name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT
            )
            actuator.set_to_position(kp=axis_stiffness_n_m, kv=axis_damping_ns_m)
            actuator.forcelimited = True
            actuator.forcerange = [-axis_force_limit_n, axis_force_limit_n]
        add_spindle(
            spec,
            carriage,
            tool,
            torque_limit_nm=limits.max_torque_nm,
            ft_quaternion=ft_quaternion,
        )
        spec.worldbody.add_light(pos=[0.1, -0.1, 0.2])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.binding = StockCollisionBinding(self.model, self.stock)
        self.loads = PhysicalLoads(self.model)
        self.process = MillingProcess(
            self.model,
            self.binding,
            tool,
            coefficients,
            limits,
            tool_body="mill_rotor",
            tool_tip_site="mill_tip",
            blade_geom="mill_blade",
            spindle_joint="mill_spindle",
            angular_samples=angular_samples,
            edge_transition_chip_m=edge_transition_chip_m,
        )
        self.scene = None
        self.reset(seed=0)

    @classmethod
    def from_scene(cls, root):
        """P4 菜谱经严格校验再建模；台架不是机器人加工环境。"""
        from ibero.core.scene_loader import SceneLoader
        from ibero.materials.parameters import StockParameters, strict_parameters
        from ibero.processes.tools import EndMillGeometry
        from ibero.processes.milling_forces import MillingCoefficients, MillingLimits

        scene = SceneLoader().validate(root)
        cfg = scene.config
        if cfg["kind"] != "milling_bench":
            raise ValueError("Expected milling_bench recipe")
        env = cls(
            strict_parameters(StockParameters, cfg["materials"]["stock"]),
            cfg["numerics"]["cell_size_m"],
            strict_parameters(EndMillGeometry, cfg["tool"]),
            strict_parameters(MillingCoefficients, cfg["process"]["coefficients"]),
            strict_parameters(MillingLimits, cfg["process"]["limits"]),
            timestep=cfg["physics"]["timestep_s"],
            start=cfg["initialization"]["tip_position_m"],
            max_cells=cfg["numerics"]["max_cells"],
            angular_samples=cfg["numerics"]["angular_samples"],
            edge_transition_chip_m=cfg["numerics"]["edge_transition_chip_m"],
            axis_stiffness_n_m=cfg["machine"]["axis_stiffness_n_m"],
            axis_damping_ns_m=cfg["machine"]["axis_damping_ns_m"],
            axis_force_limit_n=cfg["machine"]["axis_force_limit_n"],
        )
        env.scene = scene
        return env

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
        """台架只写三轴执行器；机器人复用过程步进而提供自己的执行器映射。"""
        for i in range(3):
            self.data.ctrl[self.model.actuator(f"mill_axis_{i}").id] = (
                target[i] - self.start[i]
            )

    def _before_physics_step(self):
        pass

    def _after_physics_step(self):
        return None
