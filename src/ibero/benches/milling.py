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
from ibero.processes.milling import MillingProcess


def add_spindle(
    spec,
    parent,
    tool,
    *,
    prefix="mill_",
    rotor_mass_kg=0.02,
    housing_mass_kg=0.08,
    torque_limit_nm=0.15,
):
    """可复用工具：端面/侧刃圆柱与非切削刀柄分开，转速来自实际转子关节。"""
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
    )
    mount.add_site(name=prefix + "tip", pos=[0, 0, 0], size=[0.001])
    rotor = mount.add_body(name=prefix + "rotor")
    rotor.add_joint(
        name=prefix + "spindle",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        limited=False,
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
    ):
        self.stock = VoxelStock(stock_parameters, cell_size_m, max_cells=60000)
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
            actuator.set_to_position(kp=5000, kv=80)
            actuator.forcelimited = True
            actuator.forcerange = [-50, 50]
        add_spindle(spec, carriage, tool, torque_limit_nm=limits.max_torque_nm)
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
        )
        self.reset(seed=0)

    def reset(self, *, seed=0):
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

    def step(self, target, rpm, *, substeps=50, fault_at=None):
        if self._done or self._replay_restored:
            raise RuntimeError("Finished/replayed milling must reset")
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (3,)
            or not np.isfinite(target).all()
            or not np.isfinite(rpm)
            or rpm < 0
        ):
            raise ValueError("Finite target and nonnegative spindle command required")
        if type(substeps) is not int or not 1 <= substeps <= 5000:
            raise ValueError("Invalid milling substep count")
        for i in range(3):
            self.data.ctrl[self.model.actuator(f"mill_axis_{i}").id] = (
                target[i] - self.start[i]
            )
        self.data.ctrl[self.model.actuator("mill_motor").id] = rpm * np.pi / 30
        total_removed = 0.0
        for _ in range(substeps):
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
            if state.invalid_reason:
                self._done = True
                break
            total_removed += state.removed_volume_m3
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
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
        info = asdict(state)
        info.update(
            time_s=float(self.data.time),
            removed_volume_step_m3=total_removed,
            total_removed_volume_m3=self.stock.initial_volume_m3 - self.stock.volume_m3,
            tip_position_m=self.data.site("mill_tip").xpos.copy().tolist(),
            invalid_reason=self.process.invalid_reason,
            fixture_reaction_force_n=(-np.asarray(state.force_world_n)).tolist(),
            fixture_reaction_torque_about_tip_nm=(
                -np.asarray(state.torque_world_nm)
            ).tolist(),
            ft_force_n=self.data.sensor("mill_force").data.copy().tolist(),
            ft_torque_nm=self.data.sensor("mill_torque").data.copy().tolist(),
        )
        self.last_info = info
        return info
