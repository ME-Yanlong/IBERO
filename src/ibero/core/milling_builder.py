"""具体加工模型构建：共享装配产物，不引入通用多引擎容器。"""

from dataclasses import dataclass, replace
import mujoco
import numpy as np
from ibero.core.scene_loader import ValidatedScene
from ibero.materials.stock import VoxelStock
from ibero.materials.stock_collision import add_stock_geoms, StockCollisionBinding
from ibero.materials.parameters import StockParameters, strict_parameters, finite_number
from ibero.processes.tools import EndMillGeometry
from ibero.processes.milling_forces import MillingCoefficients, MillingLimits
from ibero.robots.spindle import add_spindle


@dataclass(frozen=True)
class CompiledMilling:
    """model/stock/binding 属于同一实例；运行时接管后不可跨环境共享可变状态。"""

    scene: ValidatedScene | None
    model: mujoco.MjModel
    stock: VoxelStock
    binding: StockCollisionBinding
    tool: EndMillGeometry
    coefficients: MillingCoefficients
    limits: MillingLimits
    start: np.ndarray
    axis_parameters: dict
    process_options: dict


def build_milling_fixture(
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
    axis_parameters = {
        "stiffness_n_m": finite_number(axis_stiffness_n_m, "axis_stiffness_n_m"),
        "damping_ns_m": finite_number(axis_damping_ns_m, "axis_damping_ns_m"),
        "force_limit_n": finite_number(axis_force_limit_n, "axis_force_limit_n"),
    }
    stock = VoxelStock(stock_parameters, cell_size_m, max_cells=max_cells)
    spec = mujoco.MjSpec()
    spec.option.timestep = timestep
    spec.option.gravity = [0, 0, 0]  # 辨识台架零重力；机器人场景必须恢复真实重力。
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    add_stock_geoms(spec, stock, contype=8)
    start = np.asarray(start, dtype=float)
    carriage = spec.worldbody.add_body(name="mill_carriage", pos=start)
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
    model = spec.compile()
    return CompiledMilling(
        None,
        model,
        stock,
        StockCollisionBinding(model, stock),
        tool,
        coefficients,
        limits,
        start,
        axis_parameters,
        dict(
            angular_samples=angular_samples,
            edge_transition_chip_m=edge_transition_chip_m,
        ),
    )


def compile_milling(scene):
    """已校验 P4 菜谱 → 台架或 G1 实体；保留原装配顺序、重力与执行器能力。"""
    cfg = scene.config
    tool = strict_parameters(EndMillGeometry, cfg["tool"])
    coefficients = strict_parameters(
        MillingCoefficients, cfg["process"]["coefficients"]
    )
    limits = strict_parameters(MillingLimits, cfg["process"]["limits"])
    parameters = strict_parameters(StockParameters, cfg["materials"]["stock"])
    numerics = cfg["numerics"]
    if cfg["kind"] == "milling_bench":
        result = build_milling_fixture(
            parameters,
            numerics["cell_size_m"],
            tool,
            coefficients,
            limits,
            timestep=cfg["physics"]["timestep_s"],
            start=cfg["initialization"]["tip_position_m"],
            max_cells=numerics["max_cells"],
            angular_samples=numerics["angular_samples"],
            edge_transition_chip_m=numerics["edge_transition_chip_m"],
            axis_stiffness_n_m=cfg["machine"]["axis_stiffness_n_m"],
            axis_damping_ns_m=cfg["machine"]["axis_damping_ns_m"],
            axis_force_limit_n=cfg["machine"]["axis_force_limit_n"],
        )
        return replace(result, scene=scene)
    if cfg["kind"] != "plate_milling":
        raise ValueError("Expected concrete milling recipe")
    from ibero.robots.g1_milling import milling_robot_spec, add_plate_support

    stock = VoxelStock(
        parameters,
        numerics["cell_size_m"],
        origin=cfg["workcell"]["stock_origin_m"],
        max_cells=numerics["max_cells"],
    )
    r = cfg["robot"]
    spec = milling_robot_spec(
        tool,
        limits,
        timestep=cfg["physics"]["timestep_s"],
        tip_offset_m=r["tip_offset_m"],
        bracket_mass_kg=r["bracket_mass_kg"],
    )
    spec.option.gravity = cfg["physics"]["gravity_m_s2"]
    add_stock_geoms(spec, stock, contype=8)
    add_plate_support(
        spec, origin=stock.origin, half_size=np.array(stock.params.size_m) / 2
    )
    model = spec.compile()
    return CompiledMilling(
        scene,
        model,
        stock,
        StockCollisionBinding(model, stock),
        tool,
        coefficients,
        limits,
        stock.local_to_world(cfg["initialization"]["tip_position_m"]),
        dict(r),
        dict(
            angular_samples=numerics["angular_samples"],
            edge_transition_chip_m=numerics["edge_transition_chip_m"],
            axis_tolerance_rad=r["axis_tolerance_rad"],
            holder_angular_limit_rad_s=r["holder_angular_limit_rad_s"],
        ),
    )
