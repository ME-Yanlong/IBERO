"""卡扣与线束的实体组合：复用中心线 Flex 力律，不再创建第二只端头。"""

from dataclasses import dataclass
import mujoco
import numpy as np
from ibero.materials.parameters import finite_number


@dataclass(frozen=True)
class HarnessParameters:
    """SI 物性均来自场景；端头质量/尺寸由卡扣唯一拥有，避免双计质量。"""

    length_m: float
    outer_diameter_m: float
    linear_density_kg_m: float
    minimum_bend_radius_m: float
    axial_stiffness_n_m: float
    bending_stiffness_nm2: float
    damping: float
    friction: float
    segments: int
    source: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name not in {"segments", "source"}:
                finite_number(getattr(self, name), name)
        if type(self.segments) is not int or not 4 <= self.segments <= 32:
            raise ValueError("Harness segments must be an integer in [4, 32]")
        if self.source != "provisional":
            raise ValueError("Harness currently requires explicit provisional source")
        if self.outer_diameter_m >= self.length_m / self.segments:
            raise ValueError("Cable diameter must be smaller than one segment")


def add_harness(spec, params, *, origin, quaternion, tail_offset_m):
    """真实自由节点 + 永久材料端点 connect；不连机器人、不切换 active。

    初始正弦松弛由弧长求解，不将端点距离误当材料原长。圆线的无扭转
    弯曲仍由 CableMechanics 提供；轴向边弹簧/阻尼由 MuJoCo 原生 Flex 求解。
    """
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
    # 尾部应变释放点位于壳体后方 2 mm，避免初始节点嵌入刚体。
    anchor = np.array([-0.032, 0, -0.008])
    start = np.asarray(origin) + rotation.reshape(3, 3) @ anchor
    offset = np.asarray(tail_offset_m, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Finite tail offset required")
    if not 0 < np.linalg.norm(offset) < params.length_m:
        raise ValueError("Tail span must be positive and shorter than material length")
    t = np.linspace(0, 1, params.segments + 1)

    def points(sag):
        return start + t[:, None] * offset - np.outer(np.sin(np.pi * t), [0, 0, sag])

    low, high = 0.0, params.length_m
    for _ in range(60):
        mid = (low + high) / 2
        if np.linalg.norm(np.diff(points(mid), axis=0), axis=1).sum() > params.length_m:
            high = mid
        else:
            low = mid
    vertices = points((low + high) / 2)
    bodies = []
    node_mass = params.length_m * params.linear_density_kg_m / len(vertices)
    for index, point in enumerate(vertices):
        name = f"cable_flex_{index}"
        body = spec.worldbody.add_body(name=name, pos=point)
        for axis_index, axis in enumerate(np.eye(3)):
            body.add_joint(
                name=f"{name}_{axis_index}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis
            )
        # 无碰撞、不可见节点小球只定义质量/惯量；线束碰撞由 Flex capsule 唯一承担。
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[params.outer_diameter_m / 2],
            mass=node_mass,
            contype=0,
            conaffinity=0,
            rgba=[0, 0, 0, 0],
        )
        bodies.append(name)
    spec.add_flex(
        name="cable_flex",
        dim=1,
        radius=params.outer_diameter_m / 2,
        vertbody=bodies,
        vert=np.zeros(len(vertices) * 3),
        elem=np.column_stack(
            (np.arange(params.segments), np.arange(1, len(vertices)))
        ).ravel(),
        edgestiffness=params.axial_stiffness_n_m * params.segments,
        edgedamping=params.damping * params.segments,
        contype=2,
        conaffinity=1,
        # 2.5 μs 卡扣工况采用 1 ms 接触时间常数；默认 20 ms 在细线落盘时穿透过大。
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
        selfcollide=mujoco.mjtFlexSelf.mjFLEXSELF_NONE,
        friction=[params.friction, 0.005, 0.0001],
        rgba=[0.95, 0.2, 0.04, 1],
    )
    data = np.zeros(11)
    data[:3] = anchor
    spec.add_equality(
        name="harness_material_attachment",
        type=mujoco.mjtEq.mjEQ_CONNECT,
        name1="latch_plug",
        name2=bodies[0],
        data=data,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        solref=[0.001, 1],
        solimp=[0.999, 0.999, 0.001, 0.5, 2],
    )
    return vertices
