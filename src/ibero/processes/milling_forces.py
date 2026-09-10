"""旋转平均端铣载荷，SI 单位；示例系数不是钢材实测数据库。"""

from dataclasses import dataclass
import math
import numpy as np
from ibero.materials.parameters import finite_number


@dataclass(frozen=True)
class MillingCoefficients:
    """侧刃剪切系数 Pa、刃口系数 N/m；端面采用独立工程系数。

    侧刃形式参照 UBC/MAL 机械式模型，具体数值须由场景显式提供。
    face_tangent_pa / face_axial_pa 是首版平底中心刃的均匀切屑厚度近似，
    不是从侧刃系数自动推断的真实钻削/端铣标定。
    """

    tangential_pa: float
    radial_pa: float
    axial_pa: float
    tangential_edge_n_m: float
    radial_edge_n_m: float
    axial_edge_n_m: float
    face_tangent_pa: float
    face_axial_pa: float
    material_grade: str
    source: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name not in {"material_grade", "source"}:
                finite_number(getattr(self, name), name, positive="edge" not in name)
        if (
            self.material_grade != "AISI1045-provisional"
            or self.source != "provisional"
        ):
            raise ValueError(
                "Only explicitly provisional AISI1045 coefficients are supported"
            )


@dataclass(frozen=True)
class MillingLimits:
    min_rpm: float
    max_rpm: float
    max_feed_per_tooth_m: float
    max_axial_depth_m: float
    max_force_n: float
    max_torque_nm: float
    max_power_w: float
    teeth: int
    center_cutting: bool

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name not in {"teeth", "center_cutting"}:
                finite_number(getattr(self, name), name)
        if (
            self.min_rpm >= self.max_rpm
            or type(self.teeth) is not int
            or not 1 <= self.teeth <= 12
        ):
            raise ValueError("Invalid spindle range/teeth")
        if type(self.center_cutting) is not bool:
            raise ValueError("center_cutting must explicitly be boolean")


def mean_side_wrench(
    coefficients,
    *,
    radius_m,
    rpm,
    teeth,
    velocity_tool,
    axial_lengths_m,
    axial_centroids_m=None,
):
    """按周向中点积分平均载荷，输出关于刀尖的力与力矩。

    每齿厚度 h=max(v·n,0)/(转/秒×齿数)；每个方位的真实啮合轴向长度
    由剩余材料提供。有效剪切面以外不沿用预设力曲线。+Z 为刀轴向上，
    正主轴沿 +Z 旋转；切向力反向，径向力向刀轴，轴向分量向上。
    """
    finite_number(radius_m, "radius")
    finite_number(rpm, "rpm", positive=False)
    if type(teeth) is not int or teeth <= 0:
        raise ValueError("Positive integral teeth required")
    lengths = np.asarray(axial_lengths_m, dtype=float)
    velocity = np.asarray(velocity_tool, dtype=float)
    if (
        lengths.ndim != 1
        or len(lengths) < 8
        or not np.isfinite(lengths).all()
        or np.any(lengths < 0)
        or velocity.shape != (3,)
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("Invalid engagement/velocity")
    if rpm == 0 or np.linalg.norm(velocity[:2]) <= 1e-12 or not lengths.any():
        return np.zeros(3), np.zeros(3)
    phi = (np.arange(len(lengths)) + 0.5) * (2 * np.pi / len(lengths))
    radial = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(len(phi))))
    tangent = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(len(phi))))
    chip = np.maximum(radial @ velocity, 0) / (rpm / 60 * teeth)
    # 后退半周不切当前新材料；刃口项只计实际前进且已啮合的刃段。
    active = chip > 1e-15
    a = lengths * active
    c = coefficients
    ft = a * (c.tangential_pa * chip + c.tangential_edge_n_m)
    fr = a * (c.radial_pa * chip + c.radial_edge_n_m)
    fa = a * (c.axial_pa * chip + c.axial_edge_n_m)
    forces = -ft[:, None] * tangent - fr[:, None] * radial
    forces[:, 2] += fa
    z = (
        lengths / 2
        if axial_centroids_m is None
        else np.asarray(axial_centroids_m, dtype=float)
    )
    if z.shape != lengths.shape or not np.isfinite(z).all():
        raise ValueError("Invalid axial centroids")
    positions = radial * radius_m
    positions[:, 2] = z
    weight = teeth / len(lengths)
    return forces.sum(axis=0) * weight, np.cross(positions, forces).sum(axis=0) * weight


def mean_face_wrench(
    coefficients, *, radius_m, rpm, teeth, axial_velocity_m_s, engagement_fraction
):
    """已声明中心下切刀具的端面近似；每刃沿半径均匀厚度，无微观断裂模型。

    轴向力按 ∫K_z h dr，扭矩按 ∫r K_t h dr；向上退出不产生该去除载荷。
    端面占据比例是离散几何估计，偏心啮合的瞬态横向力不在本模型范围。
    """
    finite_number(radius_m, "radius")
    finite_number(rpm, "rpm", positive=False)
    finite_number(engagement_fraction, "engagement", positive=False)
    finite_number(axial_velocity_m_s, "axial velocity", minimum=-100, positive=False)
    if engagement_fraction > 1 or type(teeth) is not int or teeth < 1:
        raise ValueError("Invalid face engagement/teeth")
    if rpm == 0 or axial_velocity_m_s >= 0 or engagement_fraction == 0:
        return np.zeros(3), np.zeros(3)
    chip = -axial_velocity_m_s / (rpm / 60 * teeth)
    force = coefficients.face_axial_pa * chip * radius_m * teeth * engagement_fraction
    torque = (
        -coefficients.face_tangent_pa
        * chip
        * radius_m**2
        / 2
        * teeth
        * engagement_fraction
    )
    return np.array([0, 0, force]), np.array([0, 0, torque])


def capacity_reason(
    limits, *, rpm, velocity_tool, axial_depth_m, force_tool, torque_tool
):
    """仅判能力；不截断力后仍删除原数量的材料。超限由调用者停止提交/积分。"""
    finite_number(rpm, "rpm", positive=False)
    finite_number(axial_depth_m, "axial depth", positive=False)
    for vector in (velocity_tool, force_tool, torque_tool):
        array = np.asarray(vector)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError("Finite velocity/force/torque vectors required")
    if rpm < limits.min_rpm:
        return "spindle_not_ready"
    if rpm > limits.max_rpm:
        return "spindle_out_of_range"
    per_tooth = np.linalg.norm(velocity_tool) / (rpm / 60 * limits.teeth)
    if per_tooth > limits.max_feed_per_tooth_m:
        return "feed_out_of_range"
    if axial_depth_m > limits.max_axial_depth_m:
        return "axial_depth_out_of_range"
    if np.linalg.norm(force_tool) > limits.max_force_n:
        return "cutting_force_overload"
    if abs(torque_tool[2]) > limits.max_torque_nm:
        return "spindle_torque_overload"
    if abs(torque_tool[2] * rpm * 2 * math.pi / 60) > limits.max_power_w:
        return "spindle_power_overload"
    return None
