"""端铣刀有效刃区扫掠：平底圆柱，刀柄不参与材料删除。"""

from dataclasses import dataclass
import math
import numpy as np
from ibero.materials.parameters import finite_number


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    if (
        q.shape != (4,)
        or not np.isfinite(q).all()
        or not math.isclose(float(q @ q), 1, abs_tol=1e-10)
    ):
        raise ValueError("Unit scalar-first quaternion required")
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class ToolPose:
    position: tuple
    quaternion: tuple = (1, 0, 0, 0)

    def __post_init__(self):
        p = np.asarray(self.position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Finite tool tip position required")
        quaternion_matrix(self.quaternion)
        object.__setattr__(self, "position", tuple(float(x) for x in p))
        object.__setattr__(self, "quaternion", tuple(float(x) for x in self.quaternion))

    @property
    def rotation(self):
        return quaternion_matrix(self.quaternion)


@dataclass(frozen=True)
class EndMillGeometry:
    radius_m: float
    cutting_length_m: float
    shank_radius_m: float
    shank_length_m: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            finite_number(getattr(self, name), name)

    def contains_cutting_points(self, world_points, pose):
        local = (np.asarray(world_points) - pose.position) @ pose.rotation
        return (
            (np.sum(local[..., :2] ** 2, axis=-1) <= self.radius_m**2)
            & (local[..., 2] >= 0)
            & (local[..., 2] <= self.cutting_length_m)
        )


def interpolate_poses(
    start, end, *, max_displacement_m, radius_bound_m, max_samples=10000
):
    """平移＋姿态的刀尖/刃区位移上界细分；超容量拒绝而不是跳帧漏切。"""
    finite_number(max_displacement_m, "max_displacement_m")
    q0, q1 = np.asarray(start.quaternion), np.asarray(end.quaternion)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    angle = math.acos(np.clip(dot, -1, 1))
    span = (
        np.linalg.norm(np.asarray(end.position) - start.position)
        + 2 * angle * radius_bound_m
    )
    n = max(1, math.ceil(span / max_displacement_m))
    if n + 1 > max_samples:
        raise ValueError("Swept tool path exceeds subdivision capacity")
    for t in np.linspace(0, 1, n + 1):
        q = (
            ((1 - t) * q0 + t * q1)
            if dot > 0.9995
            else (math.sin((1 - t) * angle) * q0 + math.sin(t * angle) * q1)
            / math.sin(angle)
        )
        q /= np.linalg.norm(q)
        yield ToolPose(
            tuple((1 - t) * np.asarray(start.position) + t * np.asarray(end.position)),
            tuple(q),
        )


def swept_cells(stock, tool, start, end):
    """返回纯几何候选 ID，包含空格以支持稳定幂等；不在这里删材料或施力。"""
    bound = math.hypot(tool.radius_m, tool.cutting_length_m)
    # 保守局部 AABB 缩小候选，旋转后仍包含全部圆柱刃区。
    tips = stock.world_to_local([start.position, end.position])
    lower, upper = tips.min(axis=0) - bound, tips.max(axis=0) + bound
    candidates = np.flatnonzero(
        np.all((stock.centers >= lower) & (stock.centers <= upper), axis=1)
    )
    points = stock.local_to_world(stock.centers[candidates])
    hit = np.zeros(len(candidates), dtype=bool)
    for pose in interpolate_poses(
        start, end, max_displacement_m=stock.cell_size_m / 4, radius_bound_m=bound
    ):
        hit |= tool.contains_cutting_points(points, pose)
    return candidates[hit]
