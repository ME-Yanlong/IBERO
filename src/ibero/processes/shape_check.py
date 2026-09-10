"""独立形状检测：解析圆角矩形/圆孔与实际占据比较，不调用刀具扫掠。"""

from dataclasses import dataclass
import math
import numpy as np
from ibero.materials.parameters import finite_number


@dataclass(frozen=True)
class MachiningTarget:
    shape: str
    center_xy_m: tuple
    half_straight_xy_m: tuple
    corner_radius_m: float
    depth_m: float

    def __post_init__(self):
        if self.shape not in {"slot", "pocket", "through_hole"}:
            raise ValueError("Unknown target shape")
        for name in ("center_xy_m", "half_straight_xy_m"):
            v = np.asarray(getattr(self, name), dtype=float)
            if v.shape != (2,) or not np.isfinite(v).all():
                raise ValueError("Finite planar target geometry required")
            object.__setattr__(self, name, tuple(float(x) for x in v))
        a, b = self.half_straight_xy_m
        if (
            min(a, b) < 0
            or (self.shape == "through_hole" and (a or b))
            or (self.shape == "slot" and b)
        ):
            raise ValueError("Target type and straight dimensions disagree")
        finite_number(self.corner_radius_m, "corner_radius_m")
        finite_number(self.depth_m, "depth_m")

    @property
    def volume_m3(self):
        a, b = self.half_straight_xy_m
        r = self.corner_radius_m
        return (4 * a * b + 4 * r * (a + b) + math.pi * r * r) * self.depth_m

    def planar_sdf(self, xy):
        d = np.abs(np.asarray(xy) - self.center_xy_m) - self.half_straight_xy_m
        return (
            np.linalg.norm(np.maximum(d, 0), axis=-1)
            + np.minimum(np.max(d, axis=-1), 0)
            - self.corner_radius_m
        )


def inspect_shape(
    stock, target, *, volume_error_fraction=0.05, boundary_error_cells=2.0
):
    """体积不能单独防止偏切：同时核对删除外边界、目标边界覆盖及实际孔列。"""
    volume_error_fraction = finite_number(
        volume_error_fraction, "volume_error_fraction"
    )
    boundary_error_cells = finite_number(boundary_error_cells, "boundary_error_cells")
    if volume_error_fraction > 0.05 or boundary_error_cells > 2:
        raise ValueError("Shape inspection cannot relax frozen maximum criteria")
    size = np.asarray(stock.params.size_m)
    if target.depth_m > size[2] + 1e-12:
        raise ValueError("Target depth exceeds fixed stock")
    if np.any(
        np.abs(target.center_xy_m)
        + np.asarray(target.half_straight_xy_m)
        + target.corner_radius_m
        >= size[:2] / 2
    ):
        raise ValueError("Target must be internal; free offcuts are unsupported")
    centers = stock.centers
    removed = ~stock.occupied
    bottom = size[2] / 2 - target.depth_m
    sdf = target.planar_sdf(centers[:, :2])
    expected = (sdf <= 0) & (centers[:, 2] >= bottom)
    volume = float(stock.initial_volume_m3 - stock.volume_m3)
    error = abs(volume / target.volume_m3 - 1)
    missing = float(stock.volumes[expected & ~removed].sum())
    extra = float(stock.volumes[removed & ~expected].sum())
    occ = removed.reshape(stock.shape)
    # 每个六邻域外边界面的中心与四角，包含水平底面，不能只看俯视孔轮廓。
    points = []
    for axis in range(3):
        for sign in (-1, 1):
            neighbor = np.roll(occ, -sign, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = -1 if sign == 1 else 0
            neighbor[tuple(edge)] = False
            ids = np.flatnonzero((occ & ~neighbor).ravel())
            if not len(ids):
                continue
            p = centers[ids].copy()
            p[:, axis] += sign * stock.half_sizes[ids, axis]
            # 顶面与通孔底出口不是设计内部壁面，排除毛坯外表面。
            valid = np.abs(p[:, axis]) < size[axis] / 2 - 1e-12
            p, ids = p[valid], ids[valid]
            points.append(p)
            others = [i for i in range(3) if i != axis]
            for s0, s1 in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                q = p.copy()
                q[:, others[0]] += s0 * stock.half_sizes[ids, others[0]]
                q[:, others[1]] += s1 * stock.half_sizes[ids, others[1]]
                points.append(q)
    if points and any(len(p) for p in points):
        points = np.concatenate(points)
        planar = target.planar_sdf(points[:, :2])
        # 目标是挤出的截面：内部壁距离取侧壁与内底面的最小值；外侧取组合距离。
        vertical = bottom - points[:, 2]
        distances = np.linalg.norm(
            np.maximum(np.column_stack((planar, vertical)), 0), axis=1
        ) + np.minimum(np.maximum(planar, vertical), 0)
        boundary = float(np.max(np.abs(distances)))
    else:
        boundary = float(np.linalg.norm(size))
    # 内部保守点应已删除，外部保守点应存在；窄带读取菜谱，可比缺省两格更严格。
    band = boundary_error_cells * stock.cell_size_m
    definitely_inside = (sdf < -band) & (centers[:, 2] > bottom + band)
    definitely_outside = (sdf > band) | (centers[:, 2] < bottom - band)
    coverage = bool(
        np.all(removed[definitely_inside]) and not np.any(removed[definitely_outside])
    )
    column = np.floor(
        (np.asarray(target.center_xy_m) + size[:2] / 2) / stock.cell_size_m
    ).astype(int)
    through = (
        bool(occ[column[0], column[1], :].all())
        if target.shape == "through_hole"
        else True
    )
    return {
        "shape": target.shape,
        "expected_volume_m3": target.volume_m3,
        "removed_volume_m3": volume,
        "volume_relative_error": error,
        "missing_volume_m3": missing,
        "extra_volume_m3": extra,
        "boundary_error_m": boundary,
        "boundary_error_cells": boundary / stock.cell_size_m,
        "interior_coverage": coverage,
        "through_column_open": through,
        "volume_error_limit_fraction": volume_error_fraction,
        "boundary_error_limit_cells": boundary_error_cells,
        "passed": bool(
            error <= volume_error_fraction and boundary <= band and coverage and through
        ),
    }
