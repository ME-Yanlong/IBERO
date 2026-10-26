"""从真实线束中心线独立计算夹座匹配、弧长顺序和路径误差。

本模块只消费已经采样的世界坐标，既不读取控制器目标位姿，也不对
MuJoCo 状态施力。线束节点稀疏时，采用线段和盒体求交，避免漏掉两
个节点之间实际穿过夹座的情况。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ClipMatch:
    """某个夹座与目标线束当前中心线的只读几何关系。"""

    seated: bool
    mouth: bool
    arc_m: float | None
    lateral_error_m: float
    depth_m: float | None


def _points(value) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("Cable must have at least two 3-D centerline points")
    if not np.isfinite(points).all():
        raise ValueError("Cable centerline contains non-finite coordinates")
    return points


def _clip_frame(clip: Mapping) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(clip["origin_m"], dtype=float)
    rotation = np.asarray(clip["rotation_world"], dtype=float)
    if origin.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("Clip origin/rotation dimensions are invalid")
    if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
        raise ValueError("Clip frame must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("Clip rotation must be orthonormal")
    if np.linalg.det(rotation) < 0.99:
        raise ValueError("Clip rotation must preserve handedness")
    return origin, rotation


def _segment_box_interval(a: np.ndarray, b: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    """Return the interval of a segment lying inside an axis-aligned 3-D box."""
    start, stop = 0.0, 1.0
    delta = b - a
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            if a[axis] < lower[axis] or a[axis] > upper[axis]:
                return None
            continue
        enter = (lower[axis] - a[axis]) / delta[axis]
        leave = (upper[axis] - a[axis]) / delta[axis]
        if enter > leave:
            enter, leave = leave, enter
        start, stop = max(start, enter), min(stop, leave)
        if start > stop:
            return None
    return start, stop


def _best_box_match(local: np.ndarray, lengths: np.ndarray, lower, upper, *, min_span_m=0.001):
    """Find a real cable segment through the box, favoring the seat center."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    best = None
    arc_start = 0.0
    for index, (a, b) in enumerate(zip(local[:-1], local[1:])):
        interval = _segment_box_interval(a, b, lower, upper)
        if (interval is not None and lengths[index] > 1e-10
                and (interval[1] - interval[0]) * lengths[index] >= min_span_m):
            # 平行夹座轴线的线段有效；横切夹座的线段不算已正确铺入。
            tangent = (b - a) / lengths[index]
            if abs(tangent[1]) >= 0.45:
                lo, hi = interval
                direction = b - a
                center = (lower + upper) / 2
                t = np.clip(np.dot(center - a, direction) / max(np.dot(direction, direction), 1e-15), lo, hi)
                point = a + t * direction
                rank = float(np.linalg.norm(point - center))
                candidate = (rank, arc_start + float(t) * lengths[index], point)
                if best is None or rank < best[0]:
                    best = candidate
        arc_start += lengths[index]
    return best


def match_clip(cable_points_m, clip: Mapping) -> ClipMatch:
    """Check a cylindrical cable section against a side-entry clip's local seat.

    坐标约定：x 横跨夹口，y 沿线束，z 从布线板指向机器人。
    `seated` 检查位于唇边后方的整个线材中心；回弹和接触有效性由
    机构观察器提供，不能用这个几何函数直接宣称最终保持。
    """
    points = _points(cable_points_m)
    origin, rotation = _clip_frame(clip)
    local = (points - origin) @ rotation
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    width = float(clip["seat_half_width_m"])
    half_length = float(clip["seat_half_length_m"])
    z_min = float(clip["seat_z_min_m"])
    z_max = float(clip["seat_z_max_m"])
    lip = float(clip["lip_z_m"])
    if not (np.isfinite([width, half_length, z_min, z_max, lip]).all()
            and width > 0 and half_length > 0 and 0 <= z_min < z_max < lip):
        raise ValueError("Clip seat bounds are inconsistent")
    lower = [-width, -half_length, z_min]
    upper = [width, half_length, z_max]
    seat = _best_box_match(local, lengths, lower, upper, min_span_m=0.002)
    # 入口观察窗略宽于夹座，覆盖唇边外的一段进给距离。
    mouth = _best_box_match(
        local,
        lengths,
        [-width * 1.5, -half_length * 1.5, lip],
        [width * 1.5, half_length * 1.5, lip + max(0.015, lip - z_max)],
    )
    return ClipMatch(
        seated=seat is not None,
        mouth=mouth is not None,
        arc_m=None if seat is None else float(seat[1]),
        lateral_error_m=float("inf") if seat is None else abs(float(seat[2][0])),
        depth_m=None if seat is None else float(seat[2][2]),
    )


def _closest_route(point: np.ndarray, route: np.ndarray):
    best_dist, best_arc = float("inf"), 0.0
    traveled = 0.0
    for a, b in zip(route[:-1], route[1:]):
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            continue
        t = float(np.clip(np.dot(point - a, delta) / (length * length), 0, 1))
        distance = float(np.linalg.norm(point - (a + t * delta)))
        if distance < best_dist:
            best_dist, best_arc = distance, traveled + t * length
        traveled += length
    return best_dist, best_arc


def _interpolate_at_arc(points: np.ndarray, cumulative: np.ndarray, arc_m: float) -> np.ndarray:
    index = min(int(np.searchsorted(cumulative, arc_m, side="right")) - 1, len(points) - 2)
    index = max(index, 0)
    span = cumulative[index + 1] - cumulative[index]
    t = 0.0 if span <= 1e-12 else (arc_m - cumulative[index]) / span
    return points[index] * (1 - t) + points[index + 1] * t


def inspect_route(cable_points_m, route_points_m, start_arc_m: float, end_arc_m: float,
                  *, backtrack_tolerance_m: float = 0.0) -> tuple[float, bool]:
    """Check the cable between first and last clip, excluding free tails.

    返回最大路径误差和沿目标路线的投影是否保持单调。弧长顺序由调用方
    单独核验；这里的单调性防止线束在走廊内折返冒充正确布线。
    """
    points = _points(cable_points_m)
    route = _points(route_points_m)
    if len(route) < 2 or not np.any(np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-12):
        raise ValueError("Route must contain a nonzero segment")
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if not (0 <= start_arc_m <= end_arc_m <= cumulative[-1] + 1e-9):
        raise ValueError("Clip arc-length interval lies outside the cable")
    samples = [_interpolate_at_arc(points, cumulative, start_arc_m)]
    samples.extend(points[(cumulative > start_arc_m) & (cumulative < end_arc_m)])
    samples.append(_interpolate_at_arc(points, cumulative, end_arc_m))
    # 线束线段的中点可发现仅端点看起来接近路线时的越界。
    dense = [samples[0]]
    for a, b in zip(samples[:-1], samples[1:]):
        dense.extend((0.5 * (a + b), b))
    deviations, progress = zip(*(_closest_route(point, route) for point in dense))
    monotone = all(
        later + backtrack_tolerance_m >= earlier
        for earlier, later in zip(progress[:-1], progress[1:])
    )
    return max(deviations), monotone


def inspect_order(clip_ids: tuple[str, ...], matches: Mapping[str, ClipMatch],
                  min_separation_m: float) -> bool:
    """Each requested clip must hold a distinct later material section."""
    arcs = [matches[clip_id].arc_m for clip_id in clip_ids]
    return bool(all(arc is not None for arc in arcs) and all(
        right - left >= min_separation_m for left, right in zip(arcs[:-1], arcs[1:])
    ))
