"""自由裸线束：MuJoCo 1D Flex 接触与既有中心线弯曲力共用同一材料参数。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class RoutingCableParameters:
    length_m: float
    outer_diameter_m: float
    linear_density_kg_m: float
    minimum_bend_radius_m: float
    axial_stiffness_n_m: float
    bending_stiffness_nm2: float
    damping: float
    friction: float
    segments: int
    parameter_source: str

    @classmethod
    def from_mapping(cls, values):
        return cls(**values)


def add_routing_cable(spec: mujoco.MjSpec, params: RoutingCableParameters, points_m):
    """只连接相邻自由节点；不把线束端点焊接到机器人或世界。"""
    points = np.asarray(points_m, dtype=float)
    if points.shape != (params.segments + 1, 3) or not np.isfinite(points).all():
        raise ValueError("Cable points must be finite (segments+1, 3) coordinates")
    rest_length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    if abs(rest_length - params.length_m) > 0.005 * params.length_m:
        raise ValueError("Initial centerline arc must match declared material length")
    body_names = []
    node_mass = params.length_m * params.linear_density_kg_m / len(points)
    for i, point in enumerate(points):
        name = f"cable_flex_{i}"
        body = spec.worldbody.add_body(name=name, pos=point)
        for j, axis in enumerate(np.eye(3)):
            body.add_joint(name=f"{name}_{j}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
        # 节点只给 Flex 质量，不另造一层球形接触代理。
        body.add_geom(
            name=f"{name}_mass",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[params.outer_diameter_m / 2],
            mass=node_mass,
            contype=0,
            conaffinity=0,
            rgba=[0, 0, 0, 0],
        )
        body_names.append(name)
    spec.add_flex(
        name="cable_flex",
        dim=1,
        radius=params.outer_diameter_m / 2,
        vertbody=body_names,
        vert=np.zeros(len(points) * 3),
        elem=np.column_stack((np.arange(params.segments), np.arange(1, len(points)))).ravel(),
        edgestiffness=params.axial_stiffness_n_m * params.segments,
        edgedamping=params.damping * params.segments,
        contype=2,
        conaffinity=1,
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
        selfcollide=mujoco.mjtFlexSelf.mjFLEXSELF_NONE,
        friction=[params.friction, 0.005, 0.0001],
        rgba=[0.95, 0.28, 0.07, 1],
    )
    return tuple(body_names)


def free_cable_points(start, end, *, length_m, segments):
    """用垂向正弦弧初始化松线，二分保证线束弧长等于材料原长。"""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("Cable endpoints must be three-dimensional")
    span = float(np.linalg.norm(end - start))
    if not 0 < span <= length_m or segments < 4:
        raise ValueError("Cable length must cover a positive endpoint span")
    t = np.linspace(0, 1, segments + 1)

    def points(amplitude):
        return start + t[:, None] * (end - start) - np.outer(np.sin(np.pi * t), [0, 0, amplitude])

    lo, hi = 0.0, length_m
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.linalg.norm(np.diff(points(mid), axis=0), axis=1).sum() > length_m:
            hi = mid
        else:
            lo = mid
    return points((lo + hi) / 2)
