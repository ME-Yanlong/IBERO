"""同一物理子步的已准备平均载荷核：几何/系数只预计算一次，速度仍逐次求解。"""

import math
import numpy as np
from ibero.materials.parameters import finite_number


class PreparedMillingWrench:
    """与公开平均公式等价的线性系数展开，不缓存跨物理步的啮合或载荷。

    隐式 Newton 会在相同啮合上多次询问不同速度。径向基、力矩臂及材料
    系数只与当前几何有关，可以准备为 6×角度数矩阵，而不是每次重复三角函数。
    """

    def __init__(
        self,
        coefficients,
        *,
        radius_m,
        teeth,
        lengths_m,
        centroids_m,
        face_fraction,
        edge_transition_chip_m,
        center_cutting,
    ):
        finite_number(radius_m, "radius_m")
        finite_number(face_fraction, "face_fraction", positive=False)
        finite_number(edge_transition_chip_m, "edge_transition_chip_m", positive=False)
        if (
            type(teeth) is not int
            or teeth < 1
            or face_fraction > 1
            or type(center_cutting) is not bool
        ):
            raise ValueError("Invalid prepared milling geometry")
        a, z = np.asarray(lengths_m, dtype=float), np.asarray(centroids_m, dtype=float)
        if (
            a.ndim != 1
            or len(a) < 8
            or a.shape != z.shape
            or not np.isfinite(a).all()
            or not np.isfinite(z).all()
            or np.any(a < 0)
        ):
            raise ValueError("Invalid prepared engagement")
        phi = (np.arange(len(a)) + 0.5) * (2 * np.pi / len(a))
        radial = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(len(a))))
        tangent = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(len(a))))
        self.radial_xy = radial[:, :2].T.copy()
        c = coefficients
        shear = -c.tangential_pa * tangent - c.radial_pa * radial
        edge = -c.tangential_edge_n_m * tangent - c.radial_edge_n_m * radial
        shear[:, 2], edge[:, 2] = c.axial_pa, c.axial_edge_n_m
        shear *= a[:, None]
        edge *= a[:, None]
        points = radius_m * radial
        points[:, 2] = z
        self.shear = (
            np.column_stack((shear, np.cross(points, shear))) * (teeth / len(a))
        ).T.copy()
        self.edge = (
            np.column_stack((edge, np.cross(points, edge))) * (teeth / len(a))
        ).T.copy()
        self.face_force = c.face_axial_pa * radius_m * teeth * face_fraction
        self.face_torque = -c.face_tangent_pa * radius_m**2 / 2 * teeth * face_fraction
        self.teeth, self.delta, self.center_cutting = (
            teeth,
            edge_transition_chip_m,
            center_cutting,
        )

    def __call__(self, velocity_tool, rpm):
        v = np.asarray(velocity_tool, dtype=float)
        if (
            v.shape != (3,)
            or not np.isfinite(v).all()
            or not math.isfinite(rpm)
            or rpm < 0
        ):
            raise ValueError("Finite prepared milling velocity/rpm required")
        if rpm == 0:
            return np.zeros(6)
        divisor = rpm / 60 * self.teeth
        chip = np.maximum(v[:2] @ self.radial_xy, 0) / divisor
        active = chip > 1e-15
        if np.linalg.norm(v[:2]) <= 1e-12:
            active[:] = False
        weight = chip / (chip + self.delta) if self.delta else np.ones_like(chip)
        result = self.shear @ (chip * active) + self.edge @ (weight * active)
        if v[2] < -1e-12 and self.face_force > 0:
            if not self.center_cutting:
                raise ValueError("center_cutting_not_supported")
            face_chip = -v[2] / divisor
            result[2] += self.face_force * face_chip
            result[5] += self.face_torque * face_chip
        return result
