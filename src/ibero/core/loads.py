"""每物理子步的外加广义力汇总；不碰执行器 ctrl 或外部校准 xfrc 通道。"""

import mujoco
import numpy as np


class AppliedLoads:
    def __init__(self, model):
        self.model = model
        self.generalized = np.zeros(model.nv)
        self._open = False

    def begin(self):
        if self._open:
            raise RuntimeError("Previous load substep has not been committed")
        self.generalized.fill(0)
        self._open = True

    def add_generalized(self, indices, values):
        if not self._open:
            raise RuntimeError("Call begin once before collecting this substep's loads")
        values = np.asarray(values, dtype=float)
        indices = np.asarray(indices)
        if (
            indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= self.model.nv)
        ):
            raise ValueError("Generalized load indices must refer to actual dofs")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite material load")
        np.add.at(self.generalized, indices, values)

    def add_wrench(self, data, body_id, force, torque, point):
        """世界系力/纯力矩与世界系作用点；力臂由 MuJoCo Jacobian 处理。"""
        if not self._open:
            raise RuntimeError("Call begin before add_wrench")
        vectors = [np.asarray(v, dtype=float) for v in (force, torque, point)]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError("Wrench vectors must be finite length-three arrays")
        if (
            isinstance(body_id, (bool, np.bool_))
            or not isinstance(body_id, (int, np.integer))
            or not 0 <= body_id < self.model.nbody
        ):
            raise ValueError("Unknown body id")
        mujoco.mj_applyFT(self.model, data, *vectors, body_id, self.generalized)

    def commit(self, data):
        if not self._open:
            raise RuntimeError("No open load substep, or already committed")
        if not np.isfinite(self.generalized).all():
            raise ValueError("Accumulated loads exceed numerical range")
        data.qfrc_applied[:] = self.generalized
        self._open = False
