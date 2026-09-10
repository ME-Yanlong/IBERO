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


class PhysicalLoads(AppliedLoads):
    """加工的刚体受力位置与传感载荷通道；旧 AppliedLoads 行为不变。

    qfrc_applied 能改变运动，却丢失刚体受力位置，静态腕部 F/T 可能读成零。
    刚体载荷改为 xfrc_applied 自有增量，以 COM 力矩换算供引擎反推内部力。
    外部校准应每子步 add_wrench 或以加法注入 xfrc；不能直接覆盖同一自有分量。
    """

    def __init__(self, model):
        super().__init__(model)
        self.body_wrenches = np.zeros((model.nbody, 6))
        self.external_body = np.zeros((model.nbody, 6))
        self._last_body = np.zeros((model.nbody, 6))

    def reset(self):
        self._last_body.fill(0)
        self.body_wrenches.fill(0)
        self.external_body.fill(0)
        self.generalized.fill(0)
        self._open = False

    def begin(self, data):
        super().begin()
        self.external_body[:] = data.xfrc_applied - self._last_body
        self.body_wrenches.fill(0)

    def add_wrench(self, data, body_id, force, torque, point):
        if not self._open:
            raise RuntimeError("Call begin(data) before physical wrench collection")
        vectors = [np.asarray(v, dtype=float) for v in (force, torque, point)]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError("Finite world wrench and application point required")
        if (
            isinstance(body_id, (bool, np.bool_))
            or not isinstance(body_id, (int, np.integer))
            or not 0 <= body_id < self.model.nbody
        ):
            raise ValueError("Unknown body id")
        f, t, p = vectors
        self.body_wrenches[body_id, :3] += f
        self.body_wrenches[body_id, 3:] += t + np.cross(p - data.xipos[body_id], f)

    def commit(self, data):
        body = self.external_body + self.body_wrenches
        if not np.isfinite(body).all():
            raise ValueError("Nonfinite accumulated physical wrench")
        super().commit(data)
        data.xfrc_applied[:] = body
        self._last_body[:] = self.body_wrenches
