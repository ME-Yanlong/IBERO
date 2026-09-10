"""有界预分配体素碰撞：每格实体 box，材料提交同步碰撞与显示。"""

import threading
import mujoco
import numpy as np
from ibero.materials.stock import VoxelStock


def add_stock_geoms(spec, stock, *, active_only=False):
    """active_only 只用于受控重编译比较；正式路径保留稳定的全部 geom ID。

    固定工件无运动自由度；代理不再重复携带材料质量。真实剩余质量由 stock
    账本拥有。本阶段不支持把加工后的工件解锁为自由刚体。
    """
    ids = (
        np.flatnonzero(stock.occupied) if active_only else np.arange(len(stock.centers))
    )
    if len(ids) > 60000 or len(ids) + len(spec.geoms) > 65000:
        raise ValueError(
            "Preallocated stock collision capacity exceeded (60000 cells / 65000 total geoms)"
        )
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, stock.rotation.ravel())
    body = spec.worldbody.add_body(name="machining_stock", pos=stock.origin, quat=quat)
    names = []
    for i in ids:
        name = f"stock_cell_{i}"
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=stock.centers[i],
            size=stock.half_sizes[i],
            mass=0,
            contype=1,
            conaffinity=1,
            solref=[0.001, 1],
            solimp=[0.99, 0.999, 0.00001, 0.5, 2],
            friction=[0.3, 0.005, 0.0001],
            rgba=[0.55, 0.62, 0.68, 1],
        )
        names.append(name)
    return names


class StockCollisionBinding:
    """几何提交在物理线程的子步边界完成；显示/回放必须走同一提交入口。

    不移动 box，不改变 BVH 包围几何。仅对已真正去除的格关闭双方碰撞位；
    邻近剩余格保持实体。异常在材料发布前回滚位掩码/外观并重新 forward。
    """

    def __init__(self, model, stock):
        self.model, self.stock = model, stock
        self.geom_ids = np.array(
            [model.geom(f"stock_cell_{i}").id for i in range(len(stock.centers))]
        )
        self._lock = threading.RLock()
        self._initial_type = model.geom_contype[self.geom_ids].copy()
        self._initial_affinity = model.geom_conaffinity[self.geom_ids].copy()
        self._initial_rgba = model.geom_rgba[self.geom_ids].copy()
        self.version = stock.version
        self._write_mask(stock.occupied)

    def _write_mask(self, occupied):
        self.model.geom_contype[self.geom_ids] = self._initial_type * occupied
        self.model.geom_conaffinity[self.geom_ids] = self._initial_affinity * occupied
        self.model.geom_rgba[self.geom_ids] = self._initial_rgba
        self.model.geom_rgba[self.geom_ids[~occupied], 3] = 0

    def ensure_consistent(self):
        if self.version != self.stock.version:
            raise RuntimeError("Material changed outside the collision commit boundary")
        mask = self.stock.occupied
        if not (
            np.array_equal(
                self.model.geom_contype[self.geom_ids], self._initial_type * mask
            )
            and np.array_equal(
                self.model.geom_conaffinity[self.geom_ids],
                self._initial_affinity * mask,
            )
        ):
            raise RuntimeError("Stock collision mask diverged from material state")
        if not np.array_equal(
            self.model.geom_rgba[self.geom_ids, 3], self._initial_rgba[:, 3] * mask
        ):
            raise RuntimeError("Stock appearance diverged from material state")

    def commit(self, event, data, *, fault_at=None):
        """fault_at 为异常原子性测试钩子，正常环境不提供该选项。"""
        if fault_at not in {None, "after_collision", "after_forward"}:
            raise ValueError("Unknown commit fault injection")
        with self._lock:
            self.ensure_consistent()
            # 先纯校验，不逐次重放整个历史；每个小切削步的成本不随历史线性增长。
            ids, removed, duplicate = self.stock.validate_event(event)
            if duplicate:
                return 0.0
            previous = self.stock.occupied
            following = previous.copy()
            following[ids] = False
            try:
                self._write_mask(following)
                if fault_at == "after_collision":
                    raise RuntimeError("Injected collision commit failure")
                mujoco.mj_forward(self.model, data)
                if fault_at == "after_forward":
                    raise RuntimeError("Injected post-forward commit failure")
                self.stock.commit(event)
                self.version = self.stock.version
            except Exception:
                self._write_mask(previous)
                mujoco.mj_forward(self.model, data)
                raise
            return removed

    def _clone_with_events(self, events):
        trial = VoxelStock(
            self.stock.params,
            self.stock.cell_size_m,
            max_cells=len(self.stock.centers),
            origin=self.stock.origin,
            rotation=self.stock.rotation,
        )
        for event in events:
            trial.commit(event)
        return trial

    def restore_events(self, events, data):
        """先验证完整事件前缀再同步到同一对象；只读回放的版本允许倒退。"""
        with self._lock:
            trial = self._clone_with_events(events)
            previous = self.stock.occupied
            try:
                self._write_mask(trial.occupied)
                mujoco.mj_forward(self.model, data)
            except Exception:
                self._write_mask(previous)
                mujoco.mj_forward(self.model, data)
                raise
            # trial 已通过逐事件校验；保留 stock 身份，让所有消费者仍读同一实例。
            self.stock._occupied = trial._occupied
            self.stock.events = trial.events
            self.stock._events_by_id = trial._events_by_id
            self.stock.version = trial.version
            self.version = trial.version

    def reset(self, data):
        with self._lock:
            self.restore_events([], data)
            mujoco.mj_resetData(self.model, data)
            mujoco.mj_forward(self.model, data)
