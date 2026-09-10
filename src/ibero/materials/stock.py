"""固定毛坯的有界体素存储；只描述几何去除，不冒充切削过程。"""

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np
from ibero.materials.parameters import StockParameters, finite_number


@dataclass(frozen=True)
class RemovalEvent:
    event_id: str
    stock_hash: str
    before_version: int
    after_version: int
    removed_ids: tuple[int, ...]
    volume_m3: float
    mass_kg: float
    request_hash: str

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not 1 <= len(self.event_id) <= 128:
            raise ValueError("Invalid event identity")
        for h in (self.stock_hash, self.request_hash):
            if (
                not isinstance(h, str)
                or len(h) != 64
                or any(c not in "0123456789abcdef" for c in h)
            ):
                raise ValueError("Invalid event hash")
        if any(
            type(v) is not int or v < 0
            for v in (self.before_version, self.after_version)
        ):
            raise ValueError("Invalid material version")
        if not isinstance(self.removed_ids, tuple) or any(
            type(i) is not int or i < 0 for i in self.removed_ids
        ):
            raise ValueError("Invalid removed cell ids")
        finite_number(self.volume_m3, "removed volume", positive=False)
        finite_number(self.mass_kg, "removed mass", positive=False)

    def as_dict(self):
        return asdict(self)


class VoxelStock:
    """局部盒形毛坯，边界不足一格时保留真实体积，不扩大用户尺寸。

    中心采样决定一格是否去除；因此表面精度受 cell_size 限制。占据与账本
    不受相机影响。固定毛坯不产生自由余料，不能宣称完整切断动力学。
    """

    def __init__(
        self,
        params: StockParameters,
        cell_size_m,
        *,
        max_cells=200000,
        origin=(0, 0, 0),
        rotation=None,
    ):
        self.params = params
        self.cell_size_m = finite_number(cell_size_m, "cell_size_m")
        if type(max_cells) is not int or not 1 <= max_cells <= 2000000:
            raise ValueError("max_cells must be an integer in [1, 2000000]")
        ratios = [s / self.cell_size_m for s in params.size_m]
        if any(not math.isfinite(r) or r > max_cells for r in ratios):
            raise ValueError("Stock grid exceeds capacity")
        counts = [math.ceil(r - 1e-12) for r in ratios]
        if min(counts) < 1 or math.prod(counts) > max_cells:
            raise ValueError("Stock grid exceeds capacity or has invalid dimensions")
        self.shape = tuple(counts)
        self.origin = np.array(origin, dtype=float)
        self.rotation = np.array(
            np.eye(3) if rotation is None else rotation, dtype=float
        )
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError("Stock origin must be a finite world point")
        if (
            self.rotation.shape != (3, 3)
            or not np.isfinite(self.rotation).all()
            or not np.allclose(
                self.rotation.T @ self.rotation, np.eye(3), atol=1e-10, rtol=0
            )
            or not math.isclose(np.linalg.det(self.rotation), 1, abs_tol=1e-10)
        ):
            raise ValueError("Stock rotation must be proper and orthonormal")
        edges = [
            np.r_[np.arange(n) * self.cell_size_m, s] - s / 2
            for s, n in zip(params.size_m, counts)
        ]
        axis_centers = [(e[:-1] + e[1:]) / 2 for e in edges]
        self.centers = np.stack(
            np.meshgrid(*axis_centers, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        self.half_sizes = np.stack(
            np.meshgrid(*[np.diff(e) / 2 for e in edges], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        self.volumes = np.prod(self.half_sizes * 2, axis=1)
        self.initial_volume_m3 = float(self.volumes.sum())
        if not np.isfinite(self.initial_volume_m3) or self.initial_volume_m3 <= 0:
            raise ValueError("Stock volume exceeds numerical range")
        self.stock_hash = hashlib.sha256(
            json.dumps(
                {
                    "params": asdict(params),
                    "cell_size_m": self.cell_size_m,
                    "origin": self.origin.tolist(),
                    "rotation": self.rotation.tolist(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for a in (
            self.origin,
            self.rotation,
            self.centers,
            self.half_sizes,
            self.volumes,
        ):
            a.setflags(write=False)
        self.reset()

    def reset(self):
        self._occupied = np.ones(len(self.centers), dtype=bool)
        self.version = 0
        self.events = []
        self._events_by_id = {}

    @property
    def occupied(self):
        result = self._occupied.copy()
        result.setflags(write=False)
        return result

    @property
    def volume_m3(self):
        return float(self.volumes[self._occupied].sum())

    @property
    def mass_kg(self):
        return self.volume_m3 * self.params.density_kg_m3

    def local_to_world(self, points):
        return np.asarray(points) @ self.rotation.T + self.origin

    def world_to_local(self, points):
        return (np.asarray(points) - self.origin) @ self.rotation

    def point_indices(self, world_points):
        raw = np.asarray(world_points, dtype=float)
        if raw.ndim < 1 or raw.shape[-1] != 3 or not np.isfinite(raw).all():
            raise ValueError("Finite points with final dimension 3 required")
        try:
            with np.errstate(over="raise", invalid="raise"):
                p = self.world_to_local(raw)
        except FloatingPointError as error:
            raise ValueError(
                "Point coordinate transform exceeds numerical range"
            ) from error
        size = np.asarray(self.params.size_m)
        valid = np.all((p >= -size / 2) & (p < size / 2), axis=-1)
        # 先排除域外点，再转整数，避免巨大但有限的外部坐标触发整型溢出。
        bounded = np.where(valid[..., None], p, 0)
        index = np.floor((bounded + size / 2) / self.cell_size_m).astype(int)
        index = np.clip(index, 0, np.asarray(self.shape) - 1)
        flat = np.ravel_multi_index(tuple(np.moveaxis(index, -1, 0)), self.shape)
        return np.where(valid, flat, -1)

    def occupied_at(self, world_points):
        ids = self.point_indices(world_points)
        return (ids >= 0) & self._occupied[np.maximum(ids, 0)]

    def prepare_removal(self, ids, event_id):
        """纯候选：不改变材料。相同 ID/请求幂等，不允许同 ID 换一批体素。"""
        if not isinstance(event_id, str) or not event_id or len(event_id) > 128:
            raise ValueError("Nonempty bounded removal event id required")
        raw = np.asarray(ids)
        if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
            raise ValueError("Removal ids must be a one-dimensional integer sequence")
        selected = np.unique(raw.astype(np.int64))
        if np.any(selected < 0) or np.any(selected >= len(self.centers)):
            raise ValueError("Removal index outside stock")
        request = hashlib.sha256(selected.tobytes()).hexdigest()
        if event_id in self._events_by_id:
            previous = self._events_by_id[event_id]
            if previous.request_hash != request:
                raise ValueError("Event id reused for a different geometry request")
            return previous
        removed = selected[self._occupied[selected]]
        volume = float(self.volumes[removed].sum())
        return RemovalEvent(
            event_id,
            self.stock_hash,
            self.version,
            self.version + bool(len(removed)),
            tuple(int(i) for i in removed),
            volume,
            volume * self.params.density_kg_m3,
            request,
        )

    def validate_event(self, event):
        """无副作用校验，供几何/碰撞原子提交预检复用。"""
        if not isinstance(event, RemovalEvent) or event.stock_hash != self.stock_hash:
            raise ValueError("Removal event belongs to a different stock")
        if event.event_id in self._events_by_id:
            if event != self._events_by_id[event.event_id]:
                raise ValueError("Conflicting duplicate event")
            return np.empty(0, dtype=np.int64), 0.0, True
        if len(self.events) >= 100000:
            raise ValueError("Removal event capacity exceeded")
        if any(i >= len(self.centers) for i in event.removed_ids):
            raise ValueError("Removal index outside stock")
        ids = np.asarray(event.removed_ids, dtype=np.int64)
        if (
            event.before_version != self.version
            or event.after_version != self.version + bool(len(ids))
            or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0)
            or np.any(ids >= len(self.centers))
            or not np.all(self._occupied[ids])
        ):
            raise ValueError("Stale or invalid removal event")
        volume = float(self.volumes[ids].sum())
        if not math.isclose(
            event.volume_m3, volume, rel_tol=1e-12, abs_tol=1e-18
        ) or not math.isclose(
            event.mass_kg,
            volume * self.params.density_kg_m3,
            rel_tol=1e-12,
            abs_tol=1e-18,
        ):
            raise ValueError("Removal volume/mass ledger mismatch")
        return ids, volume, False

    def commit(self, event):
        """显式几何台架或已获准的工艺提交入口；校验完成前不改任何状态。"""
        ids, volume, duplicate = self.validate_event(event)
        if duplicate:
            return 0.0
        self._occupied[ids] = False
        self.version = event.after_version
        self.events.append(event)
        self._events_by_id[event.event_id] = event
        return volume

    def state_hash(self):
        return hashlib.sha256(
            self.stock_hash.encode()
            + np.packbits(self._occupied).tobytes()
            + str(self.version).encode()
        ).hexdigest()
