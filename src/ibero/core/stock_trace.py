"""加工审阅轨迹：物理状态＋初始材料身份＋事件前缀，不是可恢复训练状态。"""

import json
from pathlib import Path
from zipfile import ZipFile
import mujoco
import numpy as np
from ibero.materials.stock import RemovalEvent
from ibero.review import Trace


class StockTrace:
    format_version = "ibero.stock-trace/v1"

    def __init__(self, env):
        self.env = env
        self.states = []
        self.infos = []
        self.event_counts = []
        self.material_hashes = []
        self.events = []
        self.identity = {
            key: env.manifest().get(key)
            for key in ("scene_hash", "source_hash", "asset_hash", "mujoco_version")
        }
        self.stock_hash = env.stock.stock_hash

    def append(self, info):
        if self.env._replay_restored:
            raise RuntimeError("Replay cannot append new physics; reset first")
        self.env.binding.ensure_consistent()
        state = np.empty(mujoco.mj_stateSize(self.env.model, Trace.spec))
        mujoco.mj_getState(self.env.model, self.env.data, state, Trace.spec)
        self.states.append(state)
        self.infos.append(json.loads(json.dumps(info, allow_nan=False)))
        self.events = list(self.env.stock.events)
        self.event_counts.append(len(self.events))
        self.material_hashes.append(self.env.stock.state_hash())

    def restore(self, index):
        self.env.binding.restore_events(
            self.events[: self.event_counts[index]], self.env.data
        )
        mujoco.mj_setState(
            self.env.model, self.env.data, self.states[index], Trace.spec
        )
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env._replay_restored = True
        return dict(
            self.infos[index], replay=True, material_version=self.env.stock.version
        )

    def save(self, path):
        if not self.states:
            raise ValueError("Cannot save empty stock trace")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": self.format_version,
            "identity": self.identity,
            "stock_hash": self.stock_hash,
            "seed": self.env.seed_value,
            "infos": self.infos,
            "event_counts": self.event_counts,
            "material_hashes": self.material_hashes,
            "events": [e.as_dict() for e in self.events],
        }
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                states=np.stack(self.states),
                metadata=json.dumps(metadata, allow_nan=False),
            )

    def load(self, path):
        # 先检查未压缩体积，再让 NumPy 分配；不加载 pickle 或执行归档中的代码。
        with ZipFile(path) as archive:
            if {i.filename for i in archive.infolist()} != {
                "states.npy",
                "metadata.npy",
            } or sum(i.file_size for i in archive.infolist()) > 512 * 1024**2:
                raise ValueError("Unsupported or oversized stock trace archive")
        with np.load(path, allow_pickle=False) as stored:
            meta = json.loads(str(stored["metadata"]))
            states = stored["states"]
        if (
            meta.get("format") != self.format_version
            or meta.get("identity") != self.identity
        ):
            raise ValueError("Stock trace format/scene/source/asset/engine mismatch")
        self.env.reset(seed=meta["seed"])
        if meta.get("stock_hash") != self.env.stock.stock_hash:
            raise ValueError("Initial stock mismatch")
        if (
            states.ndim != 2
            or not 1 <= len(states) <= 200000
            or states.shape[1] != mujoco.mj_stateSize(self.env.model, Trace.spec)
            or not np.isfinite(states).all()
        ):
            raise ValueError("Invalid stock trace physical states")
        infos, counts, hashes = (
            meta[k] for k in ("infos", "event_counts", "material_hashes")
        )
        if not all(
            isinstance(v, list) and len(v) == len(states)
            for v in (infos, counts, hashes)
        ):
            raise ValueError("Stock trace frame arrays disagree")
        if len(meta["events"]) > 100000:
            raise ValueError("Stock trace event capacity exceeded")
        events = []
        for raw in meta["events"]:
            if not isinstance(raw, dict) or set(raw) != set(
                RemovalEvent.__dataclass_fields__
            ):
                raise ValueError("Malformed removal event")
            events.append(
                RemovalEvent(**dict(raw, removed_ids=tuple(raw["removed_ids"])))
            )
        if any(type(n) is not int or not 0 <= n <= len(events) for n in counts) or any(
            a > b for a, b in zip(counts, counts[1:])
        ):
            raise ValueError("Invalid/nonmonotonic event prefixes")
        if counts[-1] != len(events):
            raise ValueError("Unused trailing events in stock trace")
        trial = self.env.binding._clone_with_events([])
        prefixes = {0: trial.state_hash()}
        for index, event in enumerate(events, start=1):
            trial.commit(event)
            prefixes[index] = trial.state_hash()
        if any(prefixes[n] != h for n, h in zip(counts, hashes)):
            raise ValueError("Stock trace material snapshot hash mismatch")
        self.states, self.infos = list(states), infos
        self.event_counts, self.material_hashes, self.events = counts, hashes, events
        return self
