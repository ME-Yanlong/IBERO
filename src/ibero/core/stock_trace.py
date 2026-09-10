"""加工审阅轨迹：物理状态＋初始材料身份＋事件前缀，不是可恢复训练状态。"""

import json
from pathlib import Path
from zipfile import ZipFile
import mujoco
import numpy as np
from ibero.materials.stock import RemovalEvent
from ibero.review import Trace


class StockTrace:
    format_version = "ibero.stock-trace/v2"

    def __init__(self, env):
        self.env = env
        self.states = []
        self.infos = []
        self.event_counts = []
        self.material_hashes = []
        self.events = []
        self.collision_modes = []
        self.identity = {
            key: env.manifest().get(key)
            for key in ("scene_hash", "source_hash", "asset_hash", "mujoco_version")
        }
        self.stock_hash = env.stock.stock_hash

    def _model_mode(self):
        """材料位掩码来自账本；另外记录加工刃区的工作/停转接触模式。"""
        if not hasattr(self.env, "process"):
            return {}
        return {
            "milling_blade_contype": int(
                self.env.model.geom_contype[self.env.process.blade_geom]
            )
        }

    def _validate_mode(self, mode):
        expected = {"milling_blade_contype"} if hasattr(self.env, "process") else set()
        if not isinstance(mode, dict) or set(mode) != expected:
            raise ValueError("Unsupported trace collision mode")
        if expected and (
            type(mode["milling_blade_contype"]) is not int
            or mode["milling_blade_contype"] not in (1, 2)
        ):
            raise ValueError("Invalid milling blade collision mode")

    def append(self, info):
        if self.env._replay_restored:
            raise RuntimeError("Replay cannot append new physics; reset first")
        self.env.binding.ensure_consistent()
        state = np.empty(mujoco.mj_stateSize(self.env.model, Trace.spec))
        mujoco.mj_getState(self.env.model, self.env.data, state, Trace.spec)
        if not np.isfinite(state).all():
            raise ValueError("Cannot record a nonfinite physical state")
        # 预检完成后一起追加，错误 info / 模式不会留下长度不同的半帧。
        snapshot = json.loads(json.dumps(info, allow_nan=False))
        if not isinstance(snapshot, dict):
            raise ValueError("Trace frame info must be a mapping")
        mode = self._model_mode()
        self._validate_mode(mode)
        self.states.append(state)
        self.infos.append(snapshot)
        self.events = list(self.env.stock.events)
        self.event_counts.append(len(self.events))
        self.material_hashes.append(self.env.stock.state_hash())
        self.collision_modes.append(mode)

    def restore(self, index):
        mode = self.collision_modes[index]
        self._validate_mode(mode)
        self.env.binding.restore_events(
            self.events[: self.event_counts[index]], self.env.data
        )
        mujoco.mj_setState(
            self.env.model, self.env.data, self.states[index], Trace.spec
        )
        if mode:
            self.env.model.geom_contype[self.env.process.blade_geom] = mode[
                "milling_blade_contype"
            ]
            self.env.model.geom_conaffinity[self.env.process.blade_geom] = 1
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
            "collision_modes": self.collision_modes,
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
            if (
                len(archive.infolist()) != 2
                or {i.filename for i in archive.infolist()}
                != {
                    "states.npy",
                    "metadata.npy",
                }
                or sum(i.file_size for i in archive.infolist()) > 512 * 1024**2
            ):
                raise ValueError("Unsupported or oversized stock trace archive")
        with np.load(path, allow_pickle=False) as stored:

            def reject_nonfinite(value):
                raise ValueError("Nonfinite trace metadata")

            meta = json.loads(str(stored["metadata"]), parse_constant=reject_nonfinite)
            states = stored["states"]
        if not isinstance(meta, dict) or set(meta) != {
            "format",
            "identity",
            "stock_hash",
            "seed",
            "infos",
            "event_counts",
            "material_hashes",
            "events",
            "collision_modes",
        }:
            raise ValueError("Malformed stock trace metadata")
        if (
            meta.get("format") != self.format_version
            or meta.get("identity") != self.identity
        ):
            raise ValueError("Stock trace format/scene/source/asset/engine mismatch")
        if type(meta["seed"]) is not int or not 0 <= meta["seed"] < 2**32:
            raise ValueError("Invalid trace seed")
        if meta.get("stock_hash") != self.env.stock.stock_hash:
            raise ValueError("Initial stock mismatch")
        if (
            states.ndim != 2
            or not 1 <= len(states) <= 200000
            or states.shape[1] != mujoco.mj_stateSize(self.env.model, Trace.spec)
            or not np.isfinite(states).all()
        ):
            raise ValueError("Invalid stock trace physical states")
        infos, counts, hashes, modes = (
            meta[k]
            for k in ("infos", "event_counts", "material_hashes", "collision_modes")
        )
        if not all(
            isinstance(v, list) and len(v) == len(states)
            for v in (infos, counts, hashes, modes)
        ):
            raise ValueError("Stock trace frame arrays disagree")
        if any(not isinstance(info, dict) for info in infos):
            raise ValueError("Trace frame info must be a mapping")
        for mode in modes:
            self._validate_mode(mode)
        if not isinstance(meta["events"], list) or len(meta["events"]) > 100000:
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
        # 全部验证通过后才 reset；拒绝损坏文件不会先抹掉当前回合。
        self.env.reset(seed=meta["seed"])
        self.states, self.infos = list(states), infos
        self.event_counts, self.material_hashes, self.events = counts, hashes, events
        self.collision_modes = modes
        return self
