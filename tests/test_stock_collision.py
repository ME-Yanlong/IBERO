"""真实碰撞、提交回滚、材料回放的局部测试；大工件长验收另存证据。"""

import json
import mujoco
import numpy as np
import pytest
from ibero.materials.parameters import StockParameters
from ibero.materials.stock import VoxelStock
from ibero.materials.stock_collision import StockCollisionBinding
from ibero.benches.stock_probe import probe_spec
from ibero.core.stock_trace import StockTrace
from ibero.core.reproducibility import simulation_source_hash


class TinyProbe:
    def __init__(self):
        self.stock = VoxelStock(StockParameters(size_m=(0.012, 0.012, 0.004)), 0.002)
        self.model = probe_spec(self.stock).compile()
        self.data = mujoco.MjData(self.model)
        self.binding = StockCollisionBinding(self.model, self.stock)
        self.reset(seed=0)

    def reset(self, *, seed):
        self.binding.reset(self.data)
        self.seed_value = seed
        self._replay_restored = False

    def manifest(self):
        return {
            "scene_hash": self.stock.stock_hash,
            "source_hash": simulation_source_hash(),
            "mujoco_version": mujoco.__version__,
        }

    def step(self):
        if self._replay_restored:
            raise RuntimeError("reset required")
        self.binding.ensure_consistent()
        mujoco.mj_step(self.model, self.data)


@pytest.mark.parametrize("fault", ["after_collision", "after_forward"])
def test_atomic_failure_keeps_material_collision_appearance_and_qpos(fault):
    env = TinyProbe()
    q = env.data.qpos.copy()
    h = env.stock.state_hash()
    types, rgba = env.model.geom_contype.copy(), env.model.geom_rgba.copy()
    event = env.stock.prepare_removal([0, 1, 2], "a")
    with pytest.raises(RuntimeError, match="Injected"):
        env.binding.commit(event, env.data, fault_at=fault)
    assert env.stock.state_hash() == h and env.binding.version == 0
    np.testing.assert_array_equal(env.data.qpos, q)
    np.testing.assert_array_equal(env.model.geom_contype, types)
    np.testing.assert_array_equal(env.model.geom_rgba, rgba)
    env.binding.ensure_consistent()


def test_material_replay_roundtrip_reset_and_reject_corruption(tmp_path):
    env = TinyProbe()
    trace = StockTrace(env)
    trace.append({"stage": "initial"})
    event = env.stock.prepare_removal([0, 1, 2], "hole")
    env.binding.commit(event, env.data)
    env.step()
    trace.append({"stage": "cut"})
    path = tmp_path / "trace.npz"
    trace.save(path)
    loaded = StockTrace(env).load(path)
    loaded.restore(1)
    assert env.stock.version == 1 and not env.stock.occupied[0]
    assert env.model.geom_contype[env.binding.geom_ids[0]] == 0
    with pytest.raises(RuntimeError):
        env.step()
    loaded.restore(0)
    assert env.stock.version == 0 and env.stock.occupied.all()
    assert np.all(env.model.geom_contype[env.binding.geom_ids] == 1)
    env.reset(seed=0)
    env.step()
    with np.load(path, allow_pickle=False) as stored:
        states = stored["states"]
        meta = json.loads(str(stored["metadata"]))
    meta["events"][0]["mass_kg"] *= 2
    corrupt = tmp_path / "corrupt.npz"
    np.savez_compressed(corrupt, states=states, metadata=json.dumps(meta))
    with pytest.raises(ValueError, match="ledger"):
        StockTrace(env).load(corrupt)


def test_external_mask_edit_is_not_a_material_cut():
    env = TinyProbe()
    env.model.geom_contype[env.binding.geom_ids[0]] = 0
    with pytest.raises(RuntimeError, match="diverged"):
        env.binding.ensure_consistent()
    env.binding.reset(env.data)
    env.binding.ensure_consistent()
    env.model.geom_rgba[env.binding.geom_ids[0], 3] = 0
    with pytest.raises(RuntimeError, match="appearance"):
        env.binding.ensure_consistent()


def test_stock_recipe_compiles_without_fake_robot_or_cable():
    from pathlib import Path
    from dataclasses import replace
    import copy
    from ibero.core.scene_loader import SceneLoader
    from ibero.core.scene_compiler import SceneCompiler

    scene = SceneLoader().validate(
        Path(__file__).resolve().parents[1] / "scenes/stock_bench"
    )
    cfg = copy.deepcopy(scene.config)
    cfg["materials"]["stock"]["size_m"] = [0.006, 0.006, 0.002]
    result = SceneCompiler().compile(replace(scene, config=cfg))
    assert result.model.nv == 0 and result.model.ngeom == 72
    assert result.stock.volume_m3 == pytest.approx(0.006 * 0.006 * 0.002)
    assert not hasattr(result, "cable_parameters")
    result.binding.ensure_consistent()
