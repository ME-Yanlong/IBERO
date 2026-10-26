"""轨迹存档不得把运行中修改源码的回合登记为同一个模型。"""

from types import SimpleNamespace

import pytest

from ibero import review


def test_trace_rejects_source_change_before_writing(tmp_path, monkeypatch):
    current = ["first"]
    monkeypatch.setattr(review, "simulation_source_hash", lambda: current[0])
    monkeypatch.setattr(review, "vendored_robot_asset_hash", lambda: "asset")
    env = SimpleNamespace(scene=SimpleNamespace(scene_hash="scene"), np_random_seed=0)
    trace = review.Trace(env)
    current[0] = "second"

    path = tmp_path / "mixed.npz"
    with pytest.raises(ValueError, match="changed during recording"):
        trace.save(path)
    assert not path.exists()
