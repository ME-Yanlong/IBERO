"""立式 L/矩形菜谱需要真实编译成方向不同的线夹。"""

from pathlib import Path

import numpy as np
import pytest

from ibero.core.scene_loader import SceneLoader
from ibero.envs.cable_routing import CableRoutingEnv


ROOT = Path(__file__).resolve().parents[1] / "scenes"


@pytest.mark.parametrize(
    ("name", "tangents"),
    [
        ("cable_route_l", ("horizontal", "horizontal", "vertical", "vertical")),
        (
            "cable_route_rectangle",
            ("horizontal", "horizontal", "vertical", "vertical", "horizontal", "vertical"),
        ),
    ],
)
def test_pattern_scenes_use_real_horizontal_and_vertical_clip_mounts(name, tangents):
    scene = SceneLoader().validate(ROOT / name)
    assert tuple(clip["tangent"] for clip in scene.config["workcell"]["clips"]) == tangents
    env = CableRoutingEnv(scene_path=ROOT / name)
    try:
        assert len(env.clip_observers) == len(tangents)
        assert tuple(env.task.goal.clip_ids) == tuple(
            clip["id"] for clip in scene.config["workcell"]["clips"]
        )
        env.reset(seed=0)
        for tangent, observer in zip(tangents, env.clip_observers):
            mount = env.model.body(observer.names.mount_body).id
            # mj_forward 在 reset 中更新实体局部轴；竖夹的线向世界竖直。
            axis = env.data.xmat[mount].reshape(3, 3)[:, 1]
            expected = np.array([0, 1, 0] if tangent == "horizontal" else [0, 0, 1])
            assert abs(np.dot(axis, expected)) > 0.999
        assert env.last_info["stage"] == "routing"
        assert not env.last_info["is_success"]
    finally:
        env.close()
