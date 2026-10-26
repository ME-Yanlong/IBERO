"""立式 L/矩形菜谱需要真实编译成方向不同的线夹。"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from ibero.core.routing_config import validate_routing
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
        if name == "cable_route_l":
            tip_id = env.handles.guide_tip_site_id
            assert tip_id is not None
            assert env.handles.guide_nose_geom_id is not None
            assert env.handles.guide_rear_lip_geom_id is not None
            assert env.handles.guide_lead_tip_site_id is not None
            assert len(env.handles.guide_lead_geom_ids) == 2
            tip = env.data.site_xpos[tip_id]
            pinch = env.data.site_xpos[env.handles.right_anchor_site_id]
            assert np.linalg.norm(tip - pinch) > 0.04
            rear_center = env.data.geom_xpos[env.handles.guide_rear_lip_geom_id]
            assert np.linalg.norm(tip - rear_center) < 0.001
            lead_tip = env.data.site_xpos[env.handles.guide_lead_tip_site_id]
            assert 0.022 < np.linalg.norm(lead_tip - tip) < 0.023
            for geom_name in (
                "right_cable_guide_bridge", "right_cable_guide_nose",
                "right_cable_guide_rear_lip",
                "right_cable_guide_lead_bridge", "right_cable_guide_lead_nose",
            ):
                assert env.model.geom_contype[env.model.geom(geom_name).id] != 0
        else:
            assert env.handles.guide_tip_site_id is None
            assert env.handles.guide_nose_geom_id is None
            assert env.handles.guide_rear_lip_geom_id is None
            assert env.handles.guide_lead_tip_site_id is None
            assert env.handles.guide_lead_geom_ids == ()
        for tangent, observer in zip(tangents, env.clip_observers):
            mount = env.model.body(observer.names.mount_body).id
            # mj_forward 在 reset 中更新实体局部轴；竖夹的线向世界竖直。
            axis = env.data.xmat[mount].reshape(3, 3)[:, 1]
            expected = np.array([0, 1, 0] if tangent == "horizontal" else [0, 0, 1])
            assert abs(np.dot(axis, expected)) > 0.999
        assert env.last_info["stage"] == "routing"
        assert not env.last_info["is_success"]
        contact = env.last_info["guide_contact"]
        assert "rear_lip_contact_substeps" in contact
        assert "lead_contact_substeps" in contact
        assert "clip_collision_force_n" in contact
        assert "clip_collision_penetration_m" in contact
        if name == "cable_route_rectangle":
            assert not contact["lead_touching_cable"]
    finally:
        env.close()


def test_parallel_clip_bases_cannot_overlap_before_compilation():
    scene = ROOT / "cable_route_straight"
    config = yaml.safe_load((scene / "scene_config.yaml").read_text(encoding="utf-8"))
    constraints = yaml.safe_load((scene / "constraints.yaml").read_text(encoding="utf-8"))
    config["workcell"]["clips"][1]["origin_m"][1] = 0.025
    with pytest.raises(ValueError, match="Clip bases overlap"):
        validate_routing(config, constraints)
