"""场景菜谱的自由端目标须真正进入独立任务判据。"""

from pathlib import Path

import pytest
import yaml

from ibero.core.routing_config import validate_routing
from ibero.tasks.cable_routing import RoutingGoal


SCENE = Path(__file__).resolve().parents[1] / "scenes" / "cable_clip_insert"


def test_free_end_target_is_loaded_from_recipe():
    config = yaml.safe_load((SCENE / "scene_config.yaml").read_text(encoding="utf-8"))
    constraints = yaml.safe_load((SCENE / "constraints.yaml").read_text(encoding="utf-8"))
    config["routing_goal"].update({
        "free_end_center_m": [0.46, -0.12, 0.86],
        "free_end_radius_m": 0.04,
        "free_end_index": -1,
    })
    validate_routing(config, constraints)
    goal = RoutingGoal.from_scene_config(config)
    assert goal.free_end_center_m == (0.46, -0.12, 0.86)
    assert goal.free_end_radius_m == 0.04
    assert goal.free_end_index == -1

    del config["routing_goal"]["free_end_radius_m"]
    with pytest.raises(ValueError, match="Free-end center and radius"):
        validate_routing(config, constraints)
