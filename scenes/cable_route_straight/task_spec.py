"""在立式板的三只实体线夹中固定同一根线束。"""

from pathlib import Path

import yaml

from ibero.tasks.cable_routing import CableRoutingTask, RoutingGoal


SCENE_CONFIG = yaml.safe_load(
    Path(__file__).with_name("scene_config.yaml").read_text(encoding="utf-8")
)


TASK_SPEC = CableRoutingTask(
    RoutingGoal.from_scene_config(SCENE_CONFIG),
    task_id="cable_route_straight",
    instruction="双臂取线并沿立式板的三只线夹走直线，松手后全部保持。",
)
