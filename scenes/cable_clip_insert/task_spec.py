"""把裸线段压入板上的侧向弹性线夹，松手后仍应保持。"""

from pathlib import Path

import yaml

from ibero.tasks.cable_routing import CableRoutingTask, RoutingGoal


SCENE_CONFIG = yaml.safe_load(
    Path(__file__).with_name("scene_config.yaml").read_text(encoding="utf-8")
)


TASK_SPEC = CableRoutingTask(
    RoutingGoal.from_scene_config(SCENE_CONFIG),
    task_id="cable_clip_insert",
    instruction="用双手抓住裸线，压入立式板的线夹，松手后保持。",
)
