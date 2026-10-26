"""开口矩形布线：六只实体线夹；目标参数唯一来源为场景 YAML。"""

from pathlib import Path
import yaml

from ibero.tasks.cable_routing import CableRoutingTask, RoutingGoal


_CONFIG = yaml.safe_load(Path(__file__).with_name("scene_config.yaml").read_text(encoding="utf-8"))
TASK_SPEC = CableRoutingTask(
    RoutingGoal.from_scene_config(_CONFIG),
    task_id="cable_route_rectangle",
    instruction="双臂在立式板上完成开口矩形布线，六只线夹全部保持且双手退离。",
)
