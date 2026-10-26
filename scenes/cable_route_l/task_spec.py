"""沿立式布线板走圆角 L 路径；目标参数唯一来源为场景 YAML。"""

from pathlib import Path
import yaml

from ibero.tasks.cable_routing import CableRoutingTask, RoutingGoal


_CONFIG = yaml.safe_load(Path(__file__).with_name("scene_config.yaml").read_text(encoding="utf-8"))
TASK_SPEC = CableRoutingTask(
    RoutingGoal.from_scene_config(_CONFIG),
    task_id="cable_route_l",
    instruction="双臂沿立式板将裸线束走成圆角 L，依次压入四只线夹并松手保持。",
)
