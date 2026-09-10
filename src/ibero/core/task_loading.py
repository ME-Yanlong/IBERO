"""具名场景任务加载；配置检查不是 Python 安全沙箱，只加载受信任的本项目菜谱。"""

import hashlib
import importlib.util
from ibero.core.scene_loader import ValidatedScene
from ibero.core.task_api import TaskSpec


def load_task_spec(scene: ValidatedScene) -> TaskSpec:
    module_name = (
        f"ibero_scene_{hashlib.sha256(str(scene.task_path).encode()).hexdigest()[:12]}"
    )
    module_spec = importlib.util.spec_from_file_location(module_name, scene.task_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not import task specification {scene.task_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    task = getattr(module, "TASK_SPEC", None)
    if not isinstance(task, TaskSpec):
        raise TypeError("task_spec.py must expose TASK_SPEC derived from TaskSpec")
    return task
