"""只读评估实际形状与停稳/安全；控制路径完成不能替代实体检查。"""

from ibero.core.task_api import TaskResult, TaskSpec
from ibero.processes.shape_check import MachiningTarget


class PlateMillingTask(TaskSpec):
    id = "plate_milling"
    instruction = "Machine the declared shape, retract safely and stop the spindle."

    def reset(self):
        pass

    def make_target(self, shape):
        # P4 负责目标尺寸；这张表只供控制/独立验收，材料去除模块不读取它。
        targets = {
            "slot": MachiningTarget("slot", (0, 0), (0.004, 0), 0.004, 0.002),
            "pocket": MachiningTarget("pocket", (0, 0), (0.004, 0.002), 0.004, 0.002),
            "through_hole": MachiningTarget(
                "through_hole", (0, 0), (0, 0), 0.004, 0.004
            ),
        }
        if shape not in targets:
            raise ValueError("Unknown P4 machining target")
        return targets[shape]

    def evaluate(self, state, audit):
        # 环境须提供独立占据检查和真实工具退刀状态；默认缺项一律不能成功。
        metrics = state.extra
        success = bool(
            metrics.get("shape_passed")
            and metrics.get("tool_retracted")
            and metrics.get("spindle_stopped")
            and not metrics.get("invalid_reason")
        )
        return TaskResult(
            reward=float(success),
            stage="complete" if success else "machining",
            success=success,
        )


TASK_SPEC = PlateMillingTask()
