"""只读评估实际形状与停稳/安全；控制路径完成不能替代实体检查。"""

from ibero.core.task_api import TaskResult, TaskSpec


class PlateMillingTask(TaskSpec):
    id = "plate_milling"
    instruction = "Machine the declared shape, retract safely and stop the spindle."

    def reset(self):
        pass

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
