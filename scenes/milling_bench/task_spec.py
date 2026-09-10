"""材料台架不宣称机器人任务成功；形状与载荷由独立 G7 检查器判断。"""

from ibero.core.task_api import TaskResult, TaskSpec


class MillingBenchTask(TaskSpec):
    id = "milling_bench"
    instruction = "Identify limited three-axis cutting, not a robot benchmark."

    def reset(self):
        pass

    def evaluate(self, state, audit):
        return TaskResult(reward=0, stage="process_fixture", success=False)


TASK_SPEC = MillingBenchTask()
