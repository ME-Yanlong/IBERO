"""几何台架不是已经接入加工力的铣削任务，不以删除体积判加工成功。"""

from ibero.core.task_api import TaskResult, TaskSpec


class StockBenchTask(TaskSpec):
    id = "stock_bench"
    instruction = "Validate stock geometry and collision independently."

    def reset(self):
        pass

    def evaluate(self, state, audit):
        return TaskResult(reward=0, stage="geometry_fixture", success=False)


TASK_SPEC = StockBenchTask()
