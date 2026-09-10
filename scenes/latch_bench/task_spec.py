"""台架只采集机械状态，不把任务阶段写回物理世界。"""

from ibero.core.task_api import TaskResult, TaskSpec


class LatchBenchTask(TaskSpec):
    id = "latch_bench"
    instruction = "Measure elastic latch release in a controlled fixture."

    def reset(self):
        pass

    def evaluate(self, state, audit):
        invalid = state.extra.get("invalid_reason")
        success = bool(state.extra.get("released", False)) and not invalid
        return TaskResult(
            reward=float(success),
            stage=state.extra.get("latch_state", "locked"),
            terminated=bool(invalid or success),
            success=success,
            failure_reason=invalid,
        )


TASK_SPEC = LatchBenchTask()
