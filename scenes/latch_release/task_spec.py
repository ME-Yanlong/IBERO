"""无损完整退出并保持；只读物理事实，控制器阶段不是任务真值。"""

from ibero.core.task_api import TaskSpec, TaskResult


class LatchReleaseTask(TaskSpec):
    id = "latch_release"
    instruction = (
        "Press the elastic latch, withdraw the freely grasped plug, and hold it."
    )

    def reset(self):
        self.held_seconds = 0.0

    def evaluate(self, state, audit):
        x = state.extra
        failure = x.get("invalid_reason")
        if failure:
            return TaskResult(0, "invalid", terminated=True, failure_reason=failure)
        valid = (
            x["released"]
            and x["withdrawal_m"] >= x["required_withdrawal_m"]
            and state.grasps["right"]
            and x["pose_error_rad"] <= x["pose_tolerance_rad"]
        )
        self.held_seconds = self.held_seconds + x["control_dt"] if valid else 0.0
        success = self.held_seconds >= x["required_hold_seconds"]
        timeout = state.elapsed_steps >= state.max_episode_steps and not success
        return TaskResult(
            float(success),
            "held" if success else "withdrawn" if valid else x["latch_state"],
            terminated=success,
            truncated=timeout,
            success=success,
            failure_reason="timeout" if timeout else None,
            metrics={
                "hold_seconds": self.held_seconds,
                "withdrawal_m": x["withdrawal_m"],
            },
        )


TASK_SPEC = LatchReleaseTask()
