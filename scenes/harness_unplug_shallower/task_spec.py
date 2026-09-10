"""退出历史 + 真实支承 + 松爪 + 稳定，避免在出生位置或悬空误报成功。"""

from ibero.core.task_api import TaskSpec, TaskResult


class HarnessUnplugTask(TaskSpec):
    id = "harness_unplug"
    instruction = "Press and withdraw the wired plug, then deposit it on the receiver."

    def reset(self):
        self.withdrawn = False
        self.held_seconds = 0.0

    def evaluate(self, state, audit):
        x = state.extra
        failure = x.get("invalid_reason")
        if failure:
            return TaskResult(0, "invalid", terminated=True, failure_reason=failure)
        self.withdrawn |= (
            x["released"]
            and x["withdrawal_m"] >= x["required_withdrawal_m"]
            and state.grasps["right"]
            and x["pose_error_rad"] <= x["pose_tolerance_rad"]
        )
        stable = (
            self.withdrawn
            and x["released"]
            and not state.grasps["right"]
            and x["receiver_supported"]
            and x["receiver_contains_plug"]
            and x["plug_speed_m_s"] < x["placement_speed_limit_m_s"]
            and x["plug_angular_speed_rad_s"] < 0.1
        )
        self.held_seconds = self.held_seconds + x["control_dt"] if stable else 0.0
        success = self.held_seconds >= x["required_hold_seconds"]
        timeout = state.elapsed_steps >= state.max_episode_steps and not success
        return TaskResult(
            float(success),
            "placed" if success else "placing" if self.withdrawn else x["latch_state"],
            terminated=success,
            truncated=timeout,
            success=success,
            failure_reason="timeout" if timeout else None,
            metrics={
                "withdrawn_while_grasped": self.withdrawn,
                "stable_seconds": self.held_seconds,
            },
        )


TASK_SPEC = HarnessUnplugTask()
