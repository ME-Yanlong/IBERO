"""双臂张力任务：持续夹持、张力保持、扰动后恢复；不访问物理引擎。"""

from ibero.core.task_api import TaskSpec, TaskResult


class CableStretchTask(TaskSpec):
    id = "cable_stretch"
    instruction = (
        "Hold both terminals, tension the cable and recover after a disturbance."
    )

    def reset(self):
        self.hold = 0.0
        self.ever_held = False
        self.lost = 0

    def __init__(self):
        self.reset()

    def evaluate(self, state, audit):
        cfg = state.extra
        both = all(state.grasps.values())
        self.ever_held |= both
        self.lost = (
            0
            if cfg.get("left_contact", False) and cfg.get("right_contact", False)
            else self.lost + 1
        )
        error = abs(state.cable_tension_n - cfg["target_tension_n"])
        disturbed = cfg["time_s"] >= cfg["disturbance_end_s"]
        stable = both and error <= cfg["tension_tolerance_n"] and not cfg["bend_alarm"]
        self.hold = self.hold + cfg["dt"] if stable and disturbed else 0.0
        metrics = {
            "left_grasp": state.grasps["left_gripper"],
            "right_grasp": state.grasps["right_gripper"],
            "tension_error_n": error,
            "hold_seconds": self.hold,
            "disturbance_finished": disturbed,
        }
        failure = (
            "cable_damage"
            if audit.cable_damage
            else "self_collision"
            if audit.self_collision
            else "dropped"
            if self.ever_held and self.lost >= 4
            else None
        )
        if failure:
            return TaskResult(
                -2, "failed", terminated=True, failure_reason=failure, metrics=metrics
            )
        if state.elapsed_steps >= state.max_episode_steps:
            return TaskResult(
                -1, "timeout", truncated=True, failure_reason="timeout", metrics=metrics
            )
        success = self.hold >= cfg["hold_seconds"]
        stage = (
            "success"
            if success
            else "holding"
            if stable
            else "recovering"
            if disturbed
            else "tensioning"
        )
        return TaskResult(
            5 if success else -error,
            stage,
            terminated=success,
            success=success,
            metrics=metrics,
        )


TASK_SPEC = CableStretchTask()
