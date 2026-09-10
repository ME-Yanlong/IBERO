"""Semantic state machine for the Core-0.1 contact-driven cable handover."""

from __future__ import annotations

import numpy as np

from ibero.core.audit import AuditSnapshot
from ibero.core.task_api import TaskResult, TaskSpec, TaskState


class CableHandoverTask(TaskSpec):
    id = "cable_handover"
    instruction = (
        "Transfer the cable terminal from the left gripper to the right target region."
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._seen_left_holding = False
        self._seen_dual_grasp = False
        self._seen_left_release = False
        self._stage = "left_holding"
        self._settled = 0
        self._lost = 0

    def evaluate(self, state: TaskState, audit: AuditSnapshot) -> TaskResult:
        if audit.cable_damage:
            return TaskResult(
                -2.0, self._stage, terminated=True, failure_reason="cable_damage"
            )
        if audit.self_collision:
            return TaskResult(
                -2.0, self._stage, terminated=True, failure_reason="self_collision"
            )
        if state.elapsed_steps >= state.max_episode_steps:
            return TaskResult(
                -1.0, self._stage, truncated=True, failure_reason="timeout"
            )

        left = bool(state.grasps["left_gripper"])
        right = bool(state.grasps["right_gripper"])
        self._lost = (
            self._lost + 1
            if self._seen_left_holding
            and not state.extra.get("left_contact", left)
            and not state.extra.get("right_contact", right)
            else 0
        )
        if self._lost >= 4 and not self._seen_left_release:
            return TaskResult(
                -2, self._stage, terminated=True, failure_reason="dropped"
            )
        if left:
            self._seen_left_holding = True
        if self._seen_left_holding and left and right:
            self._seen_dual_grasp = True
            self._stage = "dual_grasp"
        elif self._seen_left_holding and not self._seen_dual_grasp:
            self._stage = "right_approach"
        if self._seen_dual_grasp and right and not left:
            self._seen_left_release = True
            self._stage = "left_release"

        terminal = state.body_positions["cable_terminal"]
        target = state.site_positions["target_region"]
        distance = float(np.linalg.norm(terminal - target))
        in_target = distance <= float(state.site_positions["target_radius"][0])
        stable = (
            self._seen_left_release
            and not left
            and not right
            and in_target
            and state.gripper_openings["right_gripper"] > 0.055
            and state.extra.get("supported", False)
            and state.extra.get("terminal_speed_m_s", 1) < 0.04
        )
        self._settled = self._settled + 1 if stable else 0
        if self._settled * state.extra.get("dt", 0.05) >= 0.5:
            self._stage = "success"
            return TaskResult(
                5.0,
                self._stage,
                terminated=True,
                success=True,
                metrics={
                    "terminal_target_distance_m": distance,
                    "left_grasp": left,
                    "right_grasp": right,
                    "supported": True,
                },
            )

        reward = 0.05
        if self._seen_left_holding:
            reward += 0.20
        if self._seen_dual_grasp:
            reward += 0.50
        if self._seen_left_release:
            reward += 0.75
        reward += max(0.0, 0.25 * (1.0 - distance / 0.30))
        return TaskResult(
            reward,
            self._stage,
            metrics={
                "terminal_target_distance_m": distance,
                "left_grasp": left,
                "right_grasp": right,
            },
        )


TASK_SPEC = CableHandoverTask()
