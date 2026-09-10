"""Deterministic, contact-only scripted baselines for IBERO regression tests."""

from __future__ import annotations

import numpy as np

from ibero.envs.cable_handover import CableHandoverEnv
from ibero.control.tension import orientation_error


class CableHandoverScript:
    """Approach, dual-grasp, release and place without altering object state."""

    def __init__(self) -> None:
        self.phase = "preapproach"
        self._phase_steps = 0
        self._rotations = None
        self._left_position = None

    @staticmethod
    def _move_action(
        current: np.ndarray, target: np.ndarray, scale: float = 0.2
    ) -> np.ndarray:
        return np.clip((target - current) / 0.012, -scale, scale)

    def action(self, env: CableHandoverEnv) -> np.ndarray:
        action = np.zeros(14, dtype=np.float32)
        action[6] = 1.0  # left stays closed until the release phase
        action[13] = -1.0
        right_marker = env.data.site_xpos[
            env.handles.cable.terminal_right_grasp_site_id
        ].copy()
        right_pinch = env.data.site_xpos[env.handles.right_pinch_site_id].copy()
        ids = [env.handles.left_pinch_site_id, env.handles.right_pinch_site_id]
        if self._rotations is None:
            self._rotations = env.data.site_xmat[ids].reshape(2, 3, 3).copy()
            self._left_position = env.data.site_xpos[ids[0]].copy()
            self._rotations[1] = self._rotations[0].copy()
        for i, site in enumerate(ids):
            action[7 * i + 3 : 7 * i + 6] = (
                0.3
                * orientation_error(
                    env.data.site_xmat[site].reshape(3, 3), self._rotations[i]
                )
                / 0.1
            )
        action[:3] = self._move_action(env.data.site_xpos[ids[0]], self._left_position)

        if self.phase == "preapproach":
            # 从端头后方沿工具轴接近，避免张开的指垫横向扫过端头。
            target = right_marker + np.array([-0.035, 0, 0])
            action[7:10] = self._move_action(right_pinch, target)
            if np.linalg.norm(target - right_pinch) < 0.008:
                self.phase = "approach"
        elif self.phase == "approach":
            action[7:10] = self._move_action(right_pinch, right_marker)
            if np.linalg.norm(right_marker - right_pinch) < 0.025:
                self.phase = "close"
                self._phase_steps = 0
        elif self.phase == "close":
            action[13] = 1.0
            self._phase_steps += 1
            if env._grasped("right") and self._phase_steps >= 10:
                self.phase = "release"
                self._phase_steps = 0
        elif self.phase == "release":
            action[6] = -1.0
            action[13] = 1.0
            self._phase_steps += 1
            if not env._grasped("left") and self._phase_steps >= 15:
                self.phase = "place"
        elif self.phase == "place":
            action[6] = -1.0
            action[13] = 1.0
            terminal = env.data.xpos[env.handles.cable.terminal_body_id]
            target = env.data.site_xpos[env.handles.target_region_site_id]
            # Preserve the measured right-grasp offset rather than assuming a
            # weld-like fixed transform between the terminal and gripper.
            desired_pinch = target + np.array([0, 0, 0.10]) + (right_pinch - terminal)
            action[7:10] = self._move_action(right_pinch, desired_pinch, scale=0.15)
            state = env._task_state()
            # 先减速到位再松手，不能在运动中掷向目标后算作受控放置。
            ready = (
                np.linalg.norm(desired_pinch - right_pinch) < 0.008
                and state.extra["terminal_speed_m_s"] < 0.02
            )
            if ready or (
                state.extra["supported"] and np.linalg.norm(terminal - target) < 0.04
            ):
                self.phase = "deposit"
                self._phase_steps = 0
        elif self.phase == "deposit":
            action[6] = action[13] = -1
            self._phase_steps += 1
            # 张开后沿工具轴退出约 7 cm，避免端头搁在连杆上；有限距离
            # 退出，不能持续后拉到躯干。最终成功仍由真实托台接触判定。
            if 10 < self._phase_steps < 70:
                action[7] = -0.1
        else:
            raise RuntimeError(f"Unknown handover script phase {self.phase!r}")
        return action
