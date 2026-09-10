"""A small, deterministic dual-arm resolved-rate controller.

Core-0 uses this controller rather than exposing joint actions.  It lets the
environment test the intended IBERO contract (two 6D end-effector deltas plus
two gripper commands) before adding a more general planning stack.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ibero.robots.g1_upperbody import ModelHandles


class BimanualResolvedRateController:
    """Map a normalized 14D action to G1 position-actuator targets."""

    # 每臂动作的前 6 维为 [xyz 平移, xyz 转动] 增量；0.012 m/tick 与
    # 0.10 rad/tick 是为端头接触阶段保守设置的上限，避免一帧跨过夹持区。
    action_scale = np.asarray([0.012, 0.012, 0.012, 0.10, 0.10, 0.10])

    def __init__(
        self, model: mujoco.MjModel, handles: ModelHandles, control_dt: float
    ) -> None:
        self.model = model
        self.handles = handles
        self.control_dt = control_dt
        self.arm_joint_ids = model.actuator_trnid[handles.arm_actuator_ids, 0]
        self.arm_qpos_adr = model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dof_adr = model.jnt_dofadr[self.arm_joint_ids]
        self._site_ids = np.asarray(
            [handles.left_anchor_site_id, handles.right_anchor_site_id], dtype=np.int32
        )
        self._arm_targets: np.ndarray | None = None
        self._hold_targets: np.ndarray | None = None

    def reset(self, data: mujoco.MjData) -> None:
        """Make position-control targets agree with the reset configuration."""

        self._arm_targets = data.qpos[self.arm_qpos_adr].copy()
        hold_qpos = []
        for actuator_id in self.handles.hold_actuator_ids:
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            if joint_id >= 0:
                hold_qpos.append(data.qpos[self.model.jnt_qposadr[joint_id]])
            else:
                hold_qpos.append(0.0)
        self._hold_targets = np.asarray(hold_qpos, dtype=np.float64)
        self.hold_pose(data)
        # 复位先闭合两侧 2F-85；Core-0.1 的实际接触 task 会在环境 reset 中
        # 再明确设定“左闭右开”，这里仅保证 position controller 状态自洽。
        data.ctrl[self.handles.gripper_actuator_ids] = 220.0

    def apply(self, data: mujoco.MjData, action: np.ndarray) -> None:
        """Apply a normalized 14D command at the current policy tick."""

        action = np.asarray(action, dtype=np.float64)
        if not np.isfinite(action).all():
            raise ValueError("Action must contain only finite values")
        if action.shape != (14,):
            raise ValueError(
                f"Bimanual action must have shape (14,), got {action.shape}"
            )
        action = np.clip(action, -1.0, 1.0)
        self._hold_non_arm(data)

        if self._arm_targets is None:
            self._arm_targets = data.qpos[self.arm_qpos_adr].copy()
        arm_targets = self._arm_targets.copy()
        for arm_index, site_id in enumerate(self._site_ids):
            offset = arm_index * 7
            delta = action[offset : offset + 6] * self.action_scale
            # MuJoCo site Jacobians are expressed in world coordinates.
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jac_pos, jac_rot, int(site_id))
            arm_dofs = self.arm_dof_adr[offset : offset + 7]
            jacobian = np.vstack((jac_pos[:, arm_dofs], jac_rot[:, arm_dofs]))
            desired_velocity = delta / self.control_dt
            # DLS 阻尼是奇异位姿附近的数值保护参数；太小会放大腕部速度，
            # 太大会让末端难以抵达端头。修改后须重跑 20-seed handover 回归。
            damping = 2.5e-3
            joint_velocity = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), desired_velocity
            )
            joint_velocity = np.clip(joint_velocity, -2.0, 2.0)
            target = arm_targets[offset : offset + 7] + joint_velocity * self.control_dt
            joint_ids = self.arm_joint_ids[offset : offset + 7]
            target = np.clip(
                target,
                self.model.jnt_range[joint_ids, 0],
                self.model.jnt_range[joint_ids, 1],
            )
            arm_targets[offset : offset + 7] = target

            grip = action[offset + 6]
            self._set_gripper(data, arm_index, grip)

        data.ctrl[self.handles.arm_actuator_ids] = arm_targets
        self._arm_targets = arm_targets

    def _hold_non_arm(self, data: mujoco.MjData) -> None:
        if self._hold_targets is None:
            self._hold_targets = np.zeros(
                len(self.handles.hold_actuator_ids), dtype=np.float64
            )
            for index, actuator_id in enumerate(self.handles.hold_actuator_ids):
                joint_id = self.model.actuator_trnid[actuator_id, 0]
                if joint_id >= 0:
                    self._hold_targets[index] = data.qpos[
                        self.model.jnt_qposadr[joint_id]
                    ]
        for index, actuator_id in enumerate(self.handles.hold_actuator_ids):
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            if joint_id >= 0:
                data.ctrl[actuator_id] = self._hold_targets[index]

    def hold_pose(self, data: mujoco.MjData) -> None:
        """Apply the persistent reset targets without moving either arm."""

        if self._arm_targets is None:
            self.reset(data)
            return
        self._hold_non_arm(data)
        data.ctrl[self.handles.arm_actuator_ids] = self._arm_targets

    def _set_gripper(self, data: mujoco.MjData, arm_index: int, grip: float) -> None:
        # Menagerie 的 2F-85 控制范围是 [0, 255]（张开到闭合）。Core-0.1
        # 映射完整范围，保证 scripted baseline 的“左手释放”是可观测接触事件。
        command = 255.0 * ((float(grip) + 1.0) / 2.0)
        data.ctrl[self.handles.gripper_actuator_ids[arm_index]] = command
