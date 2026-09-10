"""左无驱动压头、右夹爪的双臂控制；左握力维度明确无对应执行器。"""

from ibero.control.resolved_rate import BimanualResolvedRateController
import numpy as np


class LatchArmController(BimanualResolvedRateController):
    def _set_gripper(self, data, arm_index, grip):
        if arm_index == 1:
            data.ctrl[self.handles.gripper_actuator_ids[0]] = 255 * (
                (float(grip) + 1) / 2
            )


class LatchReleaseScript:
    """按净空与真实接触反馈运行，不使用控制阶段作为解锁真值。"""

    def __init__(self, case="normal"):
        if case not in {
            "normal",
            "no_press",
            "early_release",
            "partial_press",
            "offset_press",
        }:
            raise ValueError("Unknown robot latch case")
        self.case = case
        self.phase = "settle"
        self.targets = None
        self.grasp_time = 0.0

    def action(self, env):
        cfg = env.scene.config["control"]
        dt = env.control_dt
        info = env.last_info
        sites = [env.handles.left_anchor_site_id, env.handles.right_anchor_site_id]
        if self.targets is None:
            self.targets = [env.data.site_xpos[i].copy() for i in sites]
            self.rotations = [env.data.site_xmat[i].reshape(3, 3).copy() for i in sites]
            self.initial = [v.copy() for v in self.targets]
            if self.case == "offset_press":
                self.targets[0][0] += 0.03
        grip = cfg["grip_command"]
        if self.phase == "settle":
            grip = -1
            if env.data.time >= cfg["settle_seconds"]:
                self.phase = "grasp"
        elif self.phase == "grasp":
            self.grasp_time = self.grasp_time + dt if info["grasped"] else 0
            if self.grasp_time >= cfg["grasp_seconds"]:
                self.phase = "pull" if self.case == "no_press" else "press"
        elif self.phase in {"press", "pull"}:
            # 压头过载优先卸载；净空目标只决定运动，不改变锁止几何。
            if self.case != "no_press":
                if self.case == "early_release" and info["withdrawal_m"] > 0.004:
                    self.targets[0][2] = min(
                        self.initial[0][2], self.targets[0][2] + 0.01 * dt
                    )
                else:
                    error = cfg["press_clearance_m"] - info["clearance_m"]
                    velocity = np.clip(error * 1.5, -0.002, cfg["press_speed_m_s"])
                    if info["contact_force_n"] > 2.8:
                        velocity = -0.003
                    self.targets[0][2] -= velocity * dt
                    travel = (
                        cfg["max_press_travel_m"]
                        if self.case != "partial_press"
                        else 0.014
                    )
                    self.targets[0][2] = max(
                        self.targets[0][2], self.initial[0][2] - travel
                    )
            if (
                self.phase == "press"
                and info["clearance_m"] >= cfg["press_clearance_m"] * 0.9
            ):
                self.phase = "pull"
            if self.case in {"partial_press", "offset_press"} and env.data.time > 5:
                self.phase = "pull"
            if self.phase == "pull":
                axis = env.data.xmat[env.socket].reshape(3, 3)[:, 0]
                resistance = abs(np.dot(info["socket_force_world_n"], axis))
                if resistance < cfg["max_pull_force_n"]:
                    self.targets[1] -= axis * cfg["pull_speed_m_s"] * dt
                if info["released"]:
                    # 扣齿脱开后先停拉并让压头退到扣齿上方；边退压头边拉会撞到回弹扣齿。
                    self.targets[1] = env.data.site_xpos[sites[1]].copy()
                    self.phase = "clear_tool"
        elif self.phase == "clear_tool":
            self.targets[0][2] = min(
                self.initial[0][2], self.targets[0][2] + 0.012 * dt
            )
            if env.data.site_xpos[sites[0], 2] >= self.initial[0][2] - 0.003:
                self.phase = "withdraw"
        elif self.phase == "withdraw":
            self.targets[0][2] = min(self.initial[0][2], self.targets[0][2] + 0.01 * dt)
            axis = env.data.xmat[env.socket].reshape(3, 3)[:, 0]
            if info["withdrawal_m"] < info["required_withdrawal_m"] + 0.001:
                self.targets[1] -= axis * cfg["pull_speed_m_s"] * dt
            else:
                self.phase = "hold"
        action = np.zeros(14, dtype=np.float32)
        for arm, site in enumerate(sites):
            offset = 7 * arm
            action[offset : offset + 3] = np.clip(
                (self.targets[arm] - env.data.site_xpos[site])
                * 0.3
                / env.controller.action_scale[:3],
                -0.05,
                0.05,
            )
            current = env.data.site_xmat[site].reshape(3, 3)
            rotation_error = 0.5 * sum(
                np.cross(current[:, i], self.rotations[arm][:, i]) for i in range(3)
            )
            action[offset + 3 : offset + 6] = np.clip(
                rotation_error * 0.2 / env.controller.action_scale[3:], -0.05, 0.05
            )
        action[13] = grip
        return action
