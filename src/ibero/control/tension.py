"""双臂导纳张力控制：真实接触，基于测量张力改变端点间距。"""

import numpy as np


def orientation_error(current, target):
    return 0.5 * sum(
        (np.cross(current[:, i], target[:, i]) for i in range(3)), np.zeros(3)
    )


class CableStretchScript:
    def __init__(self):
        self.filtered = None
        self.rotations = None
        self.centers = None
        self.phase = "tensioning"
        self.unloading = False

    def action(self, env):
        cfg = env.scene.config["control"]
        ids = [env.handles.left_pinch_site_id, env.handles.right_pinch_site_id]
        poses = env.data.site_xpos[ids].copy()
        if self.rotations is None:
            self.rotations = env.data.site_xmat[ids].reshape(2, 3, 3).copy()
            self.centers = poses.copy()
        measured = env._cable_tension()
        alpha = env.control_dt / (cfg["filter_time_s"] + env.control_dt)
        self.filtered = (
            measured
            if self.filtered is None
            else self.filtered + alpha * (measured - self.filtered)
        )
        self.unloading = measured > cfg["warning_tension_n"] or (
            self.unloading and measured > cfg["recovery_tension_n"]
        )
        velocity = (
            -cfg["max_speed_m_s"]
            if self.unloading
            else np.clip(
                cfg["admittance_m_ns"] * (cfg["target_tension_n"] - self.filtered),
                -cfg["max_speed_m_s"],
                cfg["max_speed_m_s"],
            )
        )
        self.phase = (
            "unloading"
            if self.unloading
            else "holding"
            if abs(measured - cfg["target_tension_n"]) < cfg["tension_tolerance_n"]
            else "tensioning"
        )
        direction = poses[0] - poses[1]
        direction /= np.linalg.norm(direction)
        action = np.zeros(14, dtype=np.float32)
        for i, site in enumerate(ids):
            delta = (1 if i == 0 else -1) * direction * velocity * env.control_dt / 2
            # 中心位置反馈抑制两侧同步漂移，不干扰相对距离张力自由度。
            delta += 0.1 * (self.centers.mean(axis=0) - poses.mean(axis=0))
            action[7 * i : 7 * i + 3] = delta / env.controller.action_scale[:3]
            action[7 * i + 3 : 7 * i + 6] = (
                orientation_error(
                    env.data.site_xmat[site].reshape(3, 3), self.rotations[i]
                )
                * 0.3
                / 0.1
            )
            action[7 * i + 6] = 1
        return np.clip(action, -1, 1)
