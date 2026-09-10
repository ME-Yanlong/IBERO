"""卡扣链条复用接触按压控制，完全退出后降低、松爪、在接料盘稳定。"""

import numpy as np
from ibero.control.latch import LatchReleaseScript


class HarnessUnplugScript(LatchReleaseScript):
    def __init__(self, case="normal"):
        super().__init__(case)
        self.place_origin = None

    def action(self, env):
        if self.phase == "hold":
            self.phase = "lower_to_receiver"
            self.place_origin = self.targets[1].copy()
        if self.phase == "lower_to_receiver":
            receiver = env.scene.config["workcell"]["receiver"]
            goal_z = self.place_origin[2] - receiver["lowering_m"]
            self.targets[1][2] = max(
                goal_z,
                self.targets[1][2] - receiver["lowering_speed_m_s"] * env.control_dt,
            )
            if env.data.site_xpos[env.handles.right_anchor_site_id, 2] < goal_z + 0.001:
                self.phase = "deposit"
        action = super().action(env)
        if self.phase == "deposit":
            action[13] = -1
            # 这里只开爪，不改物体位置；自由落差、碰撞与最终静止都由物理求解。
            action[7:10] = np.clip(
                (self.targets[1] - env.data.site_xpos[env.handles.right_anchor_site_id])
                * 0.3
                / env.controller.action_scale[:3],
                -0.05,
                0.05,
            )
        return action
