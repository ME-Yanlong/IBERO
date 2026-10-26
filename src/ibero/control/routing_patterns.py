"""立式板转角任务的真值参考策略候选。

任务判定独立读取 Flex/夹口状态；这里的 phase 只决定机器人动作。
长路径需要先固定一段，再利用已固定的线段引导自由端转向。
"""

from __future__ import annotations

import numpy as np


class RoutingPatternScript:
    """按实体线夹反馈推进；一个阶段只改变双手目标，不移动线材。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.phase_ticks = 0

    def _next(self, phase):
        self.phase, self.phase_ticks = phase, 0

    @staticmethod
    def _guide_touches_cable(env):
        """具名鼻头的末端须在转角材料段有真实接触；杆身接触不算抓住。"""
        contact = env.last_info.get("guide_contact", {})
        return bool(contact.get("tip_touching_cable")) and bool(
            {11, 12, 13}.intersection(contact.get("tip_elements", ()))
        )

    @staticmethod
    def _rear_lip_touches_cable(env):
        """压入时只接受叉子后挡唇与目标材料段的实体接触。"""
        contact = env.last_info.get("guide_contact", {})
        instant = env._guide_contact()
        return (contact.get("rear_lip_contact_substeps", 0) >= 150
                and bool(instant.get("rear_lip_touching_cable"))
                and bool({13, 14}.intersection(instant.get("rear_lip_elements", ()))))

    @staticmethod
    def _action(env, pins, targets, grips):
        action = np.zeros(14, dtype=np.float32)
        for side in range(2):
            index = side * 7
            action[index:index + 3] = np.clip(
                (targets[side] - pins[side]) * 0.3 / env.controller.action_scale[:3],
                -0.25, 0.25,
            )
            action[index + 6] = grips[side]
        return action

    def action(self, env):
        points = env._points()
        pins = env.data.site_xpos[
            [env.handles.left_anchor_site_id, env.handles.right_anchor_site_id]
        ].copy()
        config_clips = env.scene.config["workcell"]["clips"]
        metrics = env.last_info.get("metrics", {})
        first_two = all(metrics.get(clip["id"] + "_retained", False)
                        for clip in config_clips[:2])
        index_left = max(3, env.compiled.cable.segments // 5)
        rectangle = len(config_clips) > 4
        guide_tool = env.scene.config["workcell"].get("guide_tool")
        # L 路径把右手放到更靠自由端的真实线段，夹口处留出指垫净空；
        # 这是从供料架重新选择抓点，不修改线材的任何物理状态。
        index_right = len(points) - 3
        z0 = float(config_clips[0]["origin_m"][2])
        x0 = float(config_clips[0]["origin_m"][0])
        clear_left = np.array([0.38, 0.23, 0.93])
        # L 任务已有实测反例：左手高位托住尾段会在夹口旁形成小于
        # 20 mm 的局部折弯；在座面同高保持更平顺。
        route_left = clear_left if rectangle else np.array([0.455, 0.092, z0])
        route_grips = (-1.0, 1.0) if rectangle else (1.0, 1.0)
        # 只用于控制器中止：实体压头深穿 Flex 或载荷过大时停止继续加力。
        # 线材的 20 mm 最小半径和线夹碰撞门槛仍由任务与审计独立判定。
        guide_contact = env.last_info.get("guide_contact", {})
        if (self.phase.startswith("guide_") and self.phase != "guide_abort"
                and (guide_contact.get("penetration_m", 0) > 0.003
                     or guide_contact.get("normal_force_n", 0) > 5.0
                     or guide_contact.get("clip_collision_penetration_m", 0) > 0.001
                     or guide_contact.get("clip_collision_force_n", 0) > 5.0)):
            self._next("guide_abort")

        if self.phase == "approach":
            targets = np.array([points[index_left], points[index_right]])
            grips = (-1.0, -1.0)
            if np.max(np.linalg.norm(targets - pins, axis=1)) < 0.008 or self.phase_ticks > 30:
                self._next("grasp")
        elif self.phase == "grasp":
            targets = np.array([points[index_left], points[index_right]])
            grips = (1.0, 1.0)
            if env._grasped("left") and env._grasped("right"):
                self._next("align")
        elif self.phase == "align":
            targets = np.array([
                [0.423, 0.108, z0 + 0.013],
                [0.423, -0.108, z0 + 0.013],
            ])
            grips = (1.0, 1.0)
            if np.max(np.linalg.norm(targets - pins, axis=1)) < 0.010 or self.phase_ticks > 45:
                self._next("press_first")
        elif self.phase == "press_first":
            # 第一段水平线夹沿局部法向压入，保持真实弯曲与张力反馈。
            midpoint = points[len(points) // 2]
            corrected_z = np.clip(pins[:, 2] + (z0 - midpoint[2]) * 0.5, 0.80, 0.90)
            targets = np.array([
                [x0 + 0.007, 0.108, corrected_z[0]],
                [x0 + 0.007, -0.108, corrected_z[1]],
            ])
            grips = (1.0, 1.0)
            if first_two:
                self._next(
                    "free_left" if rectangle else
                    "guide_under" if guide_tool else "support_retract"
                )
        elif self.phase == "guide_under":
            # 从真实转角线段下方绕到板侧，鼻头到位后再抬升；避免从线外侧
            # 直接推向背座，造成越压越深的错误受力方向。
            guide_id = env.handles.guide_lead_tip_site_id
            guide_tip = env.data.site_xpos[guide_id]
            target_tip = np.array([0.490, -0.090, z0 + 0.003])
            right_pinch_target = pins[1] + target_tip - guide_tip
            targets = np.array([route_left, right_pinch_target])
            grips = (1.0, -1.0)
            if np.linalg.norm(target_tip - guide_tip) < 0.006:
                self._next("guide_behind")
        elif self.phase == "guide_behind":
            # 从下方使实体端帽进入线束背侧；必须观测目标 Flex 元素接触。
            guide_tip = env.data.site_xpos[env.handles.guide_lead_tip_site_id]
            target_tip = points[14] + np.array([0.004, -0.003, -0.003])
            targets = np.array([route_left, pins[1] + target_tip - guide_tip])
            grips = (1.0, -1.0)
            if self._rear_lip_touches_cable(env):
                self._next("guide_outer")
            elif self.phase_ticks > 45:
                self._next("guide_abort")
        elif self.phase == "guide_outer":
            # 先从线下绕到材料的 -Y 外侧，再由外向内推；不能从内侧空推。
            guide_tip = env.data.site_xpos[env.handles.guide_tip_site_id]
            target_tip = np.array([0.450, points[14, 1] - 0.012, 0.880])
            targets = np.array([route_left, pins[1] + target_tip - guide_tip])
            grips = (1.0, -1.0)
            if guide_tip[1] <= points[14, 1] - 0.009:
                self._next("guide_center")
            elif self.phase_ticks > 20:
                self._next("guide_abort")
        elif self.phase == "guide_center":
            # 后挡唇在材料外侧，先横向导向夹口中线，同时保持当前高度。
            guide_tip = env.data.site_xpos[env.handles.guide_tip_site_id]
            third = config_clips[2]
            y, z = third["origin_m"][1:]
            target_tip = np.array([0.450, y - 0.007, z - 0.012])
            targets = np.array([route_left, pins[1] + target_tip - guide_tip])
            grips = (1.0, -1.0)
            if (points[13, 1] >= y - 0.010
                    and self._rear_lip_touches_cable(env)):
                self._next("guide_raise")
            elif self.phase_ticks > 25:
                self._next("guide_abort")
        elif self.phase == "guide_raise":
            # 横向对中后沿切向抬线，真实后挡唇接触不能丢失。
            guide_tip = env.data.site_xpos[env.handles.guide_tip_site_id]
            third = config_clips[2]
            y, z = third["origin_m"][1:]
            target_tip = np.array([0.460, y - 0.005, z + 0.003])
            targets = np.array([route_left, pins[1] + target_tip - guide_tip])
            grips = (1.0, -1.0)
            if (points[13, 2] >= z - 0.004
                    and metrics.get(third["id"] + "_mouth_now", False)
                    and self._rear_lip_touches_cable(env)):
                self._next("guide_press")
            elif self.phase_ticks > 25:
                self._next("guide_abort")
        elif self.phase == "guide_press":
            # 当前线段在口部且后挡唇真实接触，沿板面法向 +X 压入夹座。
            guide_id = env.model.site("right_guide_tip").id
            guide_tip = env.data.site_xpos[guide_id]
            third = config_clips[2]
            y, z = third["origin_m"][1:]
            target_tip = np.array([0.475, y - 0.008, z + 0.006])
            targets = np.array([route_left, pins[1] + target_tip - guide_tip])
            grips = (1.0, -1.0)
            if metrics.get(third["id"] + "_retained", False):
                self._next("up")
            elif (self.phase_ticks >= 2
                  and not metrics.get(third["id"] + "_seated_now", False)
                  and (not self._rear_lip_touches_cable(env)
                       or self.phase_ticks > 50)):
                self._next("guide_abort")
        elif self.phase == "guide_abort":
            targets = pins.copy()
            grips = (-1.0, -1.0)
        elif self.phase == "support_retract":
            # 右手继续保持远端裸线，左手真实松开旧抓点并退到板外；
            # 下一步再夹住转角下方的中间材料段。
            targets = np.array([clear_left, pins[1]])
            grips = (-1.0, 1.0)
            if not env._grasped("left") and self.phase_ticks >= 8:
                self._next("support_approach")
        elif self.phase == "support_approach":
            targets = np.array([points[10], pins[1]])
            grips = (-1.0, 1.0)
            if np.linalg.norm(targets[0] - pins[0]) < 0.008:
                self._next("support_grasp")
        elif self.phase == "support_grasp":
            targets = np.array([points[10], pins[1]])
            grips = (1.0, 1.0)
            if env._grasped("left"):
                self._next("corner_outside")
        elif self.phase == "corner_low":
            # 先把目标线段沿板面引向转角入口，同时保留足够弧长；
            # 然后再抬高，避免直接抬腕在第二夹边缘折出急弯。
            targets = np.array([
                route_left,
                [x0 - 0.010, -0.055, z0 + 0.004],
            ])
            grips = (1.0, 1.0)
            if np.linalg.norm(targets[1] - pins[1]) < 0.010 or self.phase_ticks > 45:
                self._next("corner_outside")
        elif self.phase == "corner_outside":
            # 沿真正的线夹开口正面布线：先让目标线段处在唇边之外，
            # 随后的 corner 阶段才沿板面法向把它压入。
            third = config_clips[2]
            y, z = third["origin_m"][1:]
            targets = (
                np.array([[0.430, -0.035, 0.865], [0.430, -0.090, 0.910]])
                if not rectangle else
                np.array([route_left, [x0 - 0.040, y - 0.010, z + 0.015]])
            )
            grips = (1.0, 1.0)
            if metrics.get(third["id"] + "_mouth_seen", False):
                self._next("corner")
        elif self.phase == "free_left":
            # 已入座的两只实体线夹承载第一段。左手放开原始尾段，
            # 以免它和夹口之间形成很短的高曲率弯折。
            targets = np.array([clear_left, pins[1]])
            grips = (-1.0, 1.0)
            if not env._grasped("left") and self.phase_ticks >= 6:
                # L 的竖直段缩到在右手当前材料弧长内可操作，直接转向；
                # 四边任务仍需要另一次真实换抓。
                self._next("regrasp_open" if rectangle else "corner")
        elif self.phase == "regrasp_open":
            # 两只已夹住的线夹暂时承载线段；右手松开旧抓点，
            # 沿自由尾段重新抓到更靠末端的位置，给转角预留材料弧长。
            targets = np.array([clear_left, points[-2]])
            grips = (-1.0, -1.0)
            if (not env._grasped("right")
                    and np.linalg.norm(targets[1] - pins[1]) < 0.010):
                self._next("regrasp_close")
        elif self.phase == "regrasp_close":
            targets = np.array([clear_left, points[-2]])
            grips = (-1.0, 1.0)
            if env._grasped("right"):
                self._next("corner")
        elif self.phase == "corner":
            # 第三只线夹形成第一个真实转角：右手抬高，第三处布线段
            # 在 y≈-0.04 处接近其竖直夹口。
            third = config_clips[2]
            yz = np.asarray(third["origin_m"], dtype=float)
            targets = (
                np.array([[0.450, -0.035, 0.865], [0.450, -0.090, 0.910]])
                if not rectangle else
                np.array([route_left, [x0 - 0.020, yz[1] - 0.040, yz[2] + 0.033]])
            )
            grips = route_grips
            if metrics.get(third["id"] + "_retained", False):
                self._next("up")
        elif self.phase == "up":
            fourth = config_clips[3]
            yz = np.asarray(fourth["origin_m"], dtype=float)
            targets = np.array([
                route_left,
                [x0 - 0.020, yz[1], yz[2] + 0.013],
            ])
            grips = route_grips
            if metrics.get(fourth["id"] + "_retained", False):
                self._next("across" if rectangle else "release")
        elif self.phase == "across":
            fifth = config_clips[4]
            yz = np.asarray(fifth["origin_m"], dtype=float)
            targets = np.array([
                clear_left,
                [x0 - 0.020, yz[1] + 0.034, yz[2] + 0.015],
            ])
            grips = (-1.0, 1.0)
            if metrics.get(fifth["id"] + "_retained", False):
                self._next("down")
        elif self.phase == "down":
            sixth = config_clips[5]
            yz = np.asarray(sixth["origin_m"], dtype=float)
            targets = np.array([
                clear_left,
                [x0 - 0.020, yz[1] + 0.020, yz[2] - 0.025],
            ])
            grips = (-1.0, 1.0)
            if metrics.get(sixth["id"] + "_retained", False):
                self._next("release")
        else:
            targets = np.array([[0.38, 0.23, 0.93], [0.38, -0.23, 0.93]])
            grips = (-1.0, -1.0)

        self.phase_ticks += 1
        action = self._action(env, pins, targets, grips)
        if self.phase in ("guide_outer", "guide_center", "guide_raise"):
            # 弹性线段在近口处需缓慢抬升，避免单控制帧的接触冲击。
            action[7:10] *= 0.4
        return action
