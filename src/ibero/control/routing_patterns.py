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
        index_right = len(points) - 1 - index_left
        z0 = float(config_clips[0]["origin_m"][2])
        x0 = float(config_clips[0]["origin_m"][0])
        rectangle = len(config_clips) > 4
        hold_left = np.array([x0 + 0.007, 0.075 if rectangle else 0.108, z0 + 0.013])

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
                [0.423, 0.075 if rectangle else 0.108, z0 + 0.013],
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
                [x0 + 0.007, 0.075 if rectangle else 0.108, corrected_z[0]],
                [x0 + 0.007, -0.108, corrected_z[1]],
            ])
            grips = (1.0, 1.0)
            if first_two:
                self._next("regrasp_open")
        elif self.phase == "regrasp_open":
            # 两只已夹住的线夹暂时承载线段；右手松开旧抓点，
            # 沿自由尾段重新抓到更靠末端的位置，给转角预留材料弧长。
            targets = np.array([hold_left, points[-2]])
            grips = (1.0, -1.0)
            if (not env._grasped("right")
                    and np.linalg.norm(targets[1] - pins[1]) < 0.010):
                self._next("regrasp_close")
        elif self.phase == "regrasp_close":
            targets = np.array([hold_left, points[-2]])
            grips = (1.0, 1.0)
            if env._grasped("right"):
                self._next("corner")
        elif self.phase == "corner":
            # 第三只线夹形成第一个真实转角：右手抬高，第三处布线段
            # 在 y≈-0.04 处接近其竖直夹口。
            third = config_clips[2]
            yz = np.asarray(third["origin_m"], dtype=float)
            targets = np.array([
                hold_left,
                [x0 - 0.020, -0.100, yz[2] + 0.033],
            ])
            grips = (1.0, 1.0)
            if metrics.get(third["id"] + "_retained", False):
                self._next("up")
        elif self.phase == "up":
            fourth = config_clips[3]
            yz = np.asarray(fourth["origin_m"], dtype=float)
            targets = np.array([
                hold_left,
                [x0 - 0.020, yz[1], yz[2] + 0.013],
            ])
            grips = (1.0, 1.0)
            if metrics.get(fourth["id"] + "_retained", False):
                self._next("across" if rectangle else "release")
        elif self.phase == "across":
            fifth = config_clips[4]
            yz = np.asarray(fifth["origin_m"], dtype=float)
            targets = np.array([
                hold_left,
                [x0 - 0.020, yz[1] + 0.034, yz[2] + 0.015],
            ])
            grips = (1.0, 1.0)
            if metrics.get(fifth["id"] + "_retained", False):
                self._next("down")
        elif self.phase == "down":
            sixth = config_clips[5]
            yz = np.asarray(sixth["origin_m"], dtype=float)
            targets = np.array([
                hold_left,
                [x0 - 0.020, yz[1] + 0.020, yz[2] - 0.025],
            ])
            grips = (1.0, 1.0)
            if metrics.get(sixth["id"] + "_retained", False):
                self._next("release")
        else:
            targets = np.array([[0.38, 0.23, 0.93], [0.38, -0.23, 0.93]])
            grips = (-1.0, -1.0)

        self.phase_ticks += 1
        return self._action(env, pins, targets, grips)
