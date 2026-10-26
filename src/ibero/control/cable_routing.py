"""使用仿真真值的双臂布线参考控制器；策略阶段不参与任务成功判断。"""

from __future__ import annotations

import numpy as np


class CableRoutingScript:
    """先抓供料架的裸线，再协同张紧、压入、松手退离。"""

    def __init__(self):
        self.phase = "approach"
        self.phase_ticks = 0

    def reset(self):
        self.phase = "approach"
        self.phase_ticks = 0

    def _next(self, name):
        self.phase = name
        self.phase_ticks = 0

    def action(self, env):
        points = env._points()
        pins = env.data.site_xpos[
            [env.handles.left_anchor_site_id, env.handles.right_anchor_site_id]
        ].copy()
        segment = env.compiled.cable.segments
        left_pick = max(3, segment // 5)
        right_pick = len(points) - 1 - left_pick
        clip_positions = [
            np.asarray(clip["origin_m"], dtype=float)
            for clip in env.scene.config["workcell"]["clips"]
        ]
        seat_x = min(float(c[0]) for c in clip_positions)
        target_z = float(np.mean([c[2] for c in clip_positions]))
        midpoint = points[len(points) // 2]
        if self.phase == "approach":
            targets = np.array([points[left_pick], points[right_pick]])
            grip = -1.0
            if np.max(np.linalg.norm(targets - pins, axis=1)) < 0.008 or self.phase_ticks > 30:
                self._next("grasp")
        elif self.phase == "grasp":
            targets = np.array([points[left_pick], points[right_pick]])
            grip = 1.0
            if env._grasped("left") and env._grasped("right"):
                self._next("align")
        elif self.phase == "align":
            targets = np.array(
                [[0.423, 0.108, target_z + 0.013], [0.423, -0.108, target_z + 0.013]]
            )
            grip = 1.0
            if np.max(np.linalg.norm(targets - pins, axis=1)) < 0.010 or self.phase_ticks > 45:
                self._next("press")
        elif self.phase == "press":
            # 张力是实测反馈；目标高度修正线材的实际下垂，绝不改变 EI。
            desired_z = np.clip(pins[:, 2] + (target_z - midpoint[2]) * 0.5, 0.84, 0.885)
            targets = np.array(
                [[seat_x + 0.007, 0.108, desired_z[0]],
                 [seat_x + 0.007, -0.108, desired_z[1]]]
            )
            grip = 1.0
            if env._last_info.get("metrics", {}).get("retained_count", 0) >= len(clip_positions):
                self._next("release")
        else:
            # 退离之后任务判据才能计时，双手没有继续托住线束。
            targets = np.array([[0.38, 0.23, 0.93], [0.38, -0.23, 0.93]])
            grip = -1.0
        action = np.zeros(14, dtype=np.float32)
        for side in range(2):
            index = side * 7
            action[index:index + 3] = np.clip(
                (targets[side] - pins[side]) * 0.3 / env.controller.action_scale[:3],
                -0.25,
                0.25,
            )
            action[index + 6] = grip
        self.phase_ticks += 1
        return action


def routing_policy_for(scene_id: str, kind: str | None = None):
    """按菜谱类型选参考控制器；贡献场景可换 ID 而不改 Core。"""
    recipe_kind = kind or scene_id
    if recipe_kind in {"cable_route_l", "cable_route_rectangle"}:
        from ibero.control.routing_patterns import RoutingPatternScript

        return RoutingPatternScript()
    if recipe_kind in {"cable_clip_insert", "cable_route_straight"}:
        return CableRoutingScript()
    raise ValueError(f"Unknown routing policy kind: {recipe_kind}")
