"""双臂布线的任务判定；只观察线、线夹、机器人和安全审计的事实。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ibero.core.task_api import TaskResult, TaskSpec
from ibero.tasks.routing_geometry import inspect_order, inspect_route, match_clip


@dataclass(frozen=True)
class RoutingGoal:
    """场景作者定义的目标与容差，所有长度为 m、时间为 s。"""

    clip_ids: tuple[str, ...]
    route_points_m: tuple[tuple[float, float, float], ...] = ()
    corridor_m: float = 0.02
    hold_time_s: float = 0.2
    speed_limit_m_s: float = 0.04
    min_bend_radius_m: float = 0.0
    min_clip_arc_gap_m: float = 0.01
    backtrack_tolerance_m: float = 0.005
    require_two_hand_grasp: bool = True
    require_entry_history: bool = True
    free_end_center_m: tuple[float, float, float] | None = None
    free_end_radius_m: float | None = None
    free_end_index: int = -1

    @classmethod
    def from_scene_config(cls, config):
        """让目标路径/容差只有菜谱一份权威来源。"""
        goal = config["routing_goal"]
        cable = config["materials"]["cable"]
        return cls(
            clip_ids=tuple(goal["ordered_clips"]),
            route_points_m=tuple(tuple(point) for point in goal["route_points_m"]),
            corridor_m=float(goal["corridor_m"]),
            hold_time_s=float(goal["hold_time_s"]),
            speed_limit_m_s=float(goal["speed_limit_m_s"]),
            min_bend_radius_m=float(cable["minimum_bend_radius_m"]),
        )

    def __post_init__(self):
        if not self.clip_ids or len(set(self.clip_ids)) != len(self.clip_ids):
            raise ValueError("Routing goal requires distinct clip IDs")
        positive = ("corridor_m", "hold_time_s", "speed_limit_m_s", "min_clip_arc_gap_m")
        nonnegative = ("min_bend_radius_m", "backtrack_tolerance_m")
        for name in positive + nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or (value <= 0 if name in positive else value < 0):
                raise ValueError(f"Invalid routing goal {name}")
        if len(self.clip_ids) > 1:
            route = np.asarray(self.route_points_m, dtype=float)
            if route.ndim != 2 or route.shape[1] != 3 or len(route) < 2:
                raise ValueError("Multi-clip route requires a 3-D target polyline")
            if not np.isfinite(route).all():
                raise ValueError("Route contains non-finite points")
        if self.free_end_center_m is not None:
            center = np.asarray(self.free_end_center_m, dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError("Free-end target must be a finite 3-D point")
            if self.free_end_radius_m is None or not np.isfinite(self.free_end_radius_m) or self.free_end_radius_m <= 0:
                raise ValueError("Free-end radius must be positive")
        elif self.free_end_radius_m is not None:
            raise ValueError("Free-end radius requires a target center")
        if self.free_end_index not in (-1, 0):
            raise ValueError("Free-end index must identify the first or last cable vertex")


class CableRoutingTask(TaskSpec):
    """Same judge for one clip, a line, an L and an open rectangle.

    `state.extra['routing']` is an observation snapshot, not a command. It has:
    `cable_points_m`, `clips`, `dt_s`, `cable_speed_m_s`, `hands_clear`,
    optional `minimum_radius_m` and `invalid_reason`.
    Each clip contains the local seat geometry, `jaw_recovered` and
    `jaw_gap_m` from the physical clip observer, wire radius, and measured
    penetration/limit.
    """

    def __init__(self, goal: RoutingGoal, *, task_id: str = "cable_routing", instruction: str | None = None):
        self.goal = goal
        self.id = task_id
        self.instruction = instruction or "Route the cable through the specified clips, then release both hands."
        self.reset()

    def reset(self) -> None:
        self.hold_seconds = 0.0
        self.seen_unseated = {clip_id: False for clip_id in self.goal.clip_ids}
        self.seen_mouth = {clip_id: False for clip_id in self.goal.clip_ids}
        self.ever_seated = {clip_id: False for clip_id in self.goal.clip_ids}
        self.seen_bimanual_grasp = False
        self.last_metrics: dict[str, float | bool | str] = {}

    @staticmethod
    def _clips(value) -> dict[str, Mapping]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Routing snapshot clips must be a list")
        clips: dict[str, Mapping] = {}
        for item in value:
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
                raise ValueError("Clip snapshot requires a string ID")
            if item["id"] in clips:
                raise ValueError("Duplicate clip snapshot ID")
            clips[item["id"]] = item
        return clips

    def _invalid(self, reason: str) -> TaskResult:
        return TaskResult(0.0, "invalid", terminated=True, failure_reason=f"invalid:{reason}")

    def evaluate(self, state, audit) -> TaskResult:
        try:
            routing = state.extra["routing"]
            if not isinstance(routing, Mapping):
                raise ValueError("Routing snapshot must be a mapping")
            invalid_reason = routing.get("invalid_reason")
            if invalid_reason:
                return self._invalid(str(invalid_reason))
            points = np.asarray(routing["cable_points_m"], dtype=float)
            clips = self._clips(routing["clips"])
            dt = float(routing["dt_s"])
            speed = float(routing["cable_speed_m_s"])
            hands_clear = routing["hands_clear"]
            if self.goal.min_bend_radius_m > 0 and "minimum_radius_m" not in routing:
                raise ValueError("Minimum bending radius observation is required")
            min_radius = float(routing.get("minimum_radius_m", float("inf")))
            if not (np.isfinite(dt) and dt > 0 and np.isfinite(speed) and speed >= 0):
                raise ValueError("Routing clock/speed invalid")
            if not isinstance(hands_clear, (bool, np.bool_)):
                raise ValueError("hands_clear must be measured as a bool")
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError("Cable centerline invalid")
            if np.isnan(min_radius) or min_radius <= 0:
                raise ValueError("Minimum bending radius invalid")
            if any(clip_id not in clips for clip_id in self.goal.clip_ids):
                raise ValueError("Goal clip missing from observation")
            matches = {clip_id: match_clip(points, clips[clip_id]) for clip_id in self.goal.clip_ids}
            for clip_id, clip in ((name, clips[name]) for name in self.goal.clip_ids):
                penetration = float(clip["max_penetration_m"])
                limit = float(clip["penetration_limit_m"])
                if not (np.isfinite(penetration) and penetration >= 0 and np.isfinite(limit) and limit > 0):
                    raise ValueError("Clip penetration data invalid")
                gap = float(clip["jaw_gap_m"])
                wire_radius = float(clip["wire_radius_m"])
                if not (np.isfinite(gap) and gap >= 0 and np.isfinite(wire_radius) and wire_radius > 0):
                    raise ValueError("Clip gap/wire radius data invalid")
            # 进场历史必须来自真实线束运动；当前位置为入座不能伪造过程。
            for clip_id, match in matches.items():
                self.seen_unseated[clip_id] |= not match.seated
                self.seen_mouth[clip_id] |= match.mouth and self.seen_unseated[clip_id]
                self.ever_seated[clip_id] |= match.seated
            self.seen_bimanual_grasp |= bool(
                state.grasps.get("left_gripper", False)
                and state.grasps.get("right_gripper", False)
            )
            retained = {
                clip_id: bool(
                    match.seated
                    and clips[clip_id]["jaw_recovered"]
                    # 夹臂回到原位仍可能张得足够宽，线可直接退出；
                    # 净口宽须小于线径，且两者均来自物理观察/材料参数。
                    and float(clips[clip_id]["jaw_gap_m"])
                    < 2 * float(clips[clip_id]["wire_radius_m"])
                    and float(clips[clip_id]["max_penetration_m"])
                    <= float(clips[clip_id]["penetration_limit_m"])
                    and (
                        not self.goal.require_entry_history
                        or self.seen_mouth[clip_id]
                    )
                )
                for clip_id, match in matches.items()
            }
            order_ok = inspect_order(self.goal.clip_ids, matches, self.goal.min_clip_arc_gap_m)
            path_error = -1.0
            path_available = len(self.goal.clip_ids) == 1
            path_monotone = True
            if len(self.goal.clip_ids) > 1 and order_ok:
                first = matches[self.goal.clip_ids[0]].arc_m
                last = matches[self.goal.clip_ids[-1]].arc_m
                path_error, path_monotone = inspect_route(
                    points,
                    self.goal.route_points_m,
                    first,
                    last,
                    backtrack_tolerance_m=self.goal.backtrack_tolerance_m,
                )
                path_available = True
            elif len(self.goal.clip_ids) > 1:
                path_monotone = False
            end_ok = True
            if self.goal.free_end_center_m is not None:
                end_ok = bool(
                    np.linalg.norm(points[self.goal.free_end_index] - self.goal.free_end_center_m)
                    <= self.goal.free_end_radius_m
                )
            bend_ok = min_radius >= self.goal.min_bend_radius_m
            release_ok = bool(hands_clear) and not any(state.grasps.values())
            prior_grasp_ok = not self.goal.require_two_hand_grasp or self.seen_bimanual_grasp
            stable = bool(
                all(retained.values()) and order_ok and path_monotone
                and (len(self.goal.clip_ids) == 1 or
                     (path_available and path_error <= self.goal.corridor_m))
                and end_ok and bend_ok
                and release_ok and prior_grasp_ok and speed <= self.goal.speed_limit_m_s
            )
            self.hold_seconds = self.hold_seconds + dt if stable else 0.0
            metrics: dict[str, float | bool | str] = {
                "retained_count": float(sum(retained.values())),
                "clip_count": float(len(self.goal.clip_ids)),
                "ordered": order_ok,
                "path_error_m": float(path_error),
                "path_available": path_available,
                "path_monotone": path_monotone,
                "end_zone_ok": end_ok,
                "bend_ok": bend_ok,
                "hands_clear": release_ok,
                "seen_bimanual_grasp": self.seen_bimanual_grasp,
                "cable_speed_m_s": speed,
                "hold_seconds": self.hold_seconds,
            }
            for clip_id in self.goal.clip_ids:
                metrics[f"{clip_id}_retained"] = retained[clip_id]
                metrics[f"{clip_id}_mouth_seen"] = self.seen_mouth[clip_id]
                arc = matches[clip_id].arc_m
                metrics[f"{clip_id}_arc_m"] = -1.0 if arc is None else float(arc)
            self.last_metrics = metrics
        except (KeyError, TypeError, ValueError, IndexError) as error:
            return self._invalid(str(error))

        if audit.cable_damage:
            return TaskResult(-1.0, "failed", terminated=True,
                              failure_reason="cable_damage", metrics=metrics)
        if audit.self_collision:
            return TaskResult(-1.0, "failed", terminated=True,
                              failure_reason="self_collision", metrics=metrics)
        if self.hold_seconds >= self.goal.hold_time_s:
            return TaskResult(1.0, "success", terminated=True, success=True, metrics=metrics)
        if state.elapsed_steps >= state.max_episode_steps:
            return TaskResult(0.0, "timeout", truncated=True,
                              failure_reason="timeout", metrics=metrics)
        stage = (
            "holding" if stable else
            "releasing" if all(retained.values()) and order_ok else
            "routing"
        )
        return TaskResult(
            reward=float(sum(retained.values())) / len(retained),
            stage=stage,
            metrics=metrics,
        )
