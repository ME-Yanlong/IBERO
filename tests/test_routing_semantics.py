"""布线判定反例：线段、唇边、顺序、松手与历史事件都要满足。"""

from dataclasses import replace

import numpy as np

from ibero.core.audit import AuditSnapshot
from ibero.core.task_api import TaskState
from ibero.tasks.cable_routing import CableRoutingTask, RoutingGoal
from ibero.tasks.routing_geometry import inspect_order, inspect_route, match_clip


def clip(name="a", y=0.0, **changes):
    value = {
        "id": name,
        "origin_m": [0, y, 0],
        "rotation_world": np.eye(3),
        "seat_z_min_m": 0.005,
        "seat_z_max_m": 0.020,
        "seat_half_width_m": 0.0056,
        "seat_half_length_m": 0.010,
        "lip_z_m": 0.025,
        "jaw_recovered": True,
        "jaw_gap_m": 0.00823,
        "wire_radius_m": 0.005,
        "max_penetration_m": 0.0,
        "penetration_limit_m": 0.002,
    }
    value.update(changes)
    return value


def line(z=0.008, *, x=0.0):
    return np.array([[x, -0.12, z], [x, 0.12, z]])


def state(points, clips, *, step=1, both=True, clear=False, speed=0.0, dt=0.1):
    return TaskState(
        body_positions={}, site_positions={}, gripper_openings={},
        grasps={"left_gripper": both, "right_gripper": both},
        contacts=frozenset(), cable_tension_n=0.4,
        elapsed_steps=step, max_episode_steps=100,
        extra={"routing": {
            "cable_points_m": points,
            "clips": clips,
            "dt_s": dt,
            "cable_speed_m_s": speed,
            "hands_clear": clear,
            "minimum_radius_m": 0.1,
        }},
    )


def audit(*, damage=False, collision=False):
    return AuditSnapshot(
        cable_damage=damage, self_collision=collision,
        cable_over_limit_seconds=0.0, peak_cable_tension_n=0.4,
        events=(),
    )


def test_clip_matching_uses_centerline_segments_and_correct_direction():
    # 无节点位于线夹内，仍必须识别穿过夹座的真实线段。
    match = match_clip(line(), clip())
    assert match.seated and match.arc_m == 0.12
    assert not match_clip(line(z=0.033), clip()).seated
    assert match_clip(line(z=0.033), clip()).mouth
    assert not match_clip(line(x=0.02), clip()).seated
    # 横切夹座的线段方向不符合侧向夹持的线束铺设。
    across = [[-0.01, 0, 0.008], [0.01, 0, 0.008]]
    assert not match_clip(across, clip()).seated


def test_order_requires_distinct_material_sections_and_path_monotonicity():
    first, second = clip("a", -0.05), clip("b", 0.05)
    matches = {item["id"]: match_clip(line(), item) for item in (first, second)}
    assert inspect_order(("a", "b"), matches, 0.03)
    assert not inspect_order(("b", "a"), matches, 0.03)
    deviation, monotone = inspect_route(
        line(), [[0, -0.05, 0.008], [0, 0.05, 0.008]],
        matches["a"].arc_m, matches["b"].arc_m,
    )
    assert deviation < 1e-12 and monotone


def test_one_clip_requires_entry_bimanual_grasp_and_hands_free_hold():
    goal = RoutingGoal(("a",), hold_time_s=0.2)
    task = CableRoutingTask(goal)
    # 入座前观察夹口外的线段，并确认双手真实保持。
    result = task.evaluate(state(line(0.033), [clip()]), audit())
    assert not result.success
    # 入座时夹爪仍托住，不可成功。
    result = task.evaluate(state(line(), [clip()], step=2), audit())
    assert result.metrics["a_retained"]
    assert not result.success
    # 第一个松手采样未满保持时间。
    released = replace(state(line(), [clip()], step=3, both=False, clear=True),
                       grasps={"left_gripper": False, "right_gripper": False})
    result = task.evaluate(released, audit())
    assert result.stage == "holding" and not result.success
    assert task.evaluate(replace(released, elapsed_steps=4), audit()).success


def test_preseated_and_never_grasped_are_not_complete_tasks():
    goal = RoutingGoal(("a",), hold_time_s=0.1)
    task = CableRoutingTask(goal)
    initial = replace(state(line(), [clip()], both=False, clear=True),
                      grasps={"left_gripper": False, "right_gripper": False})
    for step in range(1, 4):
        result = task.evaluate(replace(initial, elapsed_steps=step), audit())
    assert not result.success and not result.metrics["a_retained"]


def test_geometric_seat_is_not_retention_with_a_wide_jaw():
    task = CableRoutingTask(RoutingGoal(("a",), hold_time_s=0.1))
    task.evaluate(state(line(0.033), [clip()]), audit())
    released = replace(state(line(), [clip(jaw_gap_m=0.011)], step=2, both=False, clear=True),
                       grasps={"left_gripper": False, "right_gripper": False})
    result = task.evaluate(released, audit())
    assert match_clip(line(), clip()).seated
    assert not result.success and not result.metrics["a_retained"]
    assert result.metrics["hold_seconds"] == 0


def test_escape_clears_current_retention_and_resets_hold():
    task = CableRoutingTask(RoutingGoal(("a",), hold_time_s=0.3))
    task.evaluate(state(line(0.033), [clip()]), audit())
    task.evaluate(state(line(), [clip()], step=2), audit())
    released = replace(state(line(), [clip()], step=3, both=False, clear=True),
                       grasps={"left_gripper": False, "right_gripper": False})
    assert task.evaluate(released, audit()).metrics["hold_seconds"] == 0.1
    dropped = replace(released, elapsed_steps=4,
                      extra={"routing": {**released.extra["routing"],
                                         "cable_points_m": line(0.05)}})
    result = task.evaluate(dropped, audit())
    assert not result.metrics["a_retained"]
    assert result.metrics["hold_seconds"] == 0.0


def test_load_violation_invalid_snapshot_and_route_error_fail_closed():
    goal = RoutingGoal(("a",), hold_time_s=0.1)
    task = CableRoutingTask(goal)
    task.evaluate(state(line(0.033), [clip()]), audit())
    overloaded = task.evaluate(state(line(), [clip()], step=2), audit(damage=True))
    assert overloaded.terminated and not overloaded.success
    assert overloaded.failure_reason == "cable_damage"
    invalid = task.evaluate(state(line(), [clip(max_penetration_m=float("nan"))], step=3), audit())
    assert invalid.terminated and invalid.failure_reason.startswith("invalid:")


def test_multiclip_requires_corridor_even_when_both_clips_seated():
    goal = RoutingGoal(
        ("a", "b"),
        route_points_m=((0, -0.05, 0.008), (0, 0.05, 0.008)),
        corridor_m=0.01,
        hold_time_s=0.1,
        require_entry_history=False,
        require_two_hand_grasp=False,
    )
    clips = [clip("a", -0.05), clip("b", 0.05)]
    # 中心线中段离板路径很远；两个线夹都能碰到不能代替走线。
    curved = np.array([[0, -0.12, 0.008], [0, -0.05, 0.008],
                       [0.05, 0, 0.008], [0, 0.05, 0.008], [0, 0.12, 0.008]])
    released = replace(state(curved, clips, both=False, clear=True),
                       grasps={"left_gripper": False, "right_gripper": False})
    result = CableRoutingTask(goal).evaluate(released, audit())
    assert result.metrics["retained_count"] == 2
    assert result.metrics["path_error_m"] > goal.corridor_m
    assert not result.success
