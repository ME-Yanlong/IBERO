"""Core-0.2 布线菜谱的窄范围校验；旧场景 schema 保持独立。"""

from __future__ import annotations

import math
from collections.abc import Mapping


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _vector(value, label, n=3):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{label} must have {n} components")
    return [_number(x, f"{label}[{i}]") for i, x in enumerate(value)]


def validate_routing(config, constraints):
    """拒绝未声明物性、错位目标与无效长度；动态可达性另在台架检验。"""
    allowed = {"schema_version", "id", "kind", "suite", "backend", "robot", "physics", "materials", "mechanism", "workcell", "initialization", "routing_goal", "render"}
    if set(config) != allowed:
        raise ValueError(f"Routing keys must be exactly {sorted(allowed)}")
    if config["schema_version"] != "ibero.routing/v0.2" or config["backend"] != "mujoco":
        raise ValueError("Routing requires ibero.routing/v0.2 and MuJoCo")
    if config["kind"] not in {"cable_clip_insert", "cable_route_straight", "cable_route_l", "cable_route_rectangle", "cable_clip_bench"}:
        raise ValueError("Unsupported routing kind")
    # 场景 ID 是贡献者可自定义的目录名；kind 只选已有物理/策略模板。
    if config["suite"] != "harness" or not isinstance(config["id"], str) or not config["id"]:
        raise ValueError("Routing id/suite mismatch")
    robot = _mapping(config["robot"], "robot")
    if robot != {"preset": "g1_upperbody_v0", "end_effector": "robotiq_2f85"}:
        raise ValueError("Routing currently requires G1 with 2F-85")
    physics = _mapping(config["physics"], "physics")
    if set(physics) != {"timestep_s", "control_hz", "max_episode_steps"}:
        raise ValueError("Routing physics fields mismatch")
    dt = _number(physics["timestep_s"], "physics.timestep_s", positive=True)
    hz = _number(physics["control_hz"], "physics.control_hz", positive=True)
    substeps = 1 / (dt * hz)
    if abs(substeps - round(substeps)) > 1e-8:
        raise ValueError("control period must contain an integer number of physics steps")
    if type(physics["max_episode_steps"]) is not int or physics["max_episode_steps"] <= 0:
        raise ValueError("max_episode_steps must be positive integer")
    cable = _mapping(_mapping(config["materials"], "materials")["cable"], "materials.cable")
    expected_cable = {"length_m", "outer_diameter_m", "linear_density_kg_m", "minimum_bend_radius_m", "axial_stiffness_n_m", "bending_stiffness_nm2", "damping", "friction", "segments", "parameter_source"}
    if set(cable) != expected_cable:
        raise ValueError("Routing cable parameter fields mismatch")
    for key in expected_cable - {"segments", "parameter_source"}:
        _number(cable[key], f"cable.{key}", positive=True)
    if type(cable["segments"]) is not int or not 8 <= cable["segments"] <= 64:
        raise ValueError("Routing cable segments must be in [8,64]")
    if cable["outer_diameter_m"] >= cable["length_m"] / cable["segments"]:
        raise ValueError("Cable diameter must be smaller than each flex edge")
    if cable["parameter_source"] != "provisional":
        raise ValueError("Default routing cable needs explicit provisional provenance")
    mechanism = _mapping(config["mechanism"], "mechanism")
    for key in ("wire_radius_m", "seat_half_width_m", "mouth_half_width_m", "seat_depth_m", "jaw_length_m", "jaw_width_m", "jaw_thickness_m", "young_pa", "density_kg_m3", "relaxation_time_s", "lip_depth_m", "friction"):
        _number(mechanism[key], f"mechanism.{key}", positive=True)
    if mechanism["parameter_source"] != "provisional":
        raise ValueError("Clip parameter provenance must be provisional")
    if abs(mechanism["wire_radius_m"] - cable["outer_diameter_m"] / 2) > 1e-9:
        raise ValueError("Clip wire radius must equal cable radius")
    if not mechanism["mouth_half_width_m"] < mechanism["wire_radius_m"] < mechanism["seat_half_width_m"]:
        raise ValueError("Clip mouth/seat must straddle cable radius")
    workcell = _mapping(config["workcell"], "workcell")
    if not {"board", "clips"}.issubset(workcell) or set(workcell) - {
        "board", "clips", "guide_tool", "feed_support_y_m", "feed_support_half_y_m"
    }:
        raise ValueError("Routing workcell requires board/clips and known optional fields")
    if "guide_tool" in workcell and workcell["guide_tool"] != "right_offset_nose_v1":
        raise ValueError("Unsupported routing guide tool")
    if "feed_support_y_m" in workcell:
        feed_y = _vector(workcell["feed_support_y_m"], "workcell.feed_support_y_m")
        if any(abs(value) > 0.25 for value in feed_y) or not feed_y[0] > feed_y[1] > feed_y[2]:
            raise ValueError("Feed supports must be ordered within the board width")
    if "feed_support_half_y_m" in workcell:
        widths = _vector(workcell["feed_support_half_y_m"], "workcell.feed_support_half_y_m")
        if any(not 0.005 <= value <= 0.05 for value in widths):
            raise ValueError("Feed support half widths must be within [5, 50] mm")
    board = _mapping(workcell["board"], "board")
    _vector(board["center_m"], "board.center_m")
    if any(value <= 0 for value in _vector(board["half_size_m"], "board.half_size_m")):
        raise ValueError("Board half sizes must be positive")
    clips = workcell["clips"]
    if not isinstance(clips, list) or not clips:
        raise ValueError("At least one clip is required")
    ids = []
    for i, clip in enumerate(clips):
        if not isinstance(clip, Mapping) or set(clip) not in ({"id", "origin_m"}, {"id", "origin_m", "tangent"}):
            raise ValueError(f"clips[{i}] needs id/origin_m and optional tangent")
        if not isinstance(clip["id"], str) or not clip["id"].strip():
            raise ValueError(f"clips[{i}].id must be a nonempty string")
        ids.append(clip["id"])
        _vector(clip["origin_m"], f"clips[{i}].origin_m")
        if clip.get("tangent", "horizontal") not in {"horizontal", "vertical"}:
            raise ValueError(f"clips[{i}].tangent must be horizontal or vertical")
    if len(set(ids)) != len(ids):
        raise ValueError("Clip IDs must be unique")
    # 同向线夹的固定背座是实体碰撞盒。若两盒互穿，指垫会被下一只
    # 夹座挡住，物理回合尚未开始就已不存在可行的压入通道。
    base_half_along = mechanism["jaw_width_m"] / 2 + 0.003
    base_half_across = (
        mechanism["seat_half_width_m"] + mechanism["jaw_thickness_m"] + 0.004
    )
    for i, first in enumerate(clips):
        for second in clips[i + 1:]:
            tangent = first.get("tangent", "horizontal")
            if tangent != second.get("tangent", "horizontal"):
                continue
            longitudinal = 1 if tangent == "horizontal" else 2
            transverse = 2 if tangent == "horizontal" else 1
            a, b = first["origin_m"], second["origin_m"]
            if (abs(a[longitudinal] - b[longitudinal]) < 2 * base_half_along
                    and abs(a[transverse] - b[transverse]) < 2 * base_half_across):
                raise ValueError(f"Clip bases overlap: {first['id']} and {second['id']}")
    init = _mapping(config["initialization"], "initialization")
    if set(init) not in ({"cable_start_m", "cable_end_m"}, {"cable_start_m", "cable_end_m", "cable_translation_jitter_m"}):
        raise ValueError("Routing initialization fields mismatch")
    a = _vector(init["cable_start_m"], "initialization.cable_start_m")
    b = _vector(init["cable_end_m"], "initialization.cable_end_m")
    if "cable_translation_jitter_m" in init:
        jitter = _vector(init["cable_translation_jitter_m"], "initialization.cable_translation_jitter_m")
        if any(value < 0 or value > 0.003 for value in jitter):
            raise ValueError("Cable initial jitter must be within [0, 3] mm per axis")
    if math.dist(a, b) >= cable["length_m"]:
        raise ValueError("Cable initial span must be shorter than its length")
    goal = _mapping(config["routing_goal"], "routing_goal")
    if goal["ordered_clips"] != ids:
        raise ValueError("Clip material order must match listed unique clips")
    route = goal["route_points_m"]
    if not isinstance(route, list) or len(route) < 2:
        raise ValueError("Route needs at least two points")
    for i, point in enumerate(route):
        _vector(point, f"route_points_m[{i}]")
    if sum(math.dist(x, y) for x, y in zip(route, route[1:])) >= cable["length_m"]:
        raise ValueError("Cable length insufficient for target route")
    for key in ("corridor_m", "hold_time_s", "speed_limit_m_s"):
        _number(goal[key], f"routing_goal.{key}", positive=True)
    if ("free_end_center_m" in goal) != ("free_end_radius_m" in goal):
        raise ValueError("Free-end center and radius must be specified together")
    if "free_end_center_m" in goal:
        _vector(goal["free_end_center_m"], "routing_goal.free_end_center_m")
        _number(goal["free_end_radius_m"], "routing_goal.free_end_radius_m", positive=True)
    if "free_end_index" in goal and (
        type(goal["free_end_index"]) is not int or goal["free_end_index"] not in (0, -1)
    ):
        raise ValueError("routing_goal.free_end_index must be first or last cable vertex")
    limits = _mapping(constraints, "constraints")
    if set(limits) != {"cable", "clip", "collision"}:
        raise ValueError("Routing constraint keys mismatch")
    _number(limits["cable"]["tension_limit_n"], "tension_limit_n", positive=True)
    _number(limits["cable"]["limit_duration_s"], "limit_duration_s", positive=True)
    _number(limits["clip"]["penetration_limit_m"], "penetration_limit_m", positive=True)
    return config
