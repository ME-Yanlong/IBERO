"""受限工业台架菜谱；旧线束 schema 不变，不接受未经支持的物理能力。"""

import math

from ibero.materials.parameters import (
    BeamParameters,
    LatchParameters,
    StockParameters,
    finite_number,
    strict_parameters,
)

SCHEMA = "ibero.industrial/v0.1"


def exact(mapping, keys, path):
    if not isinstance(mapping, dict) or set(mapping) != set(keys):
        raise ValueError(f"{path} requires exactly {sorted(keys)}")


def validate_industrial(config, constraints):
    robot_scene = config.get("kind") == "latch_release"
    exact(
        config,
        {
            "schema_version",
            "id",
            "kind",
            "backend",
            "physics",
            "materials",
            "mechanism",
            "numerics",
            "parameter_status",
        }
        | ({"robot", "initialization", "control"} if robot_scene else set()),
        "scene",
    )
    if config["schema_version"] != SCHEMA or config["backend"] != "mujoco":
        raise ValueError(
            "Industrial fixtures require the declared schema and mujoco backend"
        )
    if config["kind"] not in {
        "empty_bench",
        "latch_bench",
        "stock_bench",
        "latch_release",
    }:
        raise ValueError("Unsupported industrial scene kind")
    if not isinstance(config["id"], str) or not config["id"].strip():
        raise ValueError("id must be nonempty")
    if config["parameter_status"] != "provisional":
        raise ValueError(
            "Current fixture recipes are provisional; measured certification is not implemented"
        )
    physics = config["physics"]
    exact(physics, {"timestep_s", "control_hz", "gravity_m_s2"}, "physics")
    dt = finite_number(physics["timestep_s"], "timestep_s")
    hz = finite_number(physics["control_hz"], "control_hz")
    if (
        not isinstance(physics["gravity_m_s2"], list)
        or len(physics["gravity_m_s2"]) != 3
    ):
        raise ValueError("gravity_m_s2 must have three components")
    for value in physics["gravity_m_s2"]:
        finite_number(value, "gravity", minimum=-1000, positive=False)
    steps = 1 / hz / dt
    if not 1 <= steps <= 100000 or not math.isclose(steps, round(steps), abs_tol=1e-8):
        raise ValueError(
            "control period must be an integral bounded number of physics substeps"
        )
    exact(constraints, {"safety", "task"}, "constraints")
    exact(
        constraints["safety"],
        {"max_force_n", "max_deflection_m"}
        | (
            {"self_collision_penetration_m", "grip_penetration_m"}
            if robot_scene
            else set()
        ),
        "safety",
    )
    for key, value in constraints["safety"].items():
        finite_number(value, key)
    exact(
        constraints["task"],
        {"max_seconds"}
        | (
            {"withdrawal_distance_m", "hold_seconds", "pose_tolerance_rad"}
            if robot_scene
            else set()
        ),
        "task",
    )
    for key, value in constraints["task"].items():
        finite_number(value, key)
    kind = config["kind"]
    if kind in {"latch_bench", "latch_release"}:
        exact(config["materials"], {"beam"}, "materials")
        beam = strict_parameters(BeamParameters, config["materials"]["beam"])
        latch = strict_parameters(LatchParameters, config["mechanism"])
        exact(config["numerics"], {"segments"}, "numerics")
        n = config["numerics"]["segments"]
        if type(n) is not int or not 4 <= n <= 32:
            raise ValueError("segments must be an integer from 4 to 32")
        if dt > min(0.0002, beam.relaxation_time_s / 5):
            raise ValueError(
                "timestep exceeds the provisional damped beam screening bound"
            )
        if (
            latch.hook_length_m >= beam.length_m / 4
            or latch.overlap_m >= constraints["safety"]["max_deflection_m"]
        ):
            raise ValueError(
                "Latch geometry cannot clear inside the permitted deflection"
            )
    elif kind == "stock_bench":
        exact(config["materials"], {"stock"}, "materials")
        strict_parameters(StockParameters, config["materials"]["stock"])
        exact(config["mechanism"], set(), "mechanism")
        exact(config["numerics"], {"cell_size_m", "max_cells"}, "numerics")
        finite_number(config["numerics"]["cell_size_m"], "cell_size_m")
        if (
            type(config["numerics"]["max_cells"]) is not int
            or not 1 <= config["numerics"]["max_cells"] <= 2_000_000
        ):
            raise ValueError("max_cells must be an integer in [1, 2000000]")
    else:
        for key in ("materials", "mechanism", "numerics"):
            exact(config[key], set(), key)
    if robot_scene:
        validate_latch_robot(config, constraints)


def vector(value, n, name):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{name} must have {n} components")
    for item in value:
        finite_number(item, name, minimum=-1e6, positive=False)


def validate_latch_robot(config, constraints):
    robot = config["robot"]
    exact(
        robot,
        {
            "preset",
            "arm_qpos",
            "left_tool",
            "right_tool",
            "press_tcp_offset_m",
            "press_stem_height_m",
        },
        "robot",
    )
    if (robot["preset"], robot["left_tool"], robot["right_tool"]) != (
        "g1_upperbody_v0",
        "press_probe",
        "robotiq_2f85",
    ):
        raise ValueError("Only G1 left press probe / right 2F-85 is implemented")
    vector(robot["arm_qpos"], 14, "arm_qpos")
    vector(robot["press_tcp_offset_m"], 3, "press_tcp_offset_m")
    finite_number(robot["press_stem_height_m"], "press_stem_height_m")
    init = config["initialization"]
    exact(
        init,
        {"origin_m", "quaternion_wxyz", "origin_jitter_m", "friction_jitter"},
        "initialization",
    )
    for key in ("origin_m", "origin_jitter_m"):
        vector(init[key], 3, key)
    if any(v < 0 or v > 0.001 for v in init["origin_jitter_m"]):
        raise ValueError("Initial jitter is limited to [0, 1 mm] per axis")
    vector(init["quaternion_wxyz"], 4, "quaternion_wxyz")
    if not math.isclose(sum(v * v for v in init["quaternion_wxyz"]), 1, abs_tol=1e-8):
        raise ValueError("Unit quaternion required")
    finite_number(init["friction_jitter"], "friction_jitter", positive=False)
    if init["friction_jitter"] >= config["mechanism"]["friction"]:
        raise ValueError("Friction perturbation must remain positive")
    ctrl = config["control"]
    exact(
        ctrl,
        {
            "settle_seconds",
            "grasp_seconds",
            "press_speed_m_s",
            "pull_speed_m_s",
            "press_clearance_m",
            "max_press_travel_m",
            "max_pull_force_n",
            "grip_command",
        },
        "control",
    )
    for key, value in ctrl.items():
        finite_number(value, key, minimum=-1 if key == "grip_command" else 0)
    if (
        ctrl["grip_command"] > 1
        or ctrl["max_pull_force_n"] > constraints["safety"]["max_force_n"]
    ):
        raise ValueError("Controller cannot exceed declared force/grip bounds")
    if constraints["task"]["withdrawal_distance_m"] < 0.07:
        raise ValueError("Full withdrawal must clear the 65 mm guide keel")
