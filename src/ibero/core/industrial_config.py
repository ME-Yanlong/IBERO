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
        },
        "scene",
    )
    if config["schema_version"] != SCHEMA or config["backend"] != "mujoco":
        raise ValueError(
            "Industrial fixtures require the declared schema and mujoco backend"
        )
    if config["kind"] not in {"empty_bench", "latch_bench", "stock_bench"}:
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
    exact(constraints["safety"], {"max_force_n", "max_deflection_m"}, "safety")
    for key, value in constraints["safety"].items():
        finite_number(value, key)
    exact(constraints["task"], {"max_seconds"}, "task")
    finite_number(constraints["task"]["max_seconds"], "max_seconds")
    kind = config["kind"]
    if kind == "latch_bench":
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
