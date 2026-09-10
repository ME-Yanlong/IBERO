"""受限加工台架菜谱：材料、刀具、系数、能力与控制参数显式分离。"""

import math
from ibero.core.industrial_config import exact, vector, SCHEMA
from ibero.materials.parameters import StockParameters, strict_parameters, finite_number
from ibero.processes.tools import EndMillGeometry
from ibero.processes.milling_forces import MillingCoefficients, MillingLimits


def validate_milling_bench(cfg, constraints):
    exact(
        cfg,
        {
            "schema_version",
            "id",
            "kind",
            "backend",
            "parameter_status",
            "physics",
            "materials",
            "numerics",
            "tool",
            "process",
            "initialization",
            "control",
        },
        "milling scene",
    )
    if (
        (cfg["schema_version"], cfg["backend"], cfg["parameter_status"], cfg["kind"])
        != (SCHEMA, "mujoco", "provisional", "milling_bench")
        or not isinstance(cfg["id"], str)
        or not cfg["id"].strip()
    ):
        raise ValueError("Explicit provisional MuJoCo milling bench required")
    p = cfg["physics"]
    exact(p, {"timestep_s", "control_hz", "gravity_m_s2"}, "physics")
    dt, hz = (finite_number(p[k], k) for k in ("timestep_s", "control_hz"))
    vector(p["gravity_m_s2"], 3, "gravity_m_s2")
    # 台架用于过程辨识，不把零重力设置悄悄带入后续机器人场景。
    if p["gravity_m_s2"] != [0, 0, 0]:
        raise ValueError("Identification bench requires explicitly zero gravity")
    if not 1 <= 1 / hz / dt <= 5000 or not math.isclose(
        1 / hz / dt, round(1 / hz / dt), abs_tol=1e-8
    ):
        raise ValueError("Bounded integral milling substeps required")
    exact(cfg["materials"], {"stock"}, "materials")
    stock = strict_parameters(StockParameters, cfg["materials"]["stock"])
    tool = strict_parameters(EndMillGeometry, cfg["tool"])
    exact(cfg["process"], {"coefficients", "limits"}, "process")
    coeff = strict_parameters(MillingCoefficients, cfg["process"]["coefficients"])
    limits = strict_parameters(MillingLimits, cfg["process"]["limits"])
    if stock.material_grade != coeff.material_grade:
        raise ValueError("Stock and cutting coefficient grades must agree")
    if limits.max_axial_depth_m > tool.cutting_length_m:
        raise ValueError("Declared axial depth exceeds working flute length")
    exact(cfg["numerics"], {"cell_size_m", "max_cells", "angular_samples"}, "numerics")
    n = cfg["numerics"]
    cell = finite_number(n["cell_size_m"], "cell_size_m")
    if type(n["max_cells"]) is not int or not 1 <= n["max_cells"] <= 60000:
        raise ValueError(
            "Physical stock capacity is 60000 cells, not geometry-only capacity"
        )
    count = math.prod(math.ceil(s / cell) for s in stock.size_m)
    if count > n["max_cells"]:
        raise ValueError("Stock resolution exceeds configured capacity")
    if type(n["angular_samples"]) is not int or not 32 <= n["angular_samples"] <= 512:
        raise ValueError("Bounded engagement quadrature required")
    exact(cfg["initialization"], {"tip_position_m"}, "initialization")
    vector(cfg["initialization"]["tip_position_m"], 3, "tip_position_m")
    c = cfg["control"]
    exact(
        c,
        {"rpm", "feed_m_s", "plunge_m_s", "settle_seconds", "tracking_tolerance_m"},
        "control",
    )
    for key, value in c.items():
        finite_number(value, key)
    if not limits.min_rpm <= c["rpm"] <= limits.max_rpm:
        raise ValueError("Command rpm is outside process domain")
    if (
        max(c["feed_m_s"], c["plunge_m_s"]) / (c["rpm"] / 60 * limits.teeth)
        > limits.max_feed_per_tooth_m
    ):
        raise ValueError("Command feed is outside declared per-tooth range")
    exact(constraints, {"safety", "task"}, "constraints")
    exact(
        constraints["safety"],
        {"max_tracking_error_m", "max_shank_penetration_m"},
        "safety",
    )
    exact(
        constraints["task"],
        {"max_seconds", "volume_error_fraction", "boundary_error_cells"},
        "task",
    )
    for section in constraints.values():
        for key, value in section.items():
            finite_number(value, key)
    if (
        constraints["task"]["volume_error_fraction"] > 0.05
        or constraints["task"]["boundary_error_cells"] > 2
    ):
        raise ValueError("Published shape criteria cannot relax frozen gates")
