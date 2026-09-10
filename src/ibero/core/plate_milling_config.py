"""固定钢板/G1 菜谱：复用材料与工艺严格校验，单列机器人能力和工作站。"""

import copy
import math
from ibero.core.industrial_config import exact, vector
from ibero.core.milling_config import validate_milling_bench
from ibero.materials.parameters import finite_number


def validate_plate_milling(cfg, constraints):
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
            "robot",
            "workcell",
        },
        "plate milling",
    )
    exact(cfg["workcell"], {"stock_origin_m"}, "workcell")
    vector(cfg["workcell"]["stock_origin_m"], 3, "stock_origin_m")
    exact(
        cfg["robot"],
        {
            "arm_qpos",
            "tip_offset_m",
            "bracket_mass_kg",
            "servo_hz",
            "position_kp_n_m",
            "position_kd_ns_m",
            "rotation_kp_nm_rad",
            "rotation_kd_nms_rad",
            "axis_tolerance_rad",
            "holder_angular_limit_rad_s",
        },
        "robot",
    )
    r = cfg["robot"]
    vector(r["arm_qpos"], 14, "arm_qpos")
    vector(r["tip_offset_m"], 3, "tip_offset_m")
    for k, v in r.items():
        if k not in {"arm_qpos", "tip_offset_m"}:
            finite_number(v, k)
    if (
        not math.isclose(
            1 / r["servo_hz"] / cfg["physics"]["timestep_s"],
            round(1 / r["servo_hz"] / cfg["physics"]["timestep_s"]),
            abs_tol=1e-8,
        )
        or not cfg["physics"]["control_hz"]
        <= r["servo_hz"]
        <= 1 / cfg["physics"]["timestep_s"]
    ):
        raise ValueError("Robot servo must use bounded integral physics substeps")
    if r["axis_tolerance_rad"] > 0.01 or r["holder_angular_limit_rad_s"] > 0.1:
        raise ValueError("Not a supported fixed-axis tracking envelope")
    if (cfg["tool"]["cutting_length_m"] + cfg["tool"]["radius_m"]) * math.sin(
        r["axis_tolerance_rad"]
    ) > cfg["numerics"]["cell_size_m"] / 4:
        raise ValueError("Axis projection error exceeds quarter voxel")
    exact(
        cfg["initialization"], {"tip_position_m", "joint_jitter_rad"}, "initialization"
    )
    finite_number(
        cfg["initialization"]["joint_jitter_rad"], "joint_jitter_rad", positive=False
    )
    if not 0 <= cfg["initialization"]["joint_jitter_rad"] <= 0.0001:
        raise ValueError("Robot reset jitter outside checked small envelope")
    if cfg["physics"]["gravity_m_s2"] != [0, 0, -9.81]:
        raise ValueError("Robot machining requires declared Earth gravity")
    # 只投影公共字段做同一校验；机床参数不进入机器人模型，也不提高原 G1 能力。
    shared = copy.deepcopy(cfg)
    shared.pop("robot")
    shared.pop("workcell")
    shared["kind"] = "milling_bench"
    shared["physics"]["gravity_m_s2"] = [0, 0, 0]
    shared["initialization"].pop("joint_jitter_rad")
    shared["machine"] = dict(
        axis_stiffness_n_m=1.0, axis_damping_ns_m=1.0, axis_force_limit_n=1.0
    )
    validate_milling_bench(shared, constraints)
