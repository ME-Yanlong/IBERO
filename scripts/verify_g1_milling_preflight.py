"""S8-01 静态可达性/原能力预检；不以此声明已完成机器人加工。"""

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import yaml
from PIL import Image
from ibero.core.reproducibility import simulation_source_hash
from ibero.core.scene_loader import SceneLoader
from ibero.materials.parameters import StockParameters, strict_parameters
from ibero.materials.stock import VoxelStock
from ibero.materials.stock_collision import add_stock_geoms, StockCollisionBinding
from ibero.processes.tools import EndMillGeometry
from ibero.processes.milling_forces import MillingLimits
from ibero.robots.g1_upperbody import ARM_JOINTS
from ibero.robots.g1_industrial import solve_reset_pose
from ibero.robots.g1_milling import (
    milling_robot_spec,
    add_plate_support,
    milling_robot_handles,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", type=float, nargs=3, default=[0.45, 0.16, 0.85])
    args = parser.parse_args()
    out = (
        ROOT
        / "artifacts/industrial_core01/stock/g1_preflight"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "S8_static_adaptation_not_actual_cutting",
        "source_hash": simulation_source_hash(),
        "passed": False,
        "poses": [],
    }
    try:
        cfg = SceneLoader().validate(ROOT / "scenes/milling_bench").config
        tool = strict_parameters(EndMillGeometry, cfg["tool"])
        limits = strict_parameters(MillingLimits, cfg["process"]["limits"])
        origin = np.array(args.origin)
        report["stock_origin_m"] = origin.tolist()
        stock = VoxelStock(
            strict_parameters(StockParameters, cfg["materials"]["stock"]),
            0.001,
            origin=origin,
        )
        spec = milling_robot_spec(tool, limits, timestep=0.0001)
        add_stock_geoms(spec, stock, contype=8)
        add_plate_support(
            spec, origin=origin, half_size=np.array(stock.params.size_m) / 2
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        StockCollisionBinding(model, stock)
        handles = milling_robot_handles(model)
        initial = yaml.safe_load(
            (ROOT / "scenes/latch_release/scene_config.yaml").read_text(
                encoding="utf-8"
            )
        )["robot"]["arm_qpos"]
        initial[7:] = [-0.65, -0.35, 0, 1.1, 0, 0, 0]
        for name, value in zip(ARM_JOINTS, initial):
            data.joint(name).qpos[0] = value
        mujoco.mj_forward(model, data)
        report["right_safe_tcp_m"] = data.site("right_pinch").xpos.copy().tolist()
        joints = np.array([model.joint(n).id for n in ARM_JOINTS])
        dofs = model.jnt_dofadr[joints]
        limits_nm = np.max(np.abs(model.jnt_actfrcrange[joints]), axis=1)
        report["original_joint_torque_limits_nm"] = dict(
            zip(ARM_JOINTS, limits_nm.tolist())
        )
        # 预检覆盖预计约 14 N 工况，采用 15 N 包络；后续动态环境仍逐步检查实际关节限值。
        cases = [("gravity", np.zeros(3), np.zeros(3))] + [
            (f"force_{i}_{sign}", np.eye(3)[i] * 15 * sign, np.array([0, 0, -0.15]))
            for i in range(3)
            for sign in (-1, 1)
        ]
        for label, offset in (
            ("safe", [0, 0, 0.0035]),
            ("hole_bottom", [0, 0, -0.0025]),
            ("slot_left", [-0.004, 0, 0]),
            ("slot_right", [0.004, 0, 0]),
            ("pocket_ne", [0.004, 0.002, 0]),
            ("pocket_sw", [-0.004, -0.002, 0]),
        ):
            residual = solve_reset_pose(
                model, data, "left", "mill_tip", origin + offset, np.eye(3)
            )
            # 这里是 reset 级静态分析；加工运行不能借用 IK 直接覆盖 qpos。
            mujoco.mj_forward(model, data)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, handles.left_anchor_site_id)
            loads = []
            for name, force, torque in cases:
                required = data.qfrc_bias[dofs] - (
                    jp[:, dofs].T @ force + jr[:, dofs].T @ torque
                )
                loads.append(
                    {
                        "case": name,
                        "required_joint_nm": required.tolist(),
                        "max_limit_fraction": float(
                            np.max(np.abs(required) / limits_nm)
                        ),
                    }
                )
            # 六轴向示例不足以包络任意力方向。以下是逐关节保守范数上界，
            # 同时包含工具力臂和最大主轴扭矩，不能把有限样例冒称整个球形载荷域。
            moment_bound = limits.max_torque_nm + 15 * (
                tool.radius_m + tool.cutting_length_m
            )
            upper = (
                np.abs(data.qfrc_bias[dofs])
                + 15 * np.linalg.norm(jp[:, dofs], axis=0)
                + moment_bound * np.linalg.norm(jr[:, dofs], axis=0)
            )
            loads.append(
                {
                    "case": "all_directions_conservative_norm_bound",
                    "required_joint_absolute_upper_nm": upper.tolist(),
                    "force_norm_bound_n": 15.0,
                    "moment_norm_bound_nm": moment_bound,
                    "max_limit_fraction": float(np.max(upper / limits_nm)),
                }
            )
            # 静态刀尖位于毛坯内的刀刃接触是预期；非工作面、自碰撞不能被忽略。
            blade = model.geom("mill_blade").id
            stock_geom_ids = {
                model.geom(f"stock_cell_{i}").id for i in range(len(stock.centers))
            }
            forbidden = [
                {
                    "pair": [model.geom(c.geom1).name, model.geom(c.geom2).name],
                    "penetration_m": -float(c.dist),
                }
                for c in data.contact
                if not (
                    (c.geom1 == blade and c.geom2 in stock_geom_ids)
                    or (c.geom2 == blade and c.geom1 in stock_geom_ids)
                )
                and c.dist < -0.0001
            ]
            row = {
                "pose": label,
                "target_m": (origin + offset).tolist(),
                "ik_residual": residual,
                "loads": loads,
                "forbidden_contacts": forbidden,
                "arm_qpos": data.qpos[model.jnt_qposadr[joints]].tolist(),
            }
            row["passed"] = (
                not forbidden and max(r["max_limit_fraction"] for r in loads) < 1
            )
            report["poses"].append(row)
            print(
                label,
                "passed",
                row["passed"],
                "worst torque fraction",
                max(r["max_limit_fraction"] for r in loads),
                "contacts",
                forbidden,
                flush=True,
            )
        solve_reset_pose(
            model, data, "left", "mill_tip", origin + [0, 0, 0.0035], np.eye(3)
        )
        mujoco.mj_forward(model, data)
        report["ready_arm_qpos"] = data.qpos[model.jnt_qposadr[joints]].tolist()
        renderer = mujoco.Renderer(
            model, height=480, width=640, max_geom=model.ngeom + 100
        )
        try:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.3, 0, 0.8]
            camera.distance, camera.azimuth, camera.elevation = 1.1, 135, -25
            renderer.update_scene(data, camera)
            Image.fromarray(renderer.render()).save(out / "installation.png")
        finally:
            renderer.close()
        report["passed"] = len(report["poses"]) == 6 and all(
            r["passed"] for r in report["poses"]
        )
    except Exception as error:
        report["exception"] = f"{type(error).__name__}: {error}"
    simulation_source_hash.cache_clear()
    report["frozen_source"] = report["source_hash"] == simulation_source_hash()
    report["passed"] &= report["frozen_source"]
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(out, "passed", report["passed"], report.get("exception"), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
