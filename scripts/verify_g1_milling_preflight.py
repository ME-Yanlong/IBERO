"""S8-01 静态可达性/原能力预检；不以此声明已完成机器人加工。"""

from datetime import datetime, timezone
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from ibero.core.reproducibility import simulation_source_hash
from ibero.core.scene_loader import SceneLoader
from ibero.core.task_loading import load_task_spec
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
    parser.add_argument("--scene", type=Path, default=ROOT / "scenes/plate_milling")
    parser.add_argument("--origin", type=float, nargs=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = (
        args.output
        or ROOT
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
        scene = SceneLoader().validate(args.scene)
        cfg = scene.config
        if cfg["kind"] != "plate_milling":
            raise ValueError("Static G1 preflight requires a plate_milling recipe")
        report["scene_config"] = cfg
        report["scene_hash"] = scene.scene_hash
        tool = strict_parameters(EndMillGeometry, cfg["tool"])
        limits = strict_parameters(MillingLimits, cfg["process"]["limits"])
        origin = np.array(
            cfg["workcell"]["stock_origin_m"] if args.origin is None else args.origin
        )
        report["stock_origin_m"] = origin.tolist()
        stock = VoxelStock(
            strict_parameters(StockParameters, cfg["materials"]["stock"]),
            cfg["numerics"]["cell_size_m"],
            origin=origin,
            max_cells=cfg["numerics"]["max_cells"],
        )
        spec = milling_robot_spec(
            tool,
            limits,
            timestep=cfg["physics"]["timestep_s"],
            tip_offset_m=cfg["robot"]["tip_offset_m"],
            bracket_mass_kg=cfg["robot"]["bracket_mass_kg"],
        )
        spec.option.gravity = cfg["physics"]["gravity_m_s2"]
        add_stock_geoms(spec, stock, contype=8)
        add_plate_support(
            spec, origin=origin, half_size=np.array(stock.params.size_m) / 2
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        binding = StockCollisionBinding(model, stock)
        handles = milling_robot_handles(model)
        initial = cfg["robot"]["arm_qpos"]
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
        # 包络来自实际工作站菜谱，而不是另外复制一份台架参数；动态仍逐步检查限值。
        force_bound = limits.max_force_n
        cases = [("gravity", np.zeros(3), np.zeros(3))] + [
            (
                f"force_{i}_{sign}",
                np.eye(3)[i] * force_bound * sign,
                np.array([0, 0, -limits.max_torque_nm]),
            )
            for i in range(3)
            for sign in (-1, 1)
        ]
        task = load_task_spec(scene)
        targets = {
            name: task.make_target(name) for name in ("through_hole", "slot", "pocket")
        }
        report["targets"] = {name: asdict(t) for name, t in targets.items()}
        poses = [("safe", cfg["initialization"]["tip_position_m"])]
        for shape, signs in (
            ("through_hole", [0]),
            ("slot", [-1, 1]),
            ("pocket", [1, -1]),
        ):
            t = targets[shape]
            for sign in signs:
                offset = [
                    t.center_xy_m[0] + sign * t.half_straight_xy_m[0],
                    t.center_xy_m[1] + sign * t.half_straight_xy_m[1],
                    stock.params.size_m[2] / 2
                    - t.depth_m
                    - (stock.cell_size_m / 2 if shape == "through_hole" else 0),
                ]
                poses.append((f"{shape}_{sign}", offset))
        for label, offset in poses:
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
            moment_bound = limits.max_torque_nm + force_bound * (
                tool.radius_m + tool.cutting_length_m
            )
            upper = (
                np.abs(data.qfrc_bias[dofs])
                + force_bound * np.linalg.norm(jp[:, dofs], axis=0)
                + moment_bound * np.linalg.norm(jr[:, dofs], axis=0)
            )
            loads.append(
                {
                    "case": "all_directions_conservative_norm_bound",
                    "required_joint_absolute_upper_nm": upper.tolist(),
                    "force_norm_bound_n": force_bound,
                    "moment_norm_bound_nm": moment_bound,
                    "max_limit_fraction": float(np.max(upper / limits_nm)),
                }
            )
            # 静态刀尖位于毛坯内的刀刃接触是预期；非工作面、自碰撞不能被忽略。
            blade = model.geom("mill_blade").id
            stock_geom_ids = set(int(i) for i in binding.geom_ids)
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
            model,
            data,
            "left",
            "mill_tip",
            origin + cfg["initialization"]["tip_position_m"],
            np.eye(3),
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
