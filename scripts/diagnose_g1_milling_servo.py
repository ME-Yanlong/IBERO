"""S8 动态控制独立诊断：真实重力、执行器与外载，不去除材料、不冒称 G8。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import yaml
from ibero.core.scene_loader import SceneLoader
from ibero.core.reproducibility import simulation_source_hash
from ibero.core.loads import PhysicalLoads
from ibero.materials.parameters import strict_parameters
from ibero.processes.tools import EndMillGeometry
from ibero.processes.milling_forces import MillingLimits
from ibero.robots.g1_milling import milling_robot_spec
from ibero.robots.g1_industrial import solve_reset_pose
from ibero.robots.g1_upperbody import ARM_JOINTS
from ibero.control.milling import MillingArmServo

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--force", type=float, nargs=3, default=[0, 0, 8])
    p.add_argument("--kp", type=float, default=50000.0)
    p.add_argument("--kd", type=float, default=500.0)
    args = p.parse_args()
    out = (
        ROOT
        / "artifacts/industrial_core01/stock/g1_servo"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    out.mkdir(parents=True, exist_ok=False)
    report = dict(
        scope="S8_dynamic_servo_not_cutting",
        source_hash=simulation_source_hash(),
        parameters=vars(args),
        passed=False,
        rows=[],
    )
    started = time.perf_counter()
    try:
        cfg = SceneLoader().validate(ROOT / "scenes/milling_bench").config
        spec = milling_robot_spec(
            strict_parameters(EndMillGeometry, cfg["tool"]),
            strict_parameters(MillingLimits, cfg["process"]["limits"]),
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        initial = yaml.safe_load(
            (ROOT / "scenes/latch_release/scene_config.yaml").read_text(
                encoding="utf-8"
            )
        )["robot"]["arm_qpos"]
        initial[7:] = [-0.65, -0.35, 0, 1.1, 0, 0, 0]
        for n, q in zip(ARM_JOINTS, initial):
            data.joint(n).qpos[0] = q
        origin = np.array([0.45, 0.16, 0.8535])
        solve_reset_pose(model, data, "left", "mill_tip", origin, np.eye(3))
        mujoco.mj_forward(model, data)
        servo = MillingArmServo(model, data, position_kp=args.kp, position_kd=args.kd)
        loads = PhysicalLoads(model)
        for step in range(round(args.seconds / model.opt.timestep)):
            # 先落稳，再在 0.5 s 内平滑加载，最后以 1 mm/s 跟踪小范围直线。
            target = origin + [min(max(data.time - 2, 0) * 0.001, 0.004), 0, 0]
            if step % 10 == 0:
                metrics = servo.apply(data, target)
            loads.begin(data)
            ramp = np.clip((data.time - 1) / 0.5, 0, 1)
            loads.add_wrench(
                data,
                model.body("mill_rotor").id,
                ramp * np.array(args.force),
                [0, 0, -0.04 * ramp],
                data.site("mill_tip").xpos,
            )
            loads.commit(data)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if step % 100 == 0:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    model, data, mujoco.mjtObj.mjOBJ_SITE, servo.site, velocity, 0
                )
                report["rows"].append(
                    dict(
                        time_s=float(data.time),
                        **metrics,
                        actual_joint_limit_fraction=float(
                            np.max(
                                np.abs(data.qfrc_actuator[servo.dofs]) / servo.limits
                            )
                        ),
                        holder_angular_velocity_rad_s=float(
                            np.linalg.norm(velocity[:3])
                        ),
                        tip_position_m=data.site("mill_tip").xpos.copy().tolist(),
                    )
                )
            if not np.isfinite(data.qpos).all():
                raise ValueError("Nonfinite robot state")
        report["max_tip_error_m"] = max(r["tip_error_m"] for r in report["rows"])
        report["max_orientation_error_rad"] = max(
            r["orientation_error_rad"] for r in report["rows"]
        )
        report["max_requested_fraction"] = max(
            r["requested_joint_limit_fraction"] for r in report["rows"]
        )
        report["max_angular_velocity_rad_s"] = max(
            r["holder_angular_velocity_rad_s"] for r in report["rows"]
        )
        report["passed"] = bool(
            report["max_tip_error_m"] < 0.0005
            and report["max_orientation_error_rad"] < 0.01
            and report["max_requested_fraction"] < 1
        )
    except Exception as error:
        report["exception"] = f"{type(error).__name__}: {error}"
    report["wall_seconds"] = time.perf_counter() - started
    simulation_source_hash.cache_clear()
    report["frozen_source"] = report["source_hash"] == simulation_source_hash()
    report["passed"] &= report["frozen_source"]
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        out,
        {k: v for k, v in report.items() if k not in {"rows", "parameters"}},
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
