"""Core-0.1 物理、控制和故障验收。失败时返回非零退出码并保存完整证据。"""

from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

import mujoco
import numpy as np
import yaml

import ibero
from ibero.baselines import CableHandoverScript
from ibero.control.tension import CableStretchScript
from ibero.core.scene_loader import SceneLoader
from ibero.core.reproducibility import simulation_source_hash
from ibero.envs.cable_handover import DEFAULT_SCENE_PATH
from ibero.materials.cable import CableParameters
from ibero.materials.bench import axial_fixture, bending_gradient_check, sag_fixture


def sensor_fixture():
    """六方向已知外载：验证 [力,力矩] 排序、符号、旋转坐标和力臂。"""
    from ibero.core.sensors import wrist_wrench

    xml = """<mujoco><option gravity="0 0 0"/><worldbody><body name="wrist">
    <site name="ft" quat="0.9238795325 0 0.3826834324 0"/>
    <body name="tool" pos="0.1 0.02 0"><geom type="sphere" size=".02" mass="1"/></body>
    </body></worldbody><sensor><force name="force" site="ft"/><torque name="torque" site="ft"/></sensor></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    records = []
    for axis in range(6):
        mujoco.mj_resetData(model, data)
        load = np.zeros(6)
        load[axis] = 5 if axis < 3 else 0.2
        data.xfrc_applied[model.body("tool").id] = load
        mujoco.mj_forward(model, data)
        measured = wrist_wrench(
            model,
            data,
            force_sensor_id=model.sensor("force").id,
            torque_sensor_id=model.sensor("torque").id,
        )
        rotation = data.site_xmat[model.site("ft").id].reshape(3, 3)
        expected = -np.r_[
            rotation.T @ load[:3],
            rotation.T @ (load[3:] + np.cross([0.1, 0.02, 0], load[:3])),
        ]
        records.append(
            {
                "axis": axis,
                "expected": expected.tolist(),
                "measured": measured.tolist(),
                "max_error": float(np.max(abs(measured - expected))),
            }
        )
    return records


def run_episode(env, seed=7, fault=None):
    _, info = env.reset(seed=seed)
    policy = CableStretchScript() if env.dual_end else CableHandoverScript()
    rows = []
    unloaded = False
    for step in range(env.max_episode_steps):
        action = policy.action(env)
        # 故意松手/持续外拉是验收输入；从不更改端头 pose 或伪造测量。
        if fault == "drop" and step >= 5:
            action[6] = action[13] = -1
        if fault == "overpull":
            action[1] = 0.2
            action[8] = -0.2
        _, _, terminated, truncated, info = env.step(action)
        unloaded |= policy.phase == "unloading"
        rows.append(
            {
                "time_s": float(env.data.time),
                "tension_n": env._cable_tension(),
                "stage": info["stage"],
                "controller": policy.phase,
                "left_grasp": env._grasped("left"),
                "right_grasp": env._grasped("right"),
            }
        )
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "scene_hash": env.scene.scene_hash,
        "material_parameters": env.scene.config["materials"]["cable"],
        "control": env.scene.config.get("control", {}),
        "resolved_disturbance_force_n": env._disturbance_force_n.tolist(),
        "fault": fault,
        "steps": step + 1,
        "success": info["is_success"],
        "failure_reason": info["failure_reason"],
        "unloaded": unloaded,
        "audit": info["audit"],
        "metrics": info["metrics"],
        "trace": rows,
    }


def variant_scene(root, config, constraints):
    """验收配置副本只写入临时目录；不会修改用户的菜谱。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "scene_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (root / "constraints.yaml").write_text(
        yaml.safe_dump(constraints, sort_keys=False), encoding="utf-8"
    )
    shutil.copyfile(
        DEFAULT_SCENE_PATH.parent / "cable_stretch" / "task_spec.py",
        root / "task_spec.py",
    )
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/core01_physics_review.json")
    )
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    scene = SceneLoader().validate(DEFAULT_SCENE_PATH.parent / "cable_stretch")
    params = CableParameters.from_scene(scene.config["materials"]["cable"])
    report = {
        "mujoco_version": mujoco.__version__,
        "scene_hash": scene.scene_hash,
        "simulation_source_hash": simulation_source_hash(),
        "parameter_status": "provisional",
        "axial": [axial_fixture(replace(params, segments=n)) for n in (6, 12, 24)],
        "bending": bending_gradient_check(),
        "sag": [sag_fixture(params, ei) for ei in (0.0002, 0.002)],
        "sensors": sensor_fixture(),
        "episodes": [],
    }
    checks = [
        all(row["relative_error"] < 0.02 for row in report["axial"]),
        all(
            abs(row["mass_kg"] - row["expected_mass_kg"]) < 1e-10
            for row in report["axial"]
        ),
        report["bending"]["gradient_max_error"] < 1e-6,
        report["bending"]["net_force_n"] < 1e-8,
        report["bending"]["net_torque_nm"] < 1e-8,
        all(row["max_error"] < 1e-6 for row in report["sensors"]),
    ]
    checks.extend(
        [
            all(row["load_error"] < 0.05 for row in report["sag"]),
            report["sag"][1]["sag_m"] < report["sag"][0]["sag_m"],
        ]
    )
    with tempfile.TemporaryDirectory(prefix="ibero-validation-") as temporary:
        for label, changes in [
            ("default", {}),
            ("softer", {"axial_stiffness_n_m": 120.0, "bending_stiffness_nm2": 0.0001}),
            ("heavier", {"linear_density_kg_m": 0.075, "axial_stiffness_n_m": 240.0}),
        ]:
            config = copy.deepcopy(scene.config)
            config["materials"]["cable"].update(changes)
            path = variant_scene(Path(temporary) / label, config, scene.constraints)
            env = ibero.CableHandoverEnv(scene_path=path)
            try:
                for seed in range(args.seeds):
                    result = run_episode(env, seed)
                    result["variant"] = label
                    report["episodes"].append(result)
                    checks.append(result["success"])
                    print(
                        label,
                        seed,
                        result["success"],
                        result["failure_reason"],
                        flush=True,
                    )
            finally:
                env.close()
        for fault in ("drop", "overpull"):
            env = ibero.make("ibero/CableStretch-v0")
            try:
                result = run_episode(env, fault=fault)
            finally:
                env.close()
            report["episodes"].append(result)
            checks.append(
                not result["success"]
                and result["failure_reason"]
                == ("dropped" if fault == "drop" else "cable_damage")
            )
            print(fault, result["failure_reason"], flush=True)
        config = copy.deepcopy(scene.config)
        config["control"]["disturbance_force_n"] = [0, 0, -3.0]
        config["physics"]["max_episode_steps"] = 600
        path = variant_scene(Path(temporary) / "unload", config, scene.constraints)
        env = ibero.CableHandoverEnv(scene_path=path)
        try:
            result = run_episode(env)
        finally:
            env.close()
        result["variant"] = "warning_unload"
        report["episodes"].append(result)
        checks.append(result["unloaded"] and result["success"])
        print("warning_unload", result["unloaded"], result["success"], flush=True)
    env = ibero.make("ibero/CableHandover-v0")
    try:
        for seed in range(args.seeds):
            result = run_episode(env, seed)
            result["variant"] = "handover"
            report["episodes"].append(result)
            checks.append(result["success"])
            print(
                "handover",
                seed,
                result["success"],
                result["failure_reason"],
                flush=True,
            )
    finally:
        env.close()
    report["passed"] = all(checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved", args.output, "passed=", report["passed"])
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
