"""Core-0.2 布线任务验证：真实回合、反例、回放及可追溯结果。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np

import ibero
from ibero.control.cable_routing import routing_policy_for
from ibero.core.audit import AuditSnapshot
from ibero.core.reproducibility import simulation_source_hash
from ibero.core.scene_loader import SceneLoader
from ibero.core.task_api import TaskState
from ibero.review import Trace
from ibero.robots.g1_upperbody_2f85 import vendored_robot_asset_hash
from ibero.tasks.cable_routing import CableRoutingTask, RoutingGoal


ROOT = Path(__file__).resolve().parents[2]
SCENES = {
    "cable_clip_insert": "ibero/CableClipInsert-v0",
    "cable_route_straight": "ibero/CableRouteStraight-v0",
    "cable_route_l": "ibero/CableRouteL-v0",
    "cable_route_rectangle": "ibero/CableRouteRectangle-v0",
}
SUITES = ("quick", "bench", "tasks", "heldout")


def validate_scene(name: str) -> dict:
    """只校验菜谱；真正的可达与物理成功由回合检查。"""
    scene = SceneLoader().validate(ROOT / "scenes" / name)
    return {
        "name": f"scene:{name}",
        "passed": scene.config["id"] == name,
        "scene_hash": scene.scene_hash,
        "clip_count": len(scene.config["workcell"]["clips"]),
    }


def _counterfactual_state(*, cable_z: float, jaw_gap: float) -> TaskState:
    """同一几何状态只改变线位置或夹口；不借策略阶段做判定。"""
    clip = {
        "id": "clip_01",
        "origin_m": [0, 0, 0],
        "rotation_world": np.eye(3),
        "seat_z_min_m": 0.005,
        "seat_z_max_m": 0.024,
        "seat_half_width_m": 0.0056,
        "seat_half_length_m": 0.010,
        "lip_z_m": 0.025,
        "wire_radius_m": 0.005,
        "jaw_gap_m": jaw_gap,
        "jaw_recovered": True,
        "max_penetration_m": 0.0,
        "penetration_limit_m": 0.001,
    }
    return TaskState(
        body_positions={}, site_positions={}, gripper_openings={},
        grasps={"left_gripper": False, "right_gripper": False},
        contacts=frozenset(), cable_tension_n=0.0,
        elapsed_steps=1, max_episode_steps=100,
        extra={"routing": {
            "cable_points_m": np.array([[0, -0.05, cable_z], [0, 0.05, cable_z]]),
            "clips": [clip], "dt_s": 0.05,
            "cable_speed_m_s": 0.0, "hands_clear": True,
        }},
    )


def counterfactual_checks() -> list[dict]:
    """防止未压入、夹口过宽时被评为已固定。"""
    audit = AuditSnapshot(
        cable_damage=False, self_collision=False,
        cable_over_limit_seconds=0.0, peak_cable_tension_n=0.0, events=(),
    )
    checks = []
    for name, state in (
        ("no_entry", _counterfactual_state(cable_z=0.033, jaw_gap=0.0076)),
        ("wide_jaw", _counterfactual_state(cable_z=0.008, jaw_gap=0.011)),
    ):
        goal = RoutingGoal(
            ("clip_01",), hold_time_s=0.05,
            require_two_hand_grasp=False, require_entry_history=False,
        )
        result = CableRoutingTask(goal).evaluate(state, audit)
        checks.append({
            "name": f"counterfactual:{name}",
            "passed": not result.success and result.metrics.get("retained_count") == 0,
            "success": result.success,
            "retained_count": result.metrics.get("retained_count"),
            "failure_reason": result.failure_reason,
        })
    return checks


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_episode(
    scene_name: str,
    seed: int,
    *,
    max_steps: int | None = None,
    trace_path: Path | None = None,
) -> dict:
    """真实参考策略通过 env.step 操作模型；独立任务评估结果决定通过。"""
    if scene_name not in SCENES:
        raise ValueError(f"Unknown verified scene: {scene_name}")
    env = ibero.make(SCENES[scene_name])
    started = time.perf_counter()
    try:
        _, initial_info = env.reset(seed=seed)
        initial_qpos = env.data.qpos.copy()
        if initial_info["is_success"]:
            raise AssertionError("The initial scene already reports task success")
        if max_steps is None:
            max_steps = env.max_episode_steps
        if type(max_steps) is not int or not 1 <= max_steps <= env.max_episode_steps:
            raise ValueError("max_steps must be within the scene episode length")
        policy = routing_policy_for(env.scene.config["id"], env.scene.config["kind"])
        trace = Trace(env) if trace_path is not None else None
        if trace is not None:
            if trace_path.exists():
                raise FileExistsError(f"Refusing to overwrite trace: {trace_path}")
            trace.append(initial_info)
        info = initial_info
        terminated = truncated = False
        steps = 0
        for steps in range(1, max_steps + 1):
            _, _, terminated, truncated, info = env.step(policy.action(env))
            info["controller_phase"] = policy.phase
            if trace is not None:
                trace.append(info)
            if terminated or truncated:
                break
        row = {
            "name": f"episode:{scene_name}:{seed}",
            "scene_id": scene_name,
            "scene_hash": env.scene.scene_hash,
            "seed": seed,
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated,
            "stage": info["stage"],
            "passed": bool(info["is_success"] and terminated and not truncated),
            "failure_reason": info.get("failure_reason"),
            "metrics": info.get("metrics", {}),
            "audit": info.get("audit", {}),
            "simulation_time_s": info.get("simulation_time_s"),
            "wall_seconds": time.perf_counter() - started,
            "max_steps": max_steps,
        }
        if trace is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace.save(trace_path)
            row["trace"] = str(trace_path)
            row["trace_sha256"] = _hash_file(trace_path)
            loaded = Trace(env).load(trace_path)
            restored = loaded.restore(-1)
            # 恢复物理画面仅供审阅，不能拿这个状态直接继续控制回合。
            replay_blocked = False
            try:
                env.step(np.zeros(14, dtype=np.float32))
            except RuntimeError:
                replay_blocked = True
            row["trace_roundtrip_passed"] = bool(
                restored.get("replay")
                and restored["is_success"] == info["is_success"]
                and len(loaded.states) == steps + 1
                and replay_blocked
            )
            env.reset(seed=seed)
            row["deterministic_reset_passed"] = bool(
                np.allclose(env.data.qpos, initial_qpos, atol=1e-9, rtol=0)
            )
            row["passed"] &= row["trace_roundtrip_passed"]
            row["passed"] &= row["deterministic_reset_passed"]
        return row
    finally:
        env.close()


def bench_check() -> dict:
    """直接执行可变形线夹的真实物理测试，不引用历史回合文件。"""
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_cable_clip.py"]
    started = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    return {
        "name": "bench:physical_clip",
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "command": command,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "wall_seconds": time.perf_counter() - started,
    }


def _run_named(name: str, callback) -> dict:
    try:
        return callback()
    except Exception as error:
        return {
            "name": name,
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }


def build_report(
    suite: str,
    *,
    output: Path,
    seeds: tuple[int, ...] | None = None,
    max_steps: int | None = None,
    scenes: tuple[str, ...] | None = None,
) -> dict:
    """每个预定回合都入分母；异常和数据缺失都保留为失败行。"""
    if suite not in SUITES:
        raise ValueError(f"Unsupported routing suite: {suite}")
    selected_scenes = tuple(SCENES) if scenes is None else tuple(scenes)
    if not selected_scenes or len(set(selected_scenes)) != len(selected_scenes) or any(
        name not in SCENES for name in selected_scenes
    ):
        raise ValueError("Scene selection must contain distinct known routing scenes")
    source_hash = simulation_source_hash()
    report = {
        "suite": suite,
        "gate_version": "C02-TASK-GATES-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hash": source_hash,
        "asset_hash": vendored_robot_asset_hash(),
        "mujoco_version": mujoco.__version__,
        "parameter_status": "provisional",
        "selected_scenes": list(selected_scenes),
        "checks": [],
    }
    checks = report["checks"]
    if suite in {"quick", "tasks", "heldout"}:
        for name in selected_scenes:
            checks.append(_run_named(f"scene:{name}", lambda name=name: validate_scene(name)))
    if suite == "bench":
        checks.append(_run_named("bench:physical_clip", bench_check))
    if suite == "quick":
        checks.extend(counterfactual_checks())
        chosen_seeds = (0,)
    elif suite == "tasks":
        chosen_seeds = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) if seeds is None else seeds
    elif suite == "heldout":
        chosen_seeds = tuple(range(100, 110)) if seeds is None else seeds
    else:
        chosen_seeds = ()
    for scene_name in selected_scenes if suite != "bench" else ():
        for seed in chosen_seeds:
            trace_path = (
                output.parent / f"{output.stem}_{scene_name}_seed{seed}.npz"
                if suite == "quick" and scene_name == "cable_clip_insert" else None
            )
            checks.append(_run_named(
                f"episode:{scene_name}:{seed}",
                lambda scene_name=scene_name, seed=seed, trace_path=trace_path:
                run_episode(scene_name, seed, max_steps=max_steps, trace_path=trace_path),
            ))
    episodes = [row for row in checks if row["name"].startswith("episode:")]
    non_episodes = [row for row in checks if not row["name"].startswith("episode:")]
    report["episode_denominator"] = len(episodes)
    report["episode_successes"] = sum(bool(row["passed"]) for row in episodes)
    report["seed_set"] = list(chosen_seeds)
    report["scene_rates"] = {}
    for scene_name in selected_scenes if suite != "bench" else ():
        rows = [row for row in episodes if row["name"].startswith(f"episode:{scene_name}:")]
        report["scene_rates"][scene_name] = {
            "successes": sum(bool(row["passed"]) for row in rows),
            "denominator": len(rows),
        }
    non_episode_ok = bool(checks) and all(bool(row["passed"]) for row in non_episodes)
    default_seeds = (
        tuple(range(10)) if suite == "tasks" else
        tuple(range(100, 110)) if suite == "heldout" else chosen_seeds
    )
    report["gate_complete"] = chosen_seeds == default_seeds and selected_scenes == tuple(SCENES)
    if not report["gate_complete"]:
        # 少量种子是调试结果，不能冒充预登记十例门槛。
        report["passed"] = None
    elif suite in {"tasks", "heldout"}:
        required = {
            name: (8 if name in {"cable_route_l", "cable_route_rectangle"} or suite == "heldout" else 9)
            for name in SCENES
        }
        report["required_successes_per_scene"] = required
        report["passed"] = bool(non_episode_ok and all(
            row["denominator"] == 10 and row["successes"] >= required[name]
            for name, row in report["scene_rates"].items()
        ))
    else:
        report["passed"] = bool(
            non_episode_ok and all(bool(row["passed"]) for row in episodes)
        )
    report["source_unchanged_during_run"] = simulation_source_hash() == source_hash
    if report["passed"] is not None:
        report["passed"] &= report["source_unchanged_during_run"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--output", type=Path, help="new report.json path; existing files are never overwritten")
    parser.add_argument("--steps", type=int, help="optional bounded smoke run; a short incomplete episode still fails")
    parser.add_argument("--seeds", help="comma-separated nonnegative seeds for explicit tasks/heldout runs")
    parser.add_argument("--scenes", help="comma-separated scene IDs for a diagnostic subset; cannot satisfy the full gate")
    args = parser.parse_args()
    if args.steps is not None and (args.suite == "bench" or args.steps < 1):
        parser.error("--steps must be positive and requires an episode suite")
    if args.seeds is not None and args.suite not in {"tasks", "heldout"}:
        parser.error("--seeds only applies to tasks/heldout")
    seeds = None
    if args.seeds is not None:
        try:
            values = [int(part) for part in args.seeds.split(",")]
            if not values or len(values) != len(set(values)) or any(v < 0 for v in values):
                raise ValueError
            seeds = tuple(values)
        except ValueError:
            parser.error("--seeds must be distinct nonnegative integers separated by commas")
    scenes = None
    if args.scenes is not None:
        scenes = tuple(args.scenes.split(","))
        if not scenes or len(set(scenes)) != len(scenes) or any(name not in SCENES for name in scenes):
            parser.error("--scenes must list distinct known routing scene IDs")
    output = args.output or (
        ROOT / "artifacts" / "routing_core02" / args.suite
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") / "report.json"
    )
    output = output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing report: {output}")
    report = build_report(args.suite, output=output, seeds=seeds, max_steps=args.steps, scenes=scenes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved {output} | passed={report['passed']} | episodes={report['episode_successes']}/{report['episode_denominator']}")
    raise SystemExit(3 if report["passed"] is None else 0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
