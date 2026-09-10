"""Run reproducible Core-0.1 material and wrist-sensor calibration checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ibero.baselines import CableHandoverScript
from ibero.envs.cable_handover import CableHandoverEnv
from ibero.core.reproducibility import simulation_source_hash


def _hold_action() -> np.ndarray:
    action = np.zeros(14, dtype=np.float32)
    action[6] = 1.0
    action[13] = -1.0
    return action


def wrist_static_load_check(env: CableHandoverEnv) -> dict[str, float | bool]:
    """Verify sensor ordering/sign against a synthetic, gravity-free 5 N load."""

    gravity = env.model.opt.gravity.copy()
    env.model.opt.gravity[:] = 0.0
    try:
        env.reset(seed=0)
        action = _hold_action()
        for _ in range(10):
            env.step(action)
        baseline = env._observation()["wrench"][:6].copy()
        applied_n = 5.0
        # `xfrc_applied` is a simulator-side calibration fixture.  The force
        # reaches the wrist through the real 2F-85 mounting chain and is not
        # used by the handover task or baseline.
        # MuJoCo xfrc_applied 的顺序是 [Fx,Fy,Fz,Tx,Ty,Tz]，这里是世界系外力。
        env.data.xfrc_applied[env.handles.left_gripper_base_body_id, :3] = [
            0.0,
            0.0,
            -applied_n,
        ]
        for _ in range(5):
            env.step(action)
        measured = env._observation()["wrench"][:6]
        delta_z = float(measured[2] - baseline[2])
        relative_error = abs(abs(delta_z) - applied_n) / applied_n
        return {
            "applied_force_n": applied_n,
            "measured_delta_fz_n": delta_z,
            "relative_magnitude_error": relative_error,
            "sign_matches_sensor_frame": bool(delta_z > 0.0),
            "passed": bool(relative_error <= 0.05 and delta_z > 0.0),
        }
    finally:
        env.data.xfrc_applied[env.handles.left_gripper_base_body_id] = 0.0
        env.model.opt.gravity[:] = gravity


def stability_check(
    env: CableHandoverEnv, *, seeds: int, seconds: float, seed_start: int = 0
) -> list[dict[str, Any]]:
    """Run passive safe holds and report Flex/capsule numerical stability."""

    results: list[dict[str, Any]] = []
    steps = int(round(seconds / env.control_dt))
    action = _hold_action()
    for seed in range(seed_start, seed_start + seeds):
        _, info = env.reset(seed=seed)
        terminated = truncated = False
        for step in range(steps):
            observation, _, terminated, truncated, info = env.step(action)
            if not all(np.isfinite(value).all() for value in observation.values()):
                break
            if terminated or truncated:
                break
        results.append(
            {
                "seed": seed,
                "steps": step + 1,
                "stable": bool(
                    not terminated
                    and not truncated
                    and all(np.isfinite(value).all() for value in observation.values())
                ),
                "peak_cable_tension_n": info["audit"]["peak_cable_tension_n"],
                "failure_reason": info["failure_reason"],
            }
        )
        print("stability", seed, results[-1]["stable"], flush=True)
    return results


def scripted_baseline_check(
    env: CableHandoverEnv, *, seeds: int, seed_start: int = 0
) -> list[dict[str, Any]]:
    """Measure the contact-only handover baseline over fixed seeds."""

    results: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + seeds):
        env.reset(seed=seed)
        baseline = CableHandoverScript()
        for step in range(env.max_episode_steps):
            _, _, terminated, truncated, info = env.step(baseline.action(env))
            if terminated or truncated:
                break
        results.append(
            {
                "seed": seed,
                "steps": step + 1,
                "success": info["is_success"],
                "failure_reason": info["failure_reason"],
                "peak_cable_tension_n": info["audit"]["peak_cable_tension_n"],
                "self_collision": info["audit"]["self_collision_flags"]["left_right"],
            }
        )
        print(
            "handover",
            seed,
            results[-1]["success"],
            results[-1]["failure_reason"],
            flush=True,
        )
    return results


def run_report(
    *,
    stability_seeds: int,
    stability_seconds: float,
    baseline_seeds: int,
    stability_seed_start: int = 0,
    baseline_seed_start: int = 0,
) -> dict[str, Any]:
    env = CableHandoverEnv()
    try:
        ft = wrist_static_load_check(env)
        stability = stability_check(
            env,
            seeds=stability_seeds,
            seconds=stability_seconds,
            seed_start=stability_seed_start,
        )
        baseline = scripted_baseline_check(
            env, seeds=baseline_seeds, seed_start=baseline_seed_start
        )
        return {
            "scene": env.scene.config["id"],
            "scene_hash": env.scene.scene_hash,
            "simulation_source_hash": simulation_source_hash(),
            "representation": env.cable_params.representation,
            "parameter_source": env.cable_params.parameter_source,
            "wrist_ft_static_load": ft,
            "stability": stability,
            "scripted_baseline": baseline,
            "summary": {
                "stability_passed": all(row["stable"] for row in stability),
                "baseline_successes": sum(row["success"] for row in baseline),
                "baseline_runs": len(baseline),
                "passed": bool(
                    ft["passed"]
                    and all(row["stable"] for row in stability)
                    and sum(row["success"] for row in baseline) >= 0.9 * len(baseline)
                ),
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-seeds", type=int, default=20)
    parser.add_argument("--stability-seconds", type=float, default=10.0)
    parser.add_argument("--baseline-seeds", type=int, default=20)
    parser.add_argument("--stability-seed-start", type=int, default=0)
    parser.add_argument("--baseline-seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.stability_seeds, args.baseline_seeds) <= 0
        or min(args.stability_seed_start, args.baseline_seed_start) < 0
        or not np.isfinite(args.stability_seconds)
        or args.stability_seconds <= 0
    ):
        parser.error(
            "Positive run counts/duration and nonnegative seed starts are required"
        )
    report = run_report(
        stability_seeds=args.stability_seeds,
        stability_seconds=args.stability_seconds,
        baseline_seeds=args.baseline_seeds,
        stability_seed_start=args.stability_seed_start,
        baseline_seed_start=args.baseline_seed_start,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    if not report["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
