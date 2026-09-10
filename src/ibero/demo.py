"""Run and optionally view IBERO's Core-0 or Core-0.1 scripted demos."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

import ibero
from ibero.baselines import CableHandoverScript
from ibero.control.tension import CableStretchScript
from ibero.review import Trace


def scripted_action(observation: dict[str, np.ndarray]) -> np.ndarray:
    """A deliberately simple force-band controller for smoke testing."""

    tension = float(observation["cable"][0])
    action = np.zeros(14, dtype=np.float32)
    if tension < 4.6:
        # Pull each wrist outward along world y.
        action[1] = 0.55
        action[8] = -0.55
    elif tension > 5.4:
        action[1] = -0.18
        action[8] = 0.18
    action[6] = 1.0
    action[13] = 1.0
    return action


def _run_episode(
    env: ibero.CableTensionEnv | ibero.CableHandoverEnv,
    steps: int,
    *,
    observation: dict[str, np.ndarray],
    info: dict[str, Any],
    on_step: Callable[[dict[str, Any], str, bool], None] | None = None,
    playback_rate: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run exactly one scripted episode from an already visible reset state."""

    handover = (
        (CableStretchScript() if env.dual_end else CableHandoverScript())
        if isinstance(env, ibero.CableHandoverEnv)
        else None
    )
    trace = Trace(env) if isinstance(env, ibero.CableHandoverEnv) else None
    if trace:
        trace.append(info)
    for _ in range(steps):
        # 这里故意只使用 policy action + env.step；demo 不得改变端头位姿、
        # 注入 weld 或绕过接触判定，因而仍是可审计的 baseline。
        action = (
            handover.action(env)
            if handover is not None
            else scripted_action(observation)
        )
        started = time.perf_counter()
        observation, _, terminated, truncated, info = env.step(action)
        if handover:
            info["controller_phase"] = handover.phase
        if trace:
            trace.append(info)
        if on_step is not None:
            on_step(info, "运行中", False)
        # playback_rate=1 对齐场景控制频率；例如 0.5 可放慢为两倍时长，便于
        # 肉眼检查双夹持和左手释放。无 viewer 的批处理不应无谓 sleep。
        if on_step is not None:
            remaining = env.control_dt / playback_rate - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
        if terminated or truncated:
            break
    if trace and getattr(env, "_save_trace_path", None):
        trace.save(env._save_trace_path)
    return observation, info


def rollout(
    env: ibero.CableTensionEnv | ibero.CableHandoverEnv,
    steps: int,
    viewer: bool,
    *,
    seed: int = 7,
    use_dashboard: bool = True,
    playback_rate: float = 1.0,
) -> dict[str, Any]:
    """Run one noninteractive episode or enter the persistent review session."""

    if playback_rate <= 0:
        raise ValueError("playback_rate must be positive")
    if viewer:
        from ibero.review import run_review

        return run_review(
            env,
            steps,
            seed=seed,
            use_dashboard=use_dashboard,
            playback_rate=playback_rate,
        )
    observation, info = env.reset(seed=seed)
    _, info = _run_episode(env, steps, observation=observation, info=info)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open a persistent reviewer; press Enter to run one episode",
    )
    parser.add_argument(
        "--env",
        choices=(
            "cable_tension",
            "cable_handover",
            "cable_stretch",
            "latch_release",
            "harness_unplug",
        ),
        default="cable_stretch",
        help="which scripted scene to run",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--seed", type=int, default=7, help="repeatable seed used for every Enter run"
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="interactive replay speed multiplier (0.5 is twice as slow)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="disable the optional cable_handover four-view status dashboard",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        help="override the selected environment's scene directory (YAML is the source of truth)",
    )
    parser.add_argument("--save-frame", type=Path, help="write the final rendered PNG")
    parser.add_argument(
        "--save-multiview",
        type=Path,
        help="write the final 2x2 review PNG",
    )
    parser.add_argument(
        "--save-manifest", type=Path, help="write Core-0.1 rollout metadata JSON"
    )
    parser.add_argument(
        "--save-trace",
        type=Path,
        help="write replayable physical states and audit records",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        help="review an existing trace; requires --viewer and the same scene",
    )
    args = parser.parse_args()
    if args.steps is None:
        args.steps = (
            2000
            if args.env == "harness_unplug"
            else 1500
            if args.env == "latch_release"
            else 600
        )
    if (
        args.steps <= 0
        or not np.isfinite(args.playback_rate)
        or args.playback_rate <= 0
    ):
        parser.error("--steps and --playback-rate must be finite and positive")
    if args.env in {"latch_release", "harness_unplug"}:
        if args.viewer and args.no_dashboard:
            parser.error(
                "Industrial reviewer currently uses one integrated four-view window; omit --no-dashboard"
            )
        if args.replay and not args.viewer:
            parser.error("--replay requires --viewer")
        from ibero.industrial_demo import run_demo

        run_demo(args)
        return
    if args.env == "cable_tension" and (args.replay or args.save_trace):
        parser.error("Physical trace/replay requires cable_handover or cable_stretch")
    env_id = {
        "cable_tension": "ibero/CableTension-v0",
        "cable_handover": "ibero/CableHandover-v0",
        "cable_stretch": "ibero/CableStretch-v0",
    }[args.env]
    if args.replay and not args.viewer:
        parser.error("--replay requires --viewer")
    if args.scene is not None and args.env == "cable_tension":
        parser.error("--scene is currently supported only with --env cable_handover")
    env = ibero.make(
        env_id,
        **({"scene_path": args.scene} if args.scene is not None else {}),
    )
    try:
        env._save_trace_path = args.save_trace
        env._replay_path = args.replay
        info = rollout(
            env,
            args.steps,
            args.viewer,
            seed=args.seed,
            use_dashboard=not args.no_dashboard,
            playback_rate=args.playback_rate,
        )
        if args.save_frame is not None:
            from PIL import Image

            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(env.render()).save(args.save_frame)
            print(f"Saved rendered frame: {args.save_frame}")
        if args.save_multiview is not None:
            if not isinstance(env, ibero.CableHandoverEnv):
                raise ValueError(
                    "--save-multiview is currently provided by CableHandover-v0"
                )
            from PIL import Image

            from ibero.multiview import CableHandoverMultiView

            args.save_multiview.parent.mkdir(parents=True, exist_ok=True)
            views = CableHandoverMultiView(env)
            try:
                # 与 viewer 面板使用完全相同的相机，不把人工审阅结果藏在
                # 仅交互路径中，便于 issue/PR 中保存一张可复查证据。
                Image.fromarray(views.render()).save(args.save_multiview)
            finally:
                views.close()
            print(f"Saved four-view review: {args.save_multiview}")
        if args.save_manifest is not None:
            if not isinstance(env, ibero.CableHandoverEnv):
                raise ValueError(
                    "--save-manifest is currently provided by CableHandover-v0"
                )
            env.save_manifest(args.save_manifest, info=info)
            print(f"Saved rollout manifest: {args.save_manifest}")
        print(f"{env_id} audit:", info["audit"])
        print("Success:", info["is_success"])
    finally:
        env.close()


if __name__ == "__main__":
    main()
