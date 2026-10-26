"""Core-0.2 布线参考回合、回放和审阅命令。"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
from PIL import Image

import ibero
from ibero.control.cable_routing import routing_policy_for
from ibero.core.reproducibility import simulation_source_hash
from ibero.review import Trace
from ibero.robots.g1_upperbody_2f85 import vendored_robot_asset_hash
from ibero.routing_review import RoutingReviewer, RoutingViews


ENV_IDS = {
    "cable_clip_insert": "ibero/CableClipInsert-v0",
    "cable_route_straight": "ibero/CableRouteStraight-v0",
    "cable_route_l": "ibero/CableRouteL-v0",
    "cable_route_rectangle": "ibero/CableRouteRectangle-v0",
}


def _save_multiview(env, path):
    views = RoutingViews(env.model, env.scene.config["workcell"]["clips"][0]["origin_m"])
    try:
        state = mujoco.mj_stateSize(env.model, Trace.spec)
        import numpy as np
        values = np.empty(state)
        mujoco.mj_getState(env.model, env.data, values, Trace.spec)
        views.set_state(values)
        image = views.render([env.last_info])
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    finally:
        views.close()


def run_demo(args):
    if args.replay is not None and not args.viewer:
        raise ValueError("--replay requires --viewer")
    env = ibero.make(
        ENV_IDS[args.env],
        render_mode="rgb_array" if args.save_frame else None,
        **({"scene_path": args.scene} if args.scene is not None else {}),
    )
    try:
        if args.viewer:
            info = RoutingReviewer(
                env,
                seed=args.seed,
                steps=args.steps,
                playback_rate=args.playback_rate,
                save_trace=args.save_trace,
                replay=args.replay,
            ).run()
        else:
            _, info = env.reset(seed=args.seed)
            policy = routing_policy_for(env.scene.config["id"], env.scene.config["kind"])
            trace = Trace(env) if args.save_trace else None
            if trace:
                trace.append(info)
            for _ in range(min(args.steps, env.max_episode_steps)):
                _, _, terminated, truncated, info = env.step(policy.action(env))
                info["controller_phase"] = policy.phase
                if trace:
                    trace.append(info)
                if terminated or truncated:
                    break
            if trace:
                trace.save(args.save_trace)
        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(env.render()).save(args.save_frame)
        if args.save_multiview:
            _save_multiview(env, args.save_multiview)
        if args.save_manifest:
            manifest = {
                "scene_id": env.scene.config["id"],
                "scene_hash": env.scene.scene_hash,
                "source_hash": simulation_source_hash(),
                "asset_hash": vendored_robot_asset_hash(),
                "mujoco_version": mujoco.__version__,
                "seed": args.seed,
                "result": info,
            }
            destination = Path(args.save_manifest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{ENV_IDS[args.env]} audit:", info["audit"])
        print("Success:", info["is_success"])
        print("Stage:", info["stage"])
        if info.get("failure_reason"):
            print("Failure:", info["failure_reason"])
        return info
    finally:
        env.close()
