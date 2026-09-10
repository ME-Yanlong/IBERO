"""工业场景 demo 的薄入口；物理环境、策略、轨迹与显示保持各自责任。"""

import json
from PIL import Image
from ibero.envs.latch_release import LatchReleaseEnv
from ibero.control.latch import LatchReleaseScript
from ibero.review import Trace


def run_demo(args):
    env = LatchReleaseEnv(**({"scene_path": args.scene} if args.scene else {}))
    app = None
    try:
        if args.viewer:
            from ibero.industrial_review import IndustrialReviewer

            app = IndustrialReviewer(
                env,
                seed=args.seed,
                steps=args.steps,
                playback_rate=args.playback_rate,
                save_trace=args.save_trace,
                replay=args.replay,
            )
            info = app.run()
            trace = app.trace
        else:
            env.reset(seed=args.seed)
            policy = LatchReleaseScript()
            trace = Trace(env)
            trace.append(env.last_info)
            info = env.last_info
            for _ in range(args.steps):
                _, _, term, trunc, info = env.step(policy.action(env))
                info["controller_phase"] = policy.phase
                trace.append(info)
                if term or trunc:
                    break
            if not (term or trunc):
                info = dict(info, failure_reason="demo_step_limit")
            if args.save_trace:
                trace.save(args.save_trace)
        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            frame = (
                app.last_image.crop((0, 0, 480, 320))
                if app
                else Image.fromarray(env.render())
            )
            frame.save(args.save_frame)
        if args.save_multiview:
            from ibero.industrial_review import IndustrialViews

            args.save_multiview.parent.mkdir(parents=True, exist_ok=True)
            if app:
                app.last_image.save(args.save_multiview)
            else:
                views = IndustrialViews(
                    env.model, env.resolved_config["initialization"]["origin_m"]
                )
                try:
                    views.set_state(trace.states[-1])
                    views.render(trace.infos).save(args.save_multiview)
                finally:
                    views.close()
        if args.save_manifest:
            args.save_manifest.parent.mkdir(parents=True, exist_ok=True)
            args.save_manifest.write_text(
                json.dumps(dict(env.manifest(), final_info=info), indent=2),
                encoding="utf-8",
            )
        print("ibero/LatchRelease-v0:", info)
        print("Success:", info.get("success", False))
        if not args.viewer and not info.get("success"):
            raise SystemExit(1)
    finally:
        env.close()
