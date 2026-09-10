"""加工 CLI：使用同一物理工作站，窗口持久等待，无头执行失败返回非零。"""

import json
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.control.milling_bench import MillingBenchScript
from ibero.core.stock_trace import StockTrace
from ibero.milling_review import MillingReviewer, MillingViews


def run_demo(args):
    env = PlateMillingEnv(
        shape=args.shape, **({"scene_path": args.scene} if args.scene else {})
    )
    app = None
    target = env.target
    if args.viewer:
        app = MillingReviewer(
            env,
            shape=args.shape,
            seed=args.seed,
            steps=args.steps,
            playback_rate=args.playback_rate,
            save_trace=args.save_trace,
            replay=args.replay,
            visual_chips=not args.no_visual_chips,
        )
        info = app.run()
        trace = app.trace
    else:
        env.reset(seed=args.seed)
        policy = MillingBenchScript(env, target)
        trace = StockTrace(env)
        trace.append(env.last_info)
        substeps = round(
            1 / env.scene.config["physics"]["control_hz"] / env.model.opt.timestep
        )
        for index in range(args.steps):
            command, rpm = policy.command(env)
            info = env.step(command, rpm, substeps=substeps)
            info["controller_phase"] = policy.phase
            finish = env._done or policy.finished or index + 1 == args.steps
            if finish:
                info["task_check"] = env.evaluate_task(target)
                info["success"] = bool(
                    info["task_check"]["result"]["success"] and policy.finished
                )
                if not info["success"]:
                    info["failure_reason"] = info["invalid_reason"] or (
                        "shape_or_finish_check_failed"
                        if policy.finished
                        else "demo_step_limit"
                    )
            trace.append(info)
            if finish:
                break
        if args.save_trace:
            trace.save(args.save_trace)
    if args.save_frame or args.save_multiview:
        if app:
            frame = app.last_image
        else:
            views = MillingViews(env, visual_chips=not args.no_visual_chips)
            try:
                views.set_frame(trace, len(trace.states) - 1)
                frame = views.render(trace.infos)
            finally:
                views.close()
        for path, im in (
            (args.save_frame, frame.crop((480, 0, 960, 320))),
            (args.save_multiview, frame),
        ):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                im.save(path)
    if args.save_manifest:
        args.save_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.save_manifest.write_text(
            json.dumps(
                dict(env.manifest(), final_info=info), indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
    print("ibero/plate_milling:", json.dumps(info, ensure_ascii=False, allow_nan=False))
    print("Success:", info.get("success", False))
    if not args.viewer and not info.get("success", False):
        raise SystemExit(1)
