"""S7 三形状过程台架检查；不是 S8 机器人验收，失败保留且返回非零。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import shutil
import yaml
import numpy as np
from ibero.benches.milling import MillingFixture
from ibero.control.milling_bench import MillingBenchScript, bench_target
from ibero.processes.shape_check import inspect_shape
from ibero.benches.stock_geometry import section_image
from ibero.benches.stock_probe import inspect_machined_stock
from ibero.core.stock_trace import StockTrace
from ibero.core.reproducibility import simulation_source_hash

ROOT = Path(__file__).resolve().parents[1]


def run_shape(job):
    shape, output, scene_root, *options = job
    seed = options[0] if options else 0
    out = Path(output) / shape
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "scope": "S7_fixture_not_robot",
        "shape": shape,
        "passed": False,
        "rows": [],
        "seed": seed,
    }
    env, trace = None, None
    try:
        from ibero.core.scene_loader import SceneLoader

        robot = SceneLoader().validate(scene_root).config["kind"] == "plate_milling"
        if robot:
            from ibero.envs.plate_milling import PlateMillingEnv

            env = PlateMillingEnv.from_scene(scene_root, shape=shape)
            report["scope"] = "S8_robot_single_seed_not_full_G8"
        else:
            env = MillingFixture.from_scene(scene_root)
        env.reset(seed=seed)
        report["manifest"] = env.manifest()
        target = env.target if robot else bench_target(shape)
        report["target"] = asdict(target)
        policy = MillingBenchScript(env, target)
        trace = StockTrace(env)
        trace.append(env.last_info)
        steps = round(
            1 / env.scene.config["physics"]["control_hz"] / env.model.opt.timestep
        )
        max_time = env.scene.constraints["task"]["max_seconds"]
        while env.data.time < max_time:
            command, rpm = policy.command(env)
            info = env.step(command, rpm, substeps=steps)
            info["controller_phase"] = policy.phase
            info["tracking_error_m"] = float(
                np.linalg.norm(env.data.site("mill_tip").xpos - policy.nominal)
            )
            if (
                info["tracking_error_m"]
                > env.scene.constraints["safety"]["max_tracking_error_m"]
            ):
                env._done = True
                env.process.invalid_reason = info["invalid_reason"] = (
                    "tracking_error_limit"
                )
            report["rows"].append(info)
            trace.append(info)
            if len(report["rows"]) % 200 == 0 or env._done:
                (out / "progress.json").write_text(
                    json.dumps(
                        {
                            "shape": shape,
                            "seed": seed,
                            "sim_seconds": float(env.data.time),
                            "wall_seconds": time.perf_counter() - started,
                            "last_info": info,
                            "manifest": report["manifest"],
                        },
                        indent=2,
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                print(
                    shape,
                    round(env.data.time, 3),
                    policy.phase,
                    info["invalid_reason"],
                    "removed",
                    info["total_removed_volume_m3"],
                    flush=True,
                )
            if env._done or policy.finished:
                break
        report["shape_check"] = inspect_shape(env.stock, target)
        report["invalid_reason"] = env.process.invalid_reason
        report["controller_finished"] = policy.finished
        if robot:
            report["task_check"] = env.evaluate_task(target)
        trace.save(out / "trace.npz")
        expected_hash = env.stock.state_hash()
        expected_qpos = env.data.qpos.copy()
        loaded = StockTrace(env).load(out / "trace.npz")
        checked_frames = sorted(
            {
                0,
                len(trace.states) - 1,
                next((i for i, n in enumerate(trace.event_counts) if n > 0), 0),
            }
        )
        replay_checks = []
        for frame in checked_frames:
            loaded.restore(frame)
            replay_checks.append(env.stock.state_hash() == trace.material_hashes[frame])
            env.binding.ensure_consistent()
        loaded.restore(len(trace.states) - 1)
        replay_checks += [
            env.stock.state_hash() == expected_hash,
            np.array_equal(env.data.qpos, expected_qpos),
        ]
        try:
            env.step(env.start, 0, substeps=1)
        except RuntimeError:
            replay_checks.append(True)
        else:
            replay_checks.append(False)
        report["replay_check"] = {
            "frames": checked_frames,
            "passed": all(replay_checks),
            "resume_requires_reset": replay_checks[-1],
        }
        report["probe_check"] = inspect_machined_stock(env.stock, target)
        report["passed"] = bool(
            policy.finished
            and not report["invalid_reason"]
            and report["shape_check"]["passed"]
            and report["probe_check"]["passed"]
            and report["replay_check"]["passed"]
            and (not robot or report["task_check"]["result"]["success"])
        )
    except Exception as error:
        report["exception"] = f"{type(error).__name__}: {error}"
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        if env is not None:
            report["sim_seconds"] = float(env.data.time)
            report["real_time_factor"] = float(env.data.time) / report["wall_seconds"]
            section_image(env.stock, out / "sections.png")
            if trace is not None:
                trace.save(out / "trace.npz")
        simulation_source_hash.cache_clear()
        report["frozen_source"] = (
            report.get("manifest", {}).get("source_hash") == simulation_source_hash()
        )
        report["passed"] = report["passed"] and report["frozen_source"]
        (out / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    return {k: v for k, v in report.items() if k != "rows"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape", choices=["all", "slot", "pocket", "through_hole"], default="all"
    )
    p.add_argument("--workers", type=int, choices=range(1, 4), default=3)
    p.add_argument("--output", type=Path)
    p.add_argument("--scene", type=Path, default=ROOT / "scenes/milling_bench")
    p.add_argument(
        "--timestep-s", type=float, help="显式数值敏感性覆盖，另存完整解析菜谱"
    )
    p.add_argument("--cell-size-m", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--edge-transition-chip-m", type=float)
    args = p.parse_args()
    output = (
        args.output
        or ROOT
        / "artifacts/industrial_core01/stock/milling_process"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    if any(
        v is not None
        for v in (args.timestep_s, args.cell_size_m, args.edge_transition_chip_m)
    ):
        resolved = output / "resolved_scene"
        resolved.mkdir()
        cfg = yaml.safe_load(
            (args.scene / "scene_config.yaml").read_text(encoding="utf-8")
        )
        if args.timestep_s is not None:
            cfg["physics"]["timestep_s"] = args.timestep_s
        for name, value in (
            ("cell_size_m", args.cell_size_m),
            ("edge_transition_chip_m", args.edge_transition_chip_m),
        ):
            if value is not None:
                cfg["numerics"][name] = value
        (resolved / "scene_config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        for name in ("constraints.yaml", "task_spec.py"):
            shutil.copyfile(args.scene / name, resolved / name)
        # 覆盖不绕过 P4 校验，也不改原场景文件；证据目录包含真正使用的整份菜谱。
        from ibero.core.scene_loader import SceneLoader

        SceneLoader().validate(resolved)
        args.scene = resolved
    names = ["slot", "pocket", "through_hole"] if args.shape == "all" else [args.shape]
    rows = []
    print("Evidence", output, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(run_shape, (s, str(output), str(args.scene), args.seed))
            for s in names
        ]
        for future in as_completed(jobs):
            row = future.result()
            rows.append(row)
            print(
                row["shape"], "passed", row["passed"], row.get("exception"), flush=True
            )
    report = {
        "scope": "three_shape_execution_subset_see_per_case_scope_not_full_G7_or_G8",
        "results": rows,
        "passed": all(r["passed"] for r in rows),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
