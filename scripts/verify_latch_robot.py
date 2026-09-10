"""S4 latch_release 分组长验收；不包含尚未完成的 harness_unplug，不冒称 G4 全部通过。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import yaml
from ibero.envs.latch_release import LatchReleaseEnv
from ibero.control.latch import LatchReleaseScript
from ibero.review import Trace
from ibero.core.reproducibility import simulation_source_hash

ROOT = Path(__file__).resolve().parents[1]


def run_one(job):
    scene, seed, case, output = job
    path = Path(output) / f"{scene}_{seed}_{case}"
    path.mkdir(parents=True, exist_ok=False)
    env = None
    rows = []
    started = time.monotonic()
    report = {
        "scene": scene,
        "seed": seed,
        "case": case,
        "passed": False,
        "success": False,
    }
    try:
        if scene.startswith("harness_unplug"):
            from ibero.envs.harness_unplug import HarnessUnplugEnv
            from ibero.control.harness import HarnessUnplugScript

            env = HarnessUnplugEnv(ROOT / "scenes" / scene)
            policy = HarnessUnplugScript(case)
        else:
            env = LatchReleaseEnv(ROOT / "scenes" / scene)
            policy = LatchReleaseScript(case)
        env.reset(seed=seed)
        trace = Trace(env)
        trace.append(env.last_info)
        report["manifest"] = env.manifest()
        for _ in range(env.max_episode_steps):
            _, _, term, trunc, info = env.step(policy.action(env))
            info["controller_phase"] = policy.phase
            rows.append(info)
            trace.append(info)
            if term or trunc:
                break
        final = rows[-1]
        report.update(success=final["success"], final=final, trace=rows)
        gate = yaml.safe_load(
            (ROOT / "validation/industrial_gates.yaml").read_text(encoding="utf-8")
        )["latch"]
        report["metric_checks"] = {
            "beam_force": final["peak_contact_force_n"]
            <= env.scene.constraints["safety"]["max_force_n"],
            "beam_penetration": final["peak_penetration_m"]
            <= gate["penetration_feature_fraction"]
            * env.scene.config["mechanism"]["overlap_m"],
        }
        report["passed"] = (
            (final["success"] and all(report["metric_checks"].values()))
            if case == "normal"
            else (
                not final["success"]
                and any(r["grasped"] for r in rows)
                and (not final["released"] or bool(final["invalid_reason"]))
            )
        )
        trace.save(path / "trace.npz")
        if seed == 0 and case == "normal":
            from ibero.industrial_review import IndustrialViews

            views = IndustrialViews(
                env.model, env.resolved_config["initialization"]["origin_m"]
            )
            try:
                views.set_state(trace.states[-1])
                views.render(trace.infos).save(path / "final.png")
            finally:
                views.close()
    except Exception as error:
        report["exception"] = f"{type(error).__name__}: {error}"
    finally:
        report["wall_seconds"] = time.monotonic() - started
        if env is not None:
            report["sim_seconds"] = float(env.data.time)
            report["real_time_factor"] = float(env.data.time) / report["wall_seconds"]
            env.close()
        (path / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    return {k: v for k, v in report.items() if k not in {"trace", "manifest"}} | {
        "path": str(path)
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--scene", choices=["latch_release", "harness_unplug"], default="latch_release"
    )
    p.add_argument(
        "--group", choices=["all", "default", "variants", "faults"], default="all"
    )
    args = p.parse_args()
    output = (
        args.output
        or ROOT
        / "artifacts/industrial_core01/latch/robot_validation"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    gates = yaml.safe_load(
        (ROOT / "validation/industrial_gates.yaml").read_text(encoding="utf-8")
    )
    source = simulation_source_hash()
    jobs = []
    if args.group in {"all", "default"}:
        jobs += [(args.scene, s, "normal", str(output)) for s in range(20)]
    if args.group in {"all", "variants"}:
        jobs += [
            (f"{args.scene}_{v}", s, "normal", str(output))
            for v in ("thinner", "shallower")
            for s in range(100, 105)
        ]
    if args.group in {"all", "faults"}:
        jobs += [
            (args.scene, 0, c, str(output))
            for c in ("no_press", "partial_press", "offset_press", "early_release")
        ]
    report = {
        "scope": args.scene + "_only",
        "excludes": [
            s
            for s in ["latch_release", "harness_unplug", "stock", "milling"]
            if s != args.scene
        ],
        "source_hash": source,
        "gate_version": gates["version"],
        "workers": args.workers,
        "results": [],
        "checks": {},
    }
    print("Evidence", output, "jobs", len(jobs), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_one, j) for j in jobs]):
            row = future.result()
            report["results"].append(row)
            print(
                row["scene"],
                row["seed"],
                row["case"],
                "passed",
                row["passed"],
                row.get("final", {}).get("failure_reason"),
                flush=True,
            )
            (output / "progress.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
    r = report["results"]
    gate = gates["latch"]
    if args.group in {"all", "default"}:
        report["checks"]["default"] = (
            sum(
                x["passed"]
                for x in r
                if x["scene"] == args.scene and x["case"] == "normal"
            )
            >= gate["default_minimum_successes"]
        )
    if args.group in {"all", "variants"}:
        for variant in ("thinner", "shallower"):
            report["checks"][variant] = (
                sum(x["passed"] for x in r if x["scene"] == f"{args.scene}_{variant}")
                >= gate["held_out_minimum_successes"]
            )
    if args.group in {"all", "faults"}:
        report["checks"]["counterfactuals"] = all(
            x["passed"] for x in r if x["case"] != "normal"
        )
    simulation_source_hash.cache_clear()
    report["checks"]["frozen_source"] = source == simulation_source_hash()
    report["passed"] = all(report["checks"].values())
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("Summary", report["checks"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
