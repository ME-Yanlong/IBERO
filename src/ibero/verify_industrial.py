"""工业机制分组验收：单测、物理长测、显式 GUI 与外部参考分开记录。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import mujoco
import yaml

from ibero.benches.beam import beam_static_test, flex_candidate_test
from ibero.benches.latch import run_fixture
from ibero.core.reproducibility import simulation_source_hash
from ibero.materials.parameters import BeamParameters, LatchParameters

ROOT = Path(__file__).resolve().parents[2]


def _fixture_job(job):
    label, options = job
    started = time.monotonic()
    row = run_fixture(**options)
    row.update(variant=label, wall_seconds=time.monotonic() - started)
    return row


def latch_report(*, workers=3):
    gates = yaml.safe_load(
        (ROOT / "validation/industrial_gates.yaml").read_text(encoding="utf-8")
    )
    gate = gates["latch"]
    report = {
        "suite": "latch_bench",
        "gate_version": gates["version"],
        "parameter_status": "provisional",
        "reference_status": "analytical_not_measured",
        "mujoco_version": mujoco.__version__,
        "simulation_source_hash": simulation_source_hash(),
        "checks": {},
        "analytical": [],
        "counterfactuals": [],
        "convergence": [],
        "workers": workers,
    }
    for n in (4, 8, 16):
        row = beam_static_test(segments=n)
        report["analytical"].append(row)
        report["checks"][f"beam_{n}"] = (
            row["stiffness_relative_error"]
            <= gate["analytical_stiffness_relative_error"]
            and row["residual_fraction"] <= gate["released_residual_fraction"]
        )
    jobs = []
    for label, beam, latch in (
        ("default", BeamParameters(), LatchParameters()),
        ("thinner", replace(BeamParameters(), thickness_m=0.0018), LatchParameters()),
        ("shallower", BeamParameters(), replace(LatchParameters(), overlap_m=0.0025)),
    ):
        for case in gate["counterfactuals"]:
            jobs.append((label, dict(case=case, beam=beam, latch=latch)))
    # 摩擦是机构参数，不用数值软化替代；附加扫描仍保留失败分母。
    for friction in (0.15, 0.6):
        for case in ("no_press", "press_pull"):
            jobs.append(
                (
                    f"friction_{friction}",
                    dict(
                        case=case, latch=replace(LatchParameters(), friction=friction)
                    ),
                )
            )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(_fixture_job, jobs):
            label, case = row["variant"], row["case"]
            report["counterfactuals"].append(row)
            final = row["final"]
            if case == "overload":
                passed = bool(final["invalid_reason"]) and not final["released"]
            else:
                passed = final["invalid_reason"] is None and final["released"] == (
                    case == "press_pull"
                )
                passed &= (
                    final["peak_penetration_m"]
                    <= gate["penetration_feature_fraction"]
                    * row["parameters"]["latch"]["overlap_m"]
                )
            report["checks"][f"{label}_{case}"] = bool(passed)
            print(label, case, "passed", passed, final["invalid_reason"], flush=True)
        jobs = [
            ("convergence", dict(segments=n, timestep=dt))
            for n, dt in (
                (8, 0.0000025),
                (12, 0.0000025),
                (16, 0.0000025),
                (16, 0.00000125),
            )
        ]
        for row in pool.map(_fixture_job, jobs):
            n, dt = row["segments"], row["timestep_s"]
            row["peak_press_force_n"] = row["final"]["peak_press_force_n"]
            row["release_displacement_m"] = abs(row["final"]["pull_position_m"])
            report["convergence"].append(row)
            report["checks"][f"convergence_valid_{n}_{dt}"] = (
                row["final"]["released"] and row["final"]["invalid_reason"] is None
            )
            print(
                "convergence",
                n,
                dt,
                row["peak_press_force_n"],
                row["final"]["invalid_reason"],
                flush=True,
            )
    for label, a, b in (
        ("resolution", report["convergence"][1], report["convergence"][2]),
        ("timestep", report["convergence"][2], report["convergence"][3]),
    ):
        for metric in ("peak_press_force_n", "release_displacement_m"):
            change = abs(a[metric] - b[metric]) / max(abs(b[metric]), 1e-12)
            report["checks"][label + "_" + metric] = (
                change <= gate["finest_resolution_relative_change"]
            )
    # 三维候选失败不降低默认梁门槛；保留各步长的原始模型和异常作为决策证据。
    report["flex_candidates"] = [
        flex_candidate_test(timestep=dt) for dt in (0.00001, 0.000001, 0.0000001)
    ]
    report["flex_diagnostics"] = [
        flex_candidate_test(timestep=1e-7, integrator="implicit"),
        flex_candidate_test(timestep=1e-7, damping_s=0, seconds=0.02),
    ]
    report["default_representation"] = "passive_segmented_beam"
    report["scope_excludes"] = [
        "robot_latch_task",
        "harness_unplug",
        "stock",
        "milling",
        "measured_calibration",
    ]
    report["passed"] = all(report["checks"].values())
    # 结束复查直接重新读取源码，确认运行期间身份没有改变。
    report["source_unchanged_during_run"] = (
        simulation_source_hash() == report["simulation_source_hash"]
    )
    report["passed"] &= report["source_unchanged_during_run"]
    return report


def main():
    from ibero.validation_cli import SUITES, run_suite

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=3)
    parser.add_argument(
        "--group", choices=["all", "default", "variants", "faults"], default="all"
    )
    parser.add_argument(
        "--scene",
        choices=["latch_release", "harness_unplug", "plate_milling"],
        default="plate_milling",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="GUI also requires two complete successful episodes",
    )
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--reference-metadata", type=Path)
    args = parser.parse_args()
    if args.full and args.suite != "gui":
        parser.error("--full is only for explicit GUI validation")
    if args.group != "all" and args.suite not in {"latch-robot", "harness-robot"}:
        parser.error("--group only selects latch/harness robot matrices")
    if bool(args.reference_csv) != bool(args.reference_metadata):
        parser.error("Reference CSV and metadata must be supplied together")
    if args.reference_csv and args.suite != "reference":
        parser.error("Reference inputs require --suite reference")
    if args.workers > (
        4 if args.suite in {"latch", "milling-variants"} else 3
    ) and args.suite in {
        "latch",
        "milling-bench",
        "milling-variants",
    }:
        parser.error("Requested workers exceed this suite's existing limit")
    if args.suite != "latch":
        raise SystemExit(run_suite(args))
    path = (
        args.output
        or ROOT
        / "artifacts/industrial_core01/latch"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        / "latch_bench_report.json"
    )
    if path.exists():
        parser.error(
            "Refusing to overwrite existing validation evidence; choose a new output"
        )
    report = latch_report(workers=args.workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print("Saved", path, "passed", report["passed"], flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
