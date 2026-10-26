"""源码检出环境中的工业验收编排；只调现有检查，不创造新的物理通过标准。"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ibero.core.reproducibility import simulation_source_hash

ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    "quick",
    "latch",
    "stock-geometry",
    "stock-collision",
    "milling-loads",
    "milling-bench",
    "latch-robot",
    "harness-robot",
    "milling-robot",
    "milling-variants",
    "gui",
    "reference",
)


def suite_command(args, evidence):
    """仅 GUI 分组显式调用桌面；单测和数值长测不会暗中附带开窗检查。"""
    command = [sys.executable, "-X", "utf8"]
    if args.suite == "quick":
        return command + [
            "-m",
            "pytest",
            "-q",
            "tests",
            f"--junitxml={evidence / 'tests.xml'}",
        ], None
    script = {
        "stock-geometry": "verify_stock_geometry.py",
        "stock-collision": "verify_stock_collision.py",
        "milling-loads": "verify_milling_loads.py",
        "milling-bench": "verify_milling_process.py",
        "latch-robot": "verify_latch_robot.py",
        "harness-robot": "verify_latch_robot.py",
        "milling-robot": "verify_robot_milling.py",
        "milling-variants": "verify_robot_milling_variants.py",
        "gui": "check_industrial_viewer.py",
    }[args.suite]
    payload = evidence / "result"
    command += [str(ROOT / "scripts" / script), "--output", str(payload)]
    if args.suite in {"latch-robot", "harness-robot"}:
        command += [
            "--scene",
            "latch_release" if args.suite == "latch-robot" else "harness_unplug",
            "--group",
            args.group,
            "--workers",
            str(args.workers),
        ]
    elif args.suite in {"milling-bench", "milling-robot", "milling-variants"}:
        command += ["--workers", str(args.workers)]
        if args.suite == "milling-bench":
            command += ["--scene", str(ROOT / "scenes/milling_bench"), "--shape", "all"]
    elif args.suite == "gui":
        command += ["--scene", args.scene]
        if args.full:
            command += ["--full"]
    return command, payload / "report.json"


def reference_report(csv_path, metadata_path):
    """导入合法性不是材料验证；无实测时不输出 passed=true。"""
    if csv_path is None:
        return dict(
            status="not_available",
            passed=None,
            calibrated=False,
            reason="No external curve supplied; analytical engineering gates remain separate",
        )
    from ibero.materials.references import load_force_curve

    values, metadata = load_force_curve(csv_path, metadata_path)
    return dict(
        status="imported_not_calibrated",
        passed=None,
        calibrated=False,
        metadata=metadata,
        sample_count=len(values),
        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        note="Import validates format/provenance only, not the truth of a measured label or model accuracy",
    )


def run_suite(args):
    path = (
        args.output
        or ROOT
        / "artifacts/industrial_core01/verification"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        / "report.json"
    )
    path = path.resolve()
    evidence = path.parent / (path.stem + "_evidence")
    if path.exists() or evidence.exists():
        raise ValueError("Refusing to overwrite existing verification evidence")
    evidence.mkdir(parents=True)
    source = simulation_source_hash()
    report = dict(
        suite=args.suite,
        scope="selected_suite_only_not_automatic_G9",
        passed=False,
        source_hash=source,
        calibration_status="provisional",
        evidence=str(evidence),
    )
    started = time.perf_counter()
    exit_code = 1
    try:
        if args.suite == "reference":
            report.update(reference_report(args.reference_csv, args.reference_metadata))
            exit_code = 3 if report["status"] == "not_available" else 0
        else:
            command, result_path = suite_command(args, evidence)
            report["command"] = command
            child_env = os.environ.copy()
            child_env["PYTHONPATH"] = str(ROOT / "src")
            names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            if not any(name in child_env for name in names):
                child_env.update({name: "1" for name in names})
            report["numeric_thread_environment"] = {
                name: child_env.get(name) for name in names
            }
            with (evidence / "command.log").open("w", encoding="utf-8") as log:
                print("Running", args.suite, "log:", log.name, flush=True)
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            report["exit_code"] = result.returncode
            report["passed"] = result.returncode == 0
            if result_path is not None:
                child = json.loads(result_path.read_text(encoding="utf-8"))
                report["child_report"] = str(result_path)
                report["child_scope"] = child.get("scope", args.suite)
                report["passed"] &= child.get("passed") is True
            exit_code = 0 if report["passed"] else 1
    except Exception as error:
        report["exception"] = f"{type(error).__name__}: {error}"
        report["passed"] = False
        exit_code = 1
    report["frozen_source"] = source == simulation_source_hash()
    if not report["frozen_source"]:
        report["passed"], exit_code = False, 1
    report["wall_seconds"] = time.perf_counter() - started
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print("Saved", path, "passed", report["passed"], flush=True)
    return exit_code
