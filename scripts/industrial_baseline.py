"""S0 基线采集：保留所有原始输出，分片只加速执行，不改变验收分母。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "artifacts" / "industrial_core01" / "baseline" / run_id
    output.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {output}", flush=True)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").strip()

    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    metadata = {
        "run_id": run_id,
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_before": git("status", "--porcelain"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "source_files": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in tracked if p
        },
        "gate_version": "C01-IND-GATES-1",
        "reference_status": "engineering_baseline_not_real_material_calibration",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def run(item):
        name, args = item
        started = time.monotonic()
        command = [sys.executable, "-X", "utf8", "-u", *args]
        print("Starting", name, flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        row = {
            "name": name,
            "command": command,
            "exit_code": result.returncode,
            "wall_seconds": time.monotonic() - started,
        }
        print("Finished", name, "exit", result.returncode, flush=True)
        return row

    jobs = [
        ("pytest", ["-m", "pytest", "-q"]),
        ("scene_handover", ["-m", "ibero.check_scene", "scenes/cable_handover"]),
        ("scene_stretch", ["-m", "ibero.check_scene", "scenes/cable_stretch"]),
        (
            "physics",
            [
                "-m",
                "ibero.verify_core01",
                "--seeds",
                "3",
                "--output",
                str(output / "physics.json"),
            ],
        ),
    ]
    for start in (0, 5, 10, 15):
        jobs.append(
            (
                f"calibration_{start}",
                [
                    "-m",
                    "ibero.calibrate",
                    "--stability-seeds",
                    "5",
                    "--baseline-seeds",
                    "5",
                    "--stability-seed-start",
                    str(start),
                    "--baseline-seed-start",
                    str(start),
                    "--output",
                    str(output / f"calibration_{start}.json"),
                ],
            )
        )
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, jobs))
    # GUI 是显式验收；数值批处理结束后运行，避免争抢资源误触 300 s 超时。
    results.append(
        run(
            (
                "viewer",
                ["-m", "ibero.verify_viewer", "--output", str(output / "viewer.json")],
            )
        )
    )
    results.append(
        run(
            (
                "trace_render",
                [
                    "-m",
                    "ibero.demo",
                    "--env",
                    "cable_stretch",
                    "--save-trace",
                    str(output / "stretch.npz"),
                    "--save-multiview",
                    str(output / "stretch.png"),
                    "--save-manifest",
                    str(output / "stretch_manifest.json"),
                ],
            )
        )
    )
    results.append(run(("packages", ["-m", "pip", "freeze"])))
    calibrations = [
        json.loads((output / f"calibration_{start}.json").read_text(encoding="utf-8"))
        for start in (0, 5, 10, 15)
    ]
    stability = [r for c in calibrations for r in c["stability"]]
    handover = [r for c in calibrations for r in c["scripted_baseline"]]
    # 单个 5-seed 分片可能不满足局部 90%；验收依据冻结的完整 20-seed 集。
    report = {
        "metadata": metadata,
        "commands": results,
        "stability": stability,
        "handover": handover,
        "ft_checks": [c["wrist_ft_static_load"] for c in calibrations],
        "stability_passes": sum(r["stable"] for r in stability),
        "handover_passes": sum(r["success"] for r in handover),
    }
    report["passed"] = (
        len(stability) == len(handover) == 20
        and sorted(r["seed"] for r in stability) == list(range(20))
        and sorted(r["seed"] for r in handover) == list(range(20))
        and report["stability_passes"] == 20
        and report["handover_passes"] >= 18
        and all(r["passed"] for r in report["ft_checks"])
        and all(
            r["exit_code"] == 0
            for r in results
            if not r["name"].startswith("calibration_")
        )
    )
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "Baseline passed:",
        report["passed"],
        "handover:",
        report["handover_passes"],
        "/20",
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
