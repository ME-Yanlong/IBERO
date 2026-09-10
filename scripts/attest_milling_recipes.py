"""只读补核旧报告的菜谱身份；不重写原始结果，不运行或续跑物理。"""

import argparse
import hashlib
import json
from pathlib import Path

from ibero.benches.milling import MillingFixture
from ibero.core.reproducibility import simulation_source_hash
from ibero.core.scene_loader import SceneLoader
from ibero.core.stock_trace import StockTrace


def attest(report_path, scene_path):
    report_path, scene_path = Path(report_path), Path(scene_path)
    raw_bytes = report_path.read_bytes()
    report = json.loads(raw_bytes)
    scene = SceneLoader().validate(scene_path)
    if scene.config["kind"] == "plate_milling":
        from ibero.envs.plate_milling import PlateMillingEnv

        env = PlateMillingEnv.from_scene(scene_path, shape=report["shape"])
    elif scene.config["kind"] == "milling_bench":
        env = MillingFixture.from_scene(scene_path)
    else:
        raise ValueError("Only concrete milling recipes can be attested")
    env.reset(seed=report["seed"])
    # 必须在原冻结源码上执行；模型生成身份包含完整菜谱，不能仅比较力系数。
    if report["manifest"] != env.manifest() or report.get("frozen_source") is not True:
        raise ValueError("Original report does not match the frozen recipe identity")
    before_qpos = env.data.qpos.copy()
    before_hash = env.stock.state_hash()
    trace_path = report_path.parent / "trace.npz"
    trace = StockTrace(env).load(trace_path)
    if (
        not (env.data.qpos == before_qpos).all()
        or env.stock.state_hash() != before_hash
    ):
        raise RuntimeError("Read-only trace validation unexpectedly modified state")
    simulation_source_hash.cache_clear()
    if simulation_source_hash() != report["manifest"]["source_hash"]:
        raise ValueError("Source changed during recipe attestation")
    return {
        "scope": "post_run_recipe_identity_attestation_not_new_physics_run",
        "passed": True,
        "report_path": str(report_path.resolve()),
        "report_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "trace_frames_validated": len(trace.states),
        "manifest": report["manifest"],
        "scene_path": str(scene_path.resolve()),
        "scene_config": scene.config,
        "constraints": scene.constraints,
        "original_result_unchanged": True,
        "physical_steps_executed": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", nargs=2, action="append", required=True, metavar=("REPORT", "SCENE")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace existing evidence")
    rows = []
    for report_path, scene_path in args.case:
        try:
            row = attest(report_path, scene_path)
        except Exception as error:
            row = dict(
                passed=False,
                report_path=str(Path(report_path).resolve()),
                exception=f"{type(error).__name__}: {error}",
            )
        rows.append(row)
        print(row["report_path"], row["passed"], row.get("exception"), flush=True)
    result = dict(
        scope="read_only_recipe_attestations",
        results=rows,
        passed=all(row["passed"] for row in rows),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
