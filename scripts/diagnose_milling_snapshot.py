"""失败材料帧的独立载荷求解诊断：不续跑轨迹、不提交材料、不推进 live 状态。"""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.core.stock_trace import StockTrace
from ibero.processes.implicit_wrench import ImplicitWrenchCoupling
from ibero.processes.prepared_milling_wrench import PreparedMillingWrench


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, default=Path("scenes/plate_milling"))
    p.add_argument("--shape", choices=["slot", "pocket", "through_hole"], required=True)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--engine-tolerance", type=float)
    p.add_argument(
        "--candidate-solver",
        type=Path,
        help="显式可信的本项目 Python 候选求解器；不是轨迹内容",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.case / "report.json").read_text(encoding="utf-8"))
    env = PlateMillingEnv(args.scene, shape=args.shape)
    trace = StockTrace(env).load(args.case / "trace.npz")
    trace.restore(len(trace.states) - 1)
    pose, velocity, _, _ = env.process.kinematics(env.data)
    trial_model = copy.deepcopy(env.model)
    if args.engine_tolerance is not None:
        if not np.isfinite(args.engine_tolerance) or args.engine_tolerance <= 0:
            p.error("Positive finite engine tolerance required")
        trial_model.opt.tolerance = args.engine_tolerance
    trial_data = mujoco.MjData(trial_model)
    mujoco.mj_copyData(trial_data, trial_model, env.data)
    lengths, centers, fraction, pending = env.process._engagement(pose, velocity)
    prepared = PreparedMillingWrench(
        env.coefficients,
        radius_m=env.tool.radius_m,
        teeth=env.limits.teeth,
        lengths_m=lengths,
        centroids_m=centers,
        face_fraction=fraction,
        edge_transition_chip_m=env.process.edge_transition_chip_m,
        center_cutting=env.limits.center_cutting,
    )

    def wrench_at(u):
        tool = prepared(pose.rotation.T @ u[:3], u[3] * 30 / np.pi)
        return np.r_[pose.rotation @ tool[:3], pose.rotation @ tool[3:]]

    rows = []
    solver_type = ImplicitWrenchCoupling
    if args.candidate_solver:
        spec = importlib.util.spec_from_file_location(
            "candidate_implicit_solver", args.candidate_solver
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solver_type = module.ImplicitWrenchCoupling
    # 缓存的上一步载荷未纳入可续跑状态，显式比较零/上一控制帧两个初猜，而非冒称精确续跑。
    previous = report["rows"][-2]
    anchors = {
        "zero": np.zeros(6),
        "previous_control_frame": np.r_[
            previous["force_world_n"], previous["torque_world_nm"]
        ],
    }
    qpos, qvel, stock_hash = (
        env.data.qpos.copy(),
        env.data.qvel.copy(),
        env.stock.state_hash(),
    )
    for name, anchor in anchors.items():
        coupling = solver_type(
            trial_model,
            tool_body=env.process.tool_body,
            tip_site=env.process.tip_site,
            spindle_dof=env.process.coupling.spindle_dof,
        )
        coupling.last_wrench[:] = anchor
        row = {"anchor": name}
        started = time.perf_counter()
        try:
            wrench, response, residual = coupling.solve(
                trial_data,
                trial_data.qfrc_applied,
                trial_data.xfrc_applied,
                pose.position,
                wrench_at,
            )
            row.update(
                wrench=wrench.tolist(), response=response.tolist(), residual=residual
            )
        except Exception as error:
            row["exception"] = f"{type(error).__name__}: {error}"
        row.update(
            diagnostics=coupling.last_diagnostics,
            wall_seconds=time.perf_counter() - started,
        )
        rows.append(row)
    result = dict(
        scope="fixed_snapshot_coupling_diagnostic_not_resumed_episode",
        source_case=str(args.case.resolve()),
        engine_tolerance_original=float(env.model.opt.tolerance),
        engine_tolerance_trial=float(trial_model.opt.tolerance),
        manifest=env.manifest(),
        candidate_solver=None
        if args.candidate_solver is None
        else {
            "path": str(args.candidate_solver.resolve()),
            "sha256": hashlib.sha256(args.candidate_solver.read_bytes()).hexdigest(),
        },
        pending=pending,
        face_fraction=fraction,
        rows=rows,
        state_unchanged=bool(
            np.array_equal(qpos, env.data.qpos)
            and np.array_equal(qvel, env.data.qvel)
            and stock_hash == env.stock.state_hash()
            and env._replay_restored
        ),
    )
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
