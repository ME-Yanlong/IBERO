"""独立进程记录重构前后物理摘要；不绕过正式轨迹的源码身份检查。"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from ibero.benches.milling import MillingFixture
from ibero.control.milling_bench import MillingBenchScript, bench_target
from ibero.core.reproducibility import simulation_source_hash
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.review import Trace


def model_arrays(model):
    """只读所有公开 ndarray 字段（含原资产能力），不序列化指针/对象 repr。"""
    result = {}
    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            result[name] = dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest(),
            )
    return result


def snapshot(scene, *, robot, frames):
    env = PlateMillingEnv(scene) if robot else MillingFixture.from_scene(scene)
    target = env.target if robot else bench_target("through_hole")
    env.reset(seed=0)
    policy = MillingBenchScript(env, target)
    source = simulation_source_hash()
    model_before = model_arrays(env.model)
    state = np.empty(mujoco.mj_stateSize(env.model, Trace.spec))
    state_digest, info_digest = hashlib.sha256(), hashlib.sha256()
    mujoco.mj_getState(env.model, env.data, state, Trace.spec)
    initial_state = hashlib.sha256(state.tobytes()).hexdigest()
    started = time.perf_counter()
    completed = 0
    stride = round(
        1 / env.scene.config["physics"]["control_hz"] / env.model.opt.timestep
    )
    for _ in range(frames):
        command, rpm = policy.command(env)
        info = env.step(command, rpm, substeps=stride)
        mujoco.mj_getState(env.model, env.data, state, Trace.spec)
        state_digest.update(state.tobytes())
        info_digest.update(json.dumps(info, sort_keys=True, allow_nan=False).encode())
        completed += 1
        if env._done:
            break
    return dict(
        scope="read_only_digest_evidence_not_full_task_or_cross_source_trace_replay",
        source_hash=source,
        frozen_source=source == simulation_source_hash(),
        robot=robot,
        requested_frames=frames,
        completed_frames=completed,
        initial_model_arrays=model_before,
        initial_state_digest=initial_state,
        rollout_state_digest=state_digest.hexdigest(),
        rollout_info_digest=info_digest.hexdigest(),
        material_hash=env.stock.state_hash(),
        material_events=len(env.stock.events),
        final_model_arrays=model_arrays(env.model),
        timestep_s=float(env.model.opt.timestep),
        gravity_m_s2=env.model.opt.gravity.tolist(),
        integrator=int(env.model.opt.integrator),
        sim_seconds=float(env.data.time),
        wall_seconds=time.perf_counter() - started,
        invalid_reason=env.process.invalid_reason,
        # 这只是已执行前缀的完整性，零删料不能用于宣称加工等价。
        prefix_complete=completed == frames and not env._done,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--robot", action="store_true")
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.frames <= 3000:
        parser.error("frames must be in [1, 3000]")
    if args.output.exists():
        parser.error("Refusing to replace evidence")
    result = snapshot(args.scene, robot=args.robot, frames=args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output, result["prefix_complete"], result["material_events"], flush=True)
    raise SystemExit(0 if result["prefix_complete"] and result["frozen_source"] else 1)


if __name__ == "__main__":
    main()
