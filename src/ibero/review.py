"""审阅状态与轨迹存档；回放只恢复记录的物理状态，不重新执行策略。"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
import mujoco
import numpy as np
from ibero.core.reproducibility import simulation_source_hash
from ibero.robots.g1_upperbody_2f85 import vendored_robot_asset_hash


@dataclass
class ReviewControls:
    run: threading.Event = field(default_factory=threading.Event)
    pause: threading.Event = field(default_factory=threading.Event)
    step: threading.Event = field(default_factory=threading.Event)
    replay: threading.Event = field(default_factory=threading.Event)
    back: threading.Event = field(default_factory=threading.Event)

    def key(self, code):
        # P 暂停；N 单步；R 回放；B 回看上一帧；Enter 新一轮。
        mapping = {
            13: self.run,
            257: self.run,
            335: self.run,
            80: self.pause,
            78: self.step,
            82: self.replay,
            66: self.back,
        }
        if code in mapping:
            mapping[code].set()


class Trace:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION

    def __init__(self, env):
        self.env = env
        self.states = []
        self.infos = []

    def append(self, info):
        state = np.empty(mujoco.mj_stateSize(self.env.model, self.spec))
        mujoco.mj_getState(self.env.model, self.env.data, state, self.spec)
        self.states.append(state)
        self.infos.append(json.loads(json.dumps(info)))

    def restore(self, index):
        mujoco.mj_setState(self.env.model, self.env.data, self.states[index], self.spec)
        mujoco.mj_forward(self.env.model, self.env.data)
        # 回放不是可恢复训练 checkpoint：策略/任务内部计时器没有倒带。
        # 禁止直接在回放状态继续积分；Enter/reset 才能开始新物理回合。
        self.env._replay_restored = True
        return {**self.infos[index], "replay": True}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scene_hash": self.env.scene.scene_hash,
            "seed": int(self.env.np_random_seed),
            "source_hash": simulation_source_hash(),
            "asset_hash": vendored_robot_asset_hash(),
            "mujoco_version": mujoco.__version__,
            "infos": self.infos,
        }
        # 用文件句柄保持用户指定文件名；allow_pickle=False 的纯数组格式可安全读取。
        with path.open("wb") as stream:
            np.savez_compressed(
                stream, states=np.stack(self.states), metadata=json.dumps(payload)
            )

    def load(self, path):
        with np.load(path, allow_pickle=False) as stored:
            meta = json.loads(str(stored["metadata"]))
            if (
                meta["scene_hash"] != self.env.scene.scene_hash
                or meta["mujoco_version"] != mujoco.__version__
                or meta.get("source_hash") != simulation_source_hash()
                or meta.get("asset_hash") != vendored_robot_asset_hash()
            ):
                raise ValueError(
                    "Replay scene/source/asset hash or MuJoCo version mismatch"
                )
            self.env.reset(seed=meta["seed"])
            self.states = list(stored["states"])
            self.infos = meta["infos"]
            if (
                not self.states
                or len(self.states) != len(self.infos)
                or any(
                    s.shape != (mujoco.mj_stateSize(self.env.model, self.spec),)
                    or not np.isfinite(s).all()
                    for s in self.states
                )
            ):
                raise ValueError("Invalid replay state dimensions")
        return self


def run_review(env, steps, seed, use_dashboard, playback_rate):
    """统一事件循环；暂停只暂停积分，UI 始终响应；回放后 Enter 新建回合。"""
    import mujoco.viewer
    import ibero
    from ibero.baselines import CableHandoverScript
    from ibero.control.tension import CableStretchScript
    from ibero.demo import scripted_action
    from ibero.multiview import try_create_dashboard

    is_core = isinstance(env, ibero.CableHandoverEnv)

    def script():
        return (
            (CableStretchScript() if env.dual_end else CableHandoverScript())
            if is_core
            else None
        )

    observation, info = env.reset(seed=seed)
    policy = script()
    trace = Trace(env) if is_core else None
    if trace:
        trace.append(info)
    controls = ReviewControls()
    dashboard = None
    running = False
    paused = False
    replaying = False
    finished = False
    index = 0
    count = 0
    if getattr(env, "_replay_path", None):
        trace.load(env._replay_path)
        info = trace.restore(0)
        replaying = True
    try:
        with mujoco.viewer.launch_passive(
            env.model, env.data, key_callback=controls.key
        ) as handle:
            if is_core:
                handle.cam.lookat = env.data.site_xpos[
                    [env.handles.left_pinch_site_id, env.handles.right_pinch_site_id]
                ].mean(axis=0)
                handle.cam.distance = 0.9
                handle.cam.azimuth = -135
                handle.cam.elevation = -20
            handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            handle.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            if use_dashboard and is_core:
                dashboard = try_create_dashboard(env, controls.run, controls)
            print(
                "Enter: new episode | P: pause | N: single step | R: replay | B: previous frame"
            )
            dirty = True
            next_step = time.perf_counter()
            while handle.is_running():
                if controls.run.is_set():
                    controls.run.clear()
                    observation, info = env.reset(seed=seed)
                    policy = script()
                    trace = Trace(env) if is_core else None
                    if trace:
                        trace.append(info)
                    running = True
                    paused = False
                    replaying = False
                    finished = False
                    count = 0
                    dirty = True
                if controls.pause.is_set():
                    controls.pause.clear()
                    paused = not paused
                if controls.replay.is_set():
                    controls.replay.clear()
                    if trace and len(trace.states) > 1:
                        running = False
                        replaying = True
                        paused = False
                        index = 0
                        info = trace.restore(0)
                        dirty = True
                if controls.back.is_set():
                    controls.back.clear()
                    if trace and trace.states:
                        if not replaying:
                            index = len(trace.states) - 1
                        running = False
                        replaying = True
                        paused = True
                        index = max(0, index - 1)
                        info = trace.restore(index)
                        dirty = True
                single = controls.step.is_set()
                if single:
                    controls.step.clear()
                    paused = True
                    if not running and not replaying and not finished:
                        running = True
                if (single or (not paused and time.perf_counter() >= next_step)) and (
                    running or replaying
                ):
                    if replaying:
                        index = min(index + 1, len(trace.states) - 1)
                        info = trace.restore(index)
                        if index == len(trace.states) - 1:
                            paused = True
                    else:
                        action = (
                            policy.action(env)
                            if policy
                            else scripted_action(observation)
                        )
                        observation, _, terminated, truncated, info = env.step(action)
                        if policy:
                            info["controller_phase"] = policy.phase
                        if trace:
                            trace.append(info)
                        count += 1
                        if terminated or truncated or count >= steps:
                            running = False
                            finished = True
                            if trace and getattr(env, "_save_trace_path", None):
                                trace.save(env._save_trace_path)
                            print(
                                "Episode complete:",
                                info.get("stage"),
                                info.get("is_success"),
                                info.get("failure_reason"),
                            )
                    next_step = time.perf_counter() + env.control_dt / playback_rate
                    dirty = True
                status = (
                    "回放"
                    if replaying
                    else "已完成"
                    if finished
                    else "运行中"
                    if running
                    else "就绪"
                ) + (" / 暂停" if paused else "")
                handle.sync()
                if dashboard and not dashboard.present(info, status, force=dirty):
                    dashboard.close()
                    dashboard = None
                dirty = False
                time.sleep(0.005)
    finally:
        if dashboard:
            dashboard.close()
    return info
