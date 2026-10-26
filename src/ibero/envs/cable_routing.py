"""立式板柔性线束布线：真实接触运行，场景任务独立评估。"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ibero.control.resolved_rate import BimanualResolvedRateController
from ibero.core.audit import SafetyAuditor
from ibero.core.routing_builder import compile_routing
from ibero.core.scene_loader import SceneLoader
from ibero.core.task_api import TaskState
from ibero.core.task_loading import load_task_spec
from ibero.materials.mechanics import CableMechanics
from ibero.mechanisms.cable_clip import CableClipObserver


DEFAULT_SCENE_PATH = Path(__file__).resolve().parents[3] / "scenes" / "cable_clip_insert"


class CableRoutingEnv(gym.Env):
    """单夹和多夹共用的 G1/Flex/线夹环境，不对材料执行位姿脚本。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, *, scene_path=DEFAULT_SCENE_PATH):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Unsupported render mode")
        self.render_mode = render_mode
        self.scene = SceneLoader().validate(scene_path)
        if self.scene.config.get("schema_version") != "ibero.routing/v0.2":
            raise ValueError("CableRoutingEnv requires a routing scene")
        self.compiled = compile_routing(self.scene.config)
        self.model = self.compiled.model
        self.handles = self.compiled.robot
        self.data = mujoco.MjData(self.model)
        self.task = load_task_spec(self.scene)
        self.mechanics = CableMechanics(self.model, self.compiled.cable)
        self.control_dt = 1.0 / self.scene.config["physics"]["control_hz"]
        self.physics_substeps = round(self.control_dt / self.model.opt.timestep)
        self.max_episode_steps = self.scene.config["physics"]["max_episode_steps"]
        self.controller = BimanualResolvedRateController(self.model, self.handles, self.control_dt)
        cable_limits = self.scene.constraints["cable"]
        self.auditor = SafetyAuditor(
            tension_limit_n=float(cable_limits["tension_limit_n"]),
            tension_limit_duration_s=float(cable_limits["limit_duration_s"]),
        )
        penetration_limit = float(self.scene.constraints["clip"]["penetration_limit_m"])
        self.clip_observers = tuple(
            CableClipObserver(
                self.model, name, self.compiled.clip_parameters,
                penetration_limit_m=penetration_limit,
            )
            for name in self.compiled.clip_names
        )
        self.action_space = spaces.Box(-1, 1, shape=(14,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "pinch_m": spaces.Box(-10, 10, shape=(2, 3), dtype=np.float32),
            "cable_m": spaces.Box(-10, 10, shape=(self.compiled.cable.segments + 1, 3), dtype=np.float32),
            "tension_n": spaces.Box(0, 1e5, shape=(1,), dtype=np.float32),
            "grasp": spaces.MultiBinary(2),
        })
        self._renderer = None
        self._render_camera = mujoco.MjvCamera()
        self._render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._render_camera.lookat[:] = [0.49, 0.0, 0.85]
        self._render_camera.distance = 0.95
        self._render_camera.azimuth = 60
        self._render_camera.elevation = -22
        self._step_count = 0
        self._grasp_history = {side: deque(maxlen=3) for side in ("left", "right")}
        self._last_info = {}
        self._done = False
        self._replay_restored = False

    @property
    def last_info(self):
        return self._last_info

    def _points(self):
        f = self.mechanics.flex_id
        first = self.model.flex_vertadr[f]
        return self.data.flexvert_xpos[first:first + self.model.flex_vertnum[f]].copy()

    def _flex_pad_contacts(self, side):
        pads = self.handles.left_pad_geom_ids if side == "left" else self.handles.right_pad_geom_ids
        touched = set()
        elems = set()
        cable_flex_id = self.mechanics.flex_id
        for contact in self.data.contact[:self.data.ncon]:
            # Flex 与刚体接触时 geom 一侧为 -1。用 flex/elem 编号识别材料，
            # 不能套用旧端头的双正 geom ID 检查。
            if cable_flex_id not in contact.flex:
                continue
            geom = int(contact.geom1 if contact.geom1 >= 0 else contact.geom2)
            if geom not in pads:
                continue
            touched.add(geom)
            elems.update(int(elem) for elem in contact.elem if elem >= 0)
        both_pads = bool(touched.intersection(pads[:2]) and touched.intersection(pads[2:]))
        return both_pads, elems

    def _update_grasps(self):
        for side in ("left", "right"):
            both, elements = self._flex_pad_contacts(side)
            # 起点和终点附近的材料段分配给不同双手，避免两手碰到同一小段
            # 就误报两处真实抓持。局部弧长跟踪也用于滑移检测。
            n = self.compiled.cable.segments
            if side == "left":
                assigned = {i for i in elements if i < max(3, n // 3)}
            else:
                assigned = {i for i in elements if i >= n - max(3, n // 3)}
            pinch = self.handles.left_anchor_site_id if side == "left" else self.handles.right_anchor_site_id
            pts = self._points()
            if assigned:
                index = min(assigned, key=lambda i: np.linalg.norm(pts[min(i, n)] - self.data.site_xpos[pinch]))
                point = (pts[index] + pts[index + 1]) / 2
                rotation = self.data.site_xmat[pinch].reshape(3, 3)
                relative = rotation.T @ (point - self.data.site_xpos[pinch])
            else:
                relative = np.array([np.inf, np.inf, np.inf])
            self._grasp_history[side].append((both and bool(assigned), relative))

    def _grasped(self, side):
        samples = self._grasp_history[side]
        return bool(
            len(samples) == 3
            and all(item[0] for item in samples)
            and max(np.linalg.norm(item[1] - samples[0][1]) for item in samples) < 0.008
        )

    def _hands_clear(self):
        points = self._points()
        if any(self._grasped(side) or self._flex_pad_contacts(side)[0] for side in ("left", "right")):
            return False
        for side in ("left", "right"):
            pinch = self.handles.left_anchor_site_id if side == "left" else self.handles.right_anchor_site_id
            if np.min(np.linalg.norm(points - self.data.site_xpos[pinch], axis=1)) < 0.04:
                return False
        return True

    def _robot_collision(self):
        left = self.handles.left_robot_geom_ids
        right = self.handles.right_robot_geom_ids
        for contact in self.data.contact[:self.data.ncon]:
            if contact.geom1 >= 0 and contact.geom2 >= 0:
                if (int(contact.geom1) in left and int(contact.geom2) in right) or (int(contact.geom2) in left and int(contact.geom1) in right):
                    return True
        return False

    def _check_numerics(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Routing state contains non-finite values")
        for warning in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC):
            if self.data.warning[warning].number:
                raise FloatingPointError("MuJoCo reported unstable routing dynamics")

    def _task_state(self):
        material = self.mechanics.state(self.data)
        points = self._points()
        routing = {
            "cable_points_m": points,
            "clips": [observer.observe(self.data) for observer in self.clip_observers],
            "dt_s": self.control_dt,
            "cable_speed_m_s": float(self._cable_speed),
            "hands_clear": self._hands_clear(),
            "minimum_radius_m": material["minimum_radius_m"],
            "invalid_reason": "bend_radius_alarm" if self._bend_violation else None,
        }
        return TaskState(
            body_positions={}, site_positions={},
            gripper_openings={},
            grasps={"left_gripper": self._grasped("left"), "right_gripper": self._grasped("right")},
            contacts=frozenset(),
            cable_tension_n=material["tension_n"],
            elapsed_steps=self._step_count,
            max_episode_steps=self.max_episode_steps,
            extra={"routing": routing},
        )

    def _observation(self):
        points = self._points()
        return {
            "pinch_m": np.asarray(self.data.site_xpos[[self.handles.left_anchor_site_id, self.handles.right_anchor_site_id]], dtype=np.float32).copy(),
            "cable_m": points.astype(np.float32),
            "tension_n": np.asarray([self.mechanics.tension(self.data)], dtype=np.float32),
            "grasp": np.asarray([self._grasped("left"), self._grasped("right")], dtype=np.int8),
        }

    def _info(self, result):
        return {
            "stage": result.stage,
            "is_success": bool(result.success),
            "failure_reason": result.failure_reason,
            "metrics": dict(result.metrics),
            "audit": self.auditor.manifest(base={})["audit"],
            "scene_id": self.scene.config["id"],
            "scene_hash": self.scene.scene_hash,
            "simulation_time_s": float(self.data.time),
            "time_s": float(self.data.time),
            "success": bool(result.success),
            "initial_cable_jitter_m": self._initial_cable_jitter.tolist(),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._replay_restored = False
        self._done = False
        mujoco.mj_resetData(self.model, self.data)
        limits = np.asarray(
            self.scene.config["initialization"].get("cable_translation_jitter_m", [0, 0, 0]),
            dtype=float,
        )
        self._initial_cable_jitter = self.np_random.uniform(-limits, limits)
        f = self.mechanics.flex_id
        first = self.model.flex_vertadr[f]
        count = self.model.flex_vertnum[f]
        for body_id in self.model.flex_vertbodyid[first:first + count]:
            joint = self.model.body_jntadr[body_id]
            for axis in range(3):
                self.data.qpos[self.model.jnt_qposadr[joint + axis]] += self._initial_cable_jitter[axis]
        self.controller.reset(self.data)
        self.data.ctrl[self.handles.gripper_actuator_ids] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for values in self._grasp_history.values():
            values.clear()
        self.auditor.reset()
        self.task.reset()
        self._step_count = 0
        self._cable_speed = 0.0
        self._bend_violation = False
        # 短暂物理沉降从供料架开始；保持手张开，不能在 reset 中预抓。
        for _ in range(round(0.06 / self.model.opt.timestep)):
            self.controller.hold_pose(self.data)
            self.mechanics.apply(self.data)
            mujoco.mj_step(self.model, self.data)
            self._check_numerics()
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._previous_points = self._points()
        self._update_grasps()
        result = self.task.evaluate(self._task_state(), self.auditor.snapshot())
        self._last_info = self._info(result)
        return self._observation(), self._last_info

    def step(self, action):
        if self._replay_restored:
            raise RuntimeError("Replay is read-only; reset before stepping")
        if self._done:
            raise RuntimeError("Episode ended; reset before stepping")
        self.controller.apply(self.data, action)
        for _ in range(self.physics_substeps):
            self.mechanics.apply(self.data)
            mujoco.mj_step(self.model, self.data)
            self._check_numerics()
            self.auditor.observe(
                cable_tension_n=self.mechanics.tension(self.data),
                self_collision=self._robot_collision(),
                dt=float(self.model.opt.timestep),
            )
        mujoco.mj_forward(self.model, self.data)
        self._step_count += 1
        material = self.mechanics.state(self.data)
        if material["bend_alarm"] and not self._bend_violation:
            self._bend_violation = True
            self.auditor.record(
                "bend_radius_alarm", minimum_radius_m=material["minimum_radius_m"]
            )
        points = self._points()
        node_speeds = np.linalg.norm(points - self._previous_points, axis=1) / self.control_dt
        self._global_cable_speed = float(np.max(node_speeds))
        # 固定质量针对夹口附近材料保持；松开的自由尾端允许继续摆动。
        # 全线峰值仍单独报告，不让尾端摆动替代夹口脱扣/稳定判据。
        local_speeds = []
        for clip in self.scene.config["workcell"]["clips"]:
            origin = np.asarray(clip["origin_m"], dtype=float)
            seat_center = origin + np.array([-self.compiled.clip_parameters.seat_depth_m, 0, 0])
            near = int(np.argmin(np.linalg.norm(points - seat_center, axis=1)))
            local_speeds.append(float(np.max(node_speeds[max(near - 1, 0):min(near + 2, len(points))])))
        self._cable_speed = max(local_speeds)
        self._previous_points = points
        self._update_grasps()
        result = self.task.evaluate(self._task_state(), self.auditor.snapshot())
        self._done = bool(result.terminated or result.truncated)
        self._last_info = self._info(result)
        return self._observation(), float(result.reward), bool(result.terminated), bool(result.truncated), self._last_info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self._render_camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
