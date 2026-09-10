"""Core-0.1 contact-driven G1 cable-handover environment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ibero.control import BimanualResolvedRateController
from ibero.core.audit import SafetyAuditor
from ibero.core.scene_compiler import SceneCompiler
from ibero.core.scene_loader import SceneLoader, ValidatedScene
from ibero.core.sensors import wrist_wrench
from ibero.core.task_api import TaskSpec, TaskState
from ibero.materials.mechanics import CableMechanics
from ibero.robots.g1_upperbody_2f85 import (
    HANDOVER_ARM_QPOS,
    vendored_robot_asset_hash,
)


DEFAULT_SCENE_PATH = Path(__file__).resolve().parents[3] / "scenes" / "cable_handover"


def _load_task_spec(scene: ValidatedScene) -> TaskSpec:
    module_name = (
        f"ibero_scene_{hashlib.sha256(str(scene.task_path).encode()).hexdigest()[:12]}"
    )
    module_spec = importlib.util.spec_from_file_location(module_name, scene.task_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not import task specification {scene.task_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    task = getattr(module, "TASK_SPEC", None)
    if not isinstance(task, TaskSpec):
        raise TypeError("task_spec.py must expose TASK_SPEC derived from TaskSpec")
    return task


class CableHandoverEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Transfer a physically contacted cable terminal between two 2F-85s."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        scene_path: str | Path = DEFAULT_SCENE_PATH,
    ) -> None:
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.scene = SceneLoader().validate(scene_path)
        self.end_effector_type = str(self.scene.config["robot"]["end_effector"]["type"])
        if self.end_effector_type != "robotiq_2f85":
            # 当前 G1 快照只有不可驱动的 rubber-hand 外观，缺少手指关节、
            # 指尖碰撞与力控接口。因此宁可明确拒绝 task，也不能把视觉手
            # 误报为已通过真实接触 handover 的灵巧手。
            raise NotImplementedError(
                "CableHandover-v0 currently requires robot.end_effector.type: robotiq_2f85; "
                "g1_native_hand is an inspection-only visual configuration in the vendored G1 asset."
            )
        self.task = _load_task_spec(self.scene)
        compiled = SceneCompiler().compile(self.scene)
        self.model = compiled.model
        self.handles = compiled.handles
        self.cable_params = compiled.cable_parameters
        self.dual_end = self.cable_params.end_layout == "dual_end"
        self.data = mujoco.MjData(self.model)
        self.mechanics = CableMechanics(self.model, self.cable_params)
        self.control_dt = 1.0 / float(self.scene.config["physics"]["control_hz"])
        # 控制频率来自菜谱，物理子步由该比值严格决定；拒绝非整数比可防止
        # 接触积分在不同 rollout 间累积时间误差。
        self.physics_substeps = int(round(self.control_dt / self.model.opt.timestep))
        if self.physics_substeps <= 0 or not np.isclose(
            self.physics_substeps * self.model.opt.timestep, self.control_dt, atol=1e-9
        ):
            raise ValueError(
                "control_hz must be an integer multiple of physics timestep"
            )
        self.max_episode_steps = int(self.scene.config["physics"]["max_episode_steps"])
        cable_constraints = self.scene.constraints["cable"]
        self.auditor = SafetyAuditor(
            tension_limit_n=float(cable_constraints["tension_limit_n"]),
            tension_limit_duration_s=float(cable_constraints["limit_duration_s"]),
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(-50.0, 50.0, shape=(30,), dtype=np.float32),
                "ee_pose": spaces.Box(-5.0, 5.0, shape=(14,), dtype=np.float32),
                "wrench": spaces.Box(-1_000.0, 1_000.0, shape=(12,), dtype=np.float32),
                "cable": spaces.Box(-1_000.0, 1_000.0, shape=(3,), dtype=np.float32),
            }
        )
        self.controller = BimanualResolvedRateController(
            self.model, self.handles, self.control_dt
        )
        self._renderer: mujoco.Renderer | None = None
        self._render_camera = mujoco.MjvCamera()
        self._render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        # 非交互截图使用近距离斜视角，保证端头、两个 2F-85 和 Flex 同框；
        # 交互模式的四视角由 demo 单独控制，不依赖此固定相机。
        self._render_camera.lookat = [0.45, 0, 0.78]
        self._render_camera.distance = 0.9
        self._render_camera.azimuth = -135.0
        self._render_camera.elevation = -15.0
        self._step_count = 0
        self._last_result: dict[str, Any] = {}
        self._resolved_target_region = np.asarray(
            self.scene.config["initialization"]["target_region"], dtype=np.float64
        )
        self._grasp_history = {side: deque(maxlen=3) for side in ("left", "right")}
        self._collision_pairs = []
        self._previous_bend_alarm = False
        self._previous_warning = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._replay_restored = False
        for history in self._grasp_history.values():
            history.clear()
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_scene()
        self._set_ready_pose()
        self.controller.reset(self.data)
        # 初始条件只允许左 2F-85 通过正常接触闭合夹住端头；右侧从张开状态
        # 出发，必须自行接近、夹住并维持端头，不能借由 weld 或 teleport。
        self.data.ctrl[self.handles.gripper_actuator_ids] = [
            255.0,
            255.0 if self.dual_end else 0.0,
        ]
        mujoco.mj_forward(self.model, self.data)
        # 暂时关闭重力只用于 reset 后的接触沉降：它不 weld、不改端头 pose，
        # 让手指先在指定夹持区闭合，再由线束尾端施加载荷。
        gravity = self.model.opt.gravity.copy()
        self.model.opt.gravity[:] = 0.0
        try:
            for settle_step in range(int(1.2 / self.model.opt.timestep)):
                self._hold_robot_pose()
                self.mechanics.apply(self.data)
                mujoco.mj_step(self.model, self.data)
                self._check_numerics()
                if settle_step % self.physics_substeps == 0:
                    self._update_grasp_history()
        finally:
            self.model.opt.gravity[:] = gravity
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.task.reset()
        self.auditor.reset()
        self._collision_pairs = []
        if self._has_disallowed_self_collision():
            raise ValueError(
                f"Initial robot/workcell collision: {self._collision_pairs}"
            )
        self._previous_bend_alarm = False
        self._previous_warning = False
        self._step_count = 0
        self._last_result = {}
        state = self._task_state()
        result = self.task.evaluate(state, self.auditor.snapshot())
        self._last_result = self._result_info(result)
        return self._observation(), self._info(result)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if getattr(self, "_replay_restored", False):
            raise RuntimeError(
                "Replay is read-only; reset before starting a new episode"
            )
        # 一个 policy tick 内推进固定数量物理步；审计应在每个物理步观察，
        # 否则短暂超拉或左右臂碰撞会被低频控制采样遗漏。
        self.controller.apply(self.data, action)
        for _ in range(self.physics_substeps):
            self.mechanics.apply(self.data)
            # 场景中显式声明的扰动试验，施加真实外力且写入配置/轨迹。
            if self.dual_end:
                cfg = self.scene.config["control"]
                active = (
                    cfg["disturbance_time_s"]
                    <= self.data.time
                    < cfg["disturbance_time_s"] + cfg["disturbance_duration_s"]
                )
                b = self.mechanics.bodies[len(self.mechanics.bodies) // 2]
                self.data.xfrc_applied[b, :3] = (
                    self._disturbance_force_n if active else 0
                )
            mujoco.mj_step(self.model, self.data)
            self._check_numerics()
            forbidden_contact = self._has_disallowed_self_collision()
            self.auditor.observe(
                cable_tension_n=self._cable_tension(),
                self_collision=forbidden_contact,
                dt=float(self.model.opt.timestep),
                collision_pairs=self._collision_pairs,
            )
        mujoco.mj_forward(self.model, self.data)
        self._step_count += 1
        self._update_grasp_history()
        material = self.mechanics.state(self.data)
        if material["bend_alarm"] and not self._previous_bend_alarm:
            self.auditor.record(
                "bend_radius_alarm", minimum_radius_m=material["minimum_radius_m"]
            )
        self._previous_bend_alarm = material["bend_alarm"]
        cfg = self.scene.config.get("control", {})
        warning = material["tension_n"] > cfg.get("warning_tension_n", float("inf"))
        if warning and not self._previous_warning:
            self.auditor.record("tension_warning", tension_n=material["tension_n"])
        self._previous_warning = warning
        result = self.task.evaluate(self._task_state(), self.auditor.snapshot())
        self._last_result = self._result_info(result)
        return (
            self._observation(),
            float(result.reward),
            bool(result.terminated),
            bool(result.truncated),
            self._info(result),
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self._render_camera)
        return self._renderer.render()

    def _check_numerics(self) -> None:
        # MuJoCo 可能在发出 BADQACC 后自动重置；这种回合不得伪装成稳定运行。
        warnings = (
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
        )
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or any(self.data.warning[i].number for i in warnings)
        ):
            raise FloatingPointError(
                "MuJoCo numerical instability: episode invalid; inspect timestep/material parameters"
            )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def rollout_manifest(self, info: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return reproducibility metadata; no dataset format is implied."""

        from ibero.core.reproducibility import simulation_source_hash

        base = {
            "ibero_version": "0.1.0",
            "mujoco_version": mujoco.__version__,
            "scene_id": self.scene.config["id"],
            "scene_hash": self.scene.scene_hash,
            "config_hash": self.scene.config_hash,
            "constraints_hash": self.scene.constraints_hash,
            "robot_asset_hash": vendored_robot_asset_hash(),
            "simulation_source_hash": simulation_source_hash(),
            "seed": int(self.np_random_seed),
            "control_hz": self.scene.config["physics"]["control_hz"],
            "action_interface": "bimanual_tcp_delta_v0",
            "end_effector": {"type": self.end_effector_type},
            "cable": {
                "representation": self.cable_params.representation,
                "parameters": self.scene.config["materials"]["cable"],
            },
            "resolved_initialization": {
                "target_region": self._resolved_target_region.tolist(),
                "disturbance_force_n": self._disturbance_force_n.tolist(),
            },
            "result": self._last_result,
        }
        result = self.auditor.manifest(base=base)
        if info is not None:
            # 保存回放帧时使用该帧的审计，不混入最后一次实时回合的任务状态。
            result["result"] = {
                k: info.get(k)
                for k in ("stage", "is_success", "failure_reason", "metrics")
            }
            result["frame_audit"] = info["audit"]
            result["replay"] = bool(info.get("replay", False))
            if result["replay"]:
                result["audit"] = info["audit"]
        return result

    def save_manifest(
        self, path: str | Path, info: dict[str, Any] | None = None
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.rollout_manifest(info), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _set_ready_pose(self) -> None:
        arm_joint_ids = self.model.actuator_trnid[self.handles.arm_actuator_ids, 0]
        arm_qpos = self.model.jnt_qposadr[arm_joint_ids]
        ready = np.asarray(
            self.scene.config["robot"].get("arm_qpos", HANDOVER_ARM_QPOS)
        )
        if np.any(ready < self.model.jnt_range[arm_joint_ids, 0]) or np.any(
            ready > self.model.jnt_range[arm_joint_ids, 1]
        ):
            raise ValueError("robot.arm_qpos exceeds joint limits")
        self.data.qpos[arm_qpos] = ready
        self._hold_robot_pose()
        mujoco.mj_forward(self.model, self.data)

    def _randomize_scene(self) -> None:
        """Apply only YAML-declared, seed-recorded scene variation."""

        initialization = self.scene.config["initialization"]
        nominal = np.asarray(initialization["target_region"], dtype=np.float64)
        jitter = np.asarray(initialization["target_jitter_m"], dtype=np.float64)
        if nominal.shape != (3,) or jitter.shape != (3,):
            raise ValueError(
                "target_region and target_jitter_m must both have three values"
            )
        self._resolved_target_region = nominal + self.np_random.uniform(-jitter, jitter)
        self.model.body_pos[self.handles.target_receiver_body_id] = (
            self._resolved_target_region
        )
        cfg = self.scene.config.get("control", {})
        fraction = cfg.get("disturbance_jitter_fraction", 0)
        self._disturbance_force_n = np.asarray(
            cfg.get("disturbance_force_n", [0, 0, 0]), dtype=float
        ) * (1 + self.np_random.uniform(-fraction, fraction))

    def _hold_robot_pose(self) -> None:
        self.controller.hold_pose(self.data)

    def _cable_tension(self) -> float:
        return self.mechanics.tension(self.data)

    def _gripper_opening(self, side: str) -> float:
        pad_ids = (
            self.handles.left_pad_geom_ids
            if side == "left"
            else self.handles.right_pad_geom_ids
        )
        right_pad = np.mean(self.data.geom_xpos[list(pad_ids[:2])], axis=0)
        left_pad = np.mean(self.data.geom_xpos[list(pad_ids[2:])], axis=0)
        return float(np.linalg.norm(right_pad - left_pad))

    def _terminal_pad_contact(self, side: str) -> bool:
        pad_ids = set(
            self.handles.left_pad_geom_ids
            if side == "left"
            else self.handles.right_pad_geom_ids
        )
        terminal_id = (
            self.model.geom("cable_tail_terminal_geom").id
            if self.dual_end and side == "right"
            else self.handles.cable.terminal_geom_id
        )
        touched = set()
        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 == terminal_id and contact.geom2 in pad_ids:
                touched.add(int(contact.geom2))
            if contact.geom2 == terminal_id and contact.geom1 in pad_ids:
                touched.add(int(contact.geom1))
        pads = (
            self.handles.left_pad_geom_ids
            if side == "left"
            else self.handles.right_pad_geom_ids
        )
        return bool(touched.intersection(pads[:2]) and touched.intersection(pads[2:]))

    def _grasped_raw(self, side: str) -> bool:
        if side == "left":
            marker = self.handles.cable.terminal_left_grasp_site_id
            pinch = self.handles.left_pinch_site_id
        else:
            marker = (
                self.model.site("cable_tail_grasp").id
                if self.dual_end
                else self.handles.cable.terminal_right_grasp_site_id
            )
            pinch = self.handles.right_pinch_site_id
        window = float(self.scene.constraints["task"]["grasp_window_m"])
        release = float(self.scene.constraints["task"]["release_opening_m"])
        # 抓住是“端头居中 + 夹爪闭合 + 指垫真实接触”的合取，不允许单靠
        # 接触回调或单靠闭合角度伪造成功。
        centered = (
            np.linalg.norm(self.data.site_xpos[marker] - self.data.site_xpos[pinch])
            <= window
        )
        return bool(
            centered
            and self._gripper_opening(side) <= release
            and self._terminal_pad_contact(side)
        )

    def _update_grasp_history(self):
        for side in ("left", "right"):
            pinch = (
                self.handles.left_pinch_site_id
                if side == "left"
                else self.handles.right_pinch_site_id
            )
            body = (
                self.handles.cable.tail_body_id
                if side == "right" and self.dual_end
                else self.handles.cable.terminal_body_id
            )
            rotation = self.data.site_xmat[pinch].reshape(3, 3)
            relative = rotation.T @ (self.data.xpos[body] - self.data.site_xpos[pinch])
            self._grasp_history[side].append((self._grasped_raw(side), relative.copy()))

    def _grasped(self, side):
        # 连续三个控制采样均有双指接触且相对滑移小于 4 mm 才确认保持。
        history = self._grasp_history[side]
        return bool(
            len(history) == 3
            and all(sample[0] for sample in history)
            and max(np.linalg.norm(sample[1] - history[0][1]) for sample in history)
            < 0.004
        )

    def _has_disallowed_self_collision(self) -> bool:
        """实际接触审计：跨臂、同臂非邻接连杆、躯干与工装；返回具名证据。"""
        self._collision_pairs = []
        allowed = {
            frozenset(pair)
            for pair in self.scene.constraints["collision"]["allowed_robot_pairs"]
        }
        robot_geoms = (
            self.handles.left_robot_geom_ids | self.handles.right_robot_geom_ids
        )
        for contact in self.data.contact[: self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if min(g1, g2) < 0:
                continue  # Flex 接触属于材料接触而不是机器人自碰撞。
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            n1, n2 = self.model.body(b1).name, self.model.body(b2).name
            pair = frozenset((n1, n2))
            if pair in allowed or b1 == b2:
                continue
            if g1 not in robot_geoms and g2 not in robot_geoms:
                continue
            if n1.startswith("cable_") or n2.startswith("cable_"):
                continue

            # 同一 2F-85 的闭链装配接触由源资产定义；不作为自碰撞报错。
            def gripper_root(b):
                while b:
                    if self.model.body(b).name in (
                        "left_base_mount",
                        "right_base_mount",
                    ):
                        return b
                    b = int(self.model.body_parentid[b])
                return -1

            root1, root2 = gripper_root(b1), gripper_root(b2)
            if root1 >= 0 and root1 == root2:
                continue
            ancestors = {b1: 0}
            b = b1
            for k in range(1, 3):
                b = int(self.model.body_parentid[b])
                ancestors[b] = k
            b = b2
            nearby = False
            for k in range(3):
                if b in ancestors and ancestors[b] + k <= 2 and b != 0:
                    nearby = True
                b = int(self.model.body_parentid[b])
            if nearby:
                continue
            self._collision_pairs.append((n1 or "world", n2 or "world"))
        return bool(self._collision_pairs)

    def _task_state(self) -> TaskState:
        target_radius = float(self.scene.constraints["task"]["target_radius_m"])
        contacts = frozenset()
        tray = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "receiver_support"
        )
        supported = tray >= 0 and any(
            set((int(c.geom1), int(c.geom2)))
            == {tray, self.handles.cable.terminal_geom_id}
            for c in self.data.contact[: self.data.ncon]
        )
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.handles.cable.terminal_body_id,
            velocity,
            0,
        )
        return TaskState(
            body_positions={
                "cable_terminal": self.data.xpos[
                    self.handles.cable.terminal_body_id
                ].copy(),
                "cable_tail_terminal": self.data.xpos[
                    self.handles.cable.tail_body_id
                ].copy(),
            },
            site_positions={
                "left_pinch": self.data.site_xpos[
                    self.handles.left_pinch_site_id
                ].copy(),
                "right_pinch": self.data.site_xpos[
                    self.handles.right_pinch_site_id
                ].copy(),
                "target_region": self.data.site_xpos[
                    self.handles.target_region_site_id
                ].copy(),
                "target_radius": np.asarray([target_radius], dtype=np.float64),
            },
            gripper_openings={
                "left_gripper": self._gripper_opening("left"),
                "right_gripper": self._gripper_opening("right"),
            },
            grasps={
                "left_gripper": self._grasped("left"),
                "right_gripper": self._grasped("right"),
            },
            contacts=contacts,
            cable_tension_n=self._cable_tension(),
            elapsed_steps=self._step_count,
            max_episode_steps=self.max_episode_steps,
            extra={
                **self.scene.config.get("control", {}),
                "supported": supported,
                "terminal_speed_m_s": float(np.linalg.norm(velocity[3:])),
                "left_contact": self._terminal_pad_contact("left"),
                "right_contact": self._terminal_pad_contact("right"),
                "dt": self.control_dt,
                "time_s": float(self.data.time),
                "bend_alarm": self.mechanics.state(self.data)["bend_alarm"],
                "disturbance_end_s": self.scene.config.get("control", {}).get(
                    "disturbance_time_s", 0
                )
                + self.scene.config.get("control", {}).get("disturbance_duration_s", 0),
            },
        )

    def _observation(self) -> dict[str, np.ndarray]:
        arm_joint_ids = self.model.actuator_trnid[self.handles.arm_actuator_ids, 0]
        qpos_address = self.model.jnt_qposadr[arm_joint_ids]
        qvel_address = self.model.jnt_dofadr[arm_joint_ids]
        ee_pose: list[float] = []
        for site_id, opening in (
            (self.handles.left_pinch_site_id, self._gripper_opening("left")),
            (self.handles.right_pinch_site_id, self._gripper_opening("right")),
        ):
            ee_pose.extend(self.data.site_xpos[site_id])
            ee_pose.extend(self.data.site_xmat[site_id].reshape(3, 3)[:, 2])
            ee_pose.append(opening)
        left_wrench = wrist_wrench(
            self.model,
            self.data,
            force_sensor_id=self.handles.left_ft_force_sensor_id,
            torque_sensor_id=self.handles.left_ft_torque_sensor_id,
        )
        right_wrench = wrist_wrench(
            self.model,
            self.data,
            force_sensor_id=self.handles.right_ft_force_sensor_id,
            torque_sensor_id=self.handles.right_ft_torque_sensor_id,
        )
        terminal_target_distance = np.linalg.norm(
            self.data.xpos[self.handles.cable.terminal_body_id]
            - self.data.site_xpos[self.handles.target_region_site_id]
        )
        return {
            "proprio": np.concatenate(
                (
                    self.data.qpos[qpos_address],
                    self.data.qvel[qvel_address],
                    [self._gripper_opening("left"), self._gripper_opening("right")],
                )
            ).astype(np.float32),
            "ee_pose": np.asarray(ee_pose, dtype=np.float32),
            "wrench": np.concatenate((left_wrench, right_wrench)).astype(np.float32),
            "cable": np.asarray(
                # 当前弧长，而非两端直线距离；松弛线束也有完整的材料长度。
                [
                    self._cable_tension(),
                    self.mechanics.state(self.data)["arc_length_m"],
                    terminal_target_distance,
                ],
                dtype=np.float32,
            ),
        }

    @staticmethod
    def _result_info(result: Any) -> dict[str, Any]:
        return {
            "stage": result.stage,
            "is_success": bool(result.success),
            "failure_reason": result.failure_reason,
            "metrics": dict(result.metrics),
        }

    def _info(self, result: Any) -> dict[str, Any]:
        snapshot = self.auditor.snapshot()
        return {
            **self._result_info(result),
            "audit": {
                "material": self.mechanics.state(self.data),
                "collision_pairs": self._collision_pairs,
                "cable_tension_n": self._cable_tension(),
                "peak_cable_tension_n": snapshot.peak_cable_tension_n,
                "damage_flags": {"cable_over_tension": snapshot.cable_damage},
                # left_right 为旧客户端兼容键；新版 any_forbidden 包含躯干/工装碰撞。
                "self_collision_flags": {
                    "left_right": snapshot.self_collision,
                    "any_forbidden": snapshot.self_collision,
                },
                "events": list(snapshot.events),
            },
        }
