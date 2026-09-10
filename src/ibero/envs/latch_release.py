"""自由卡扣双臂环境；执行器控制与材料接触因果独立于任务判定。"""

import copy
from dataclasses import replace
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ibero.core.scene_loader import SceneLoader
from ibero.core.scene_compiler import SceneCompiler
from ibero.core.task_api import TaskState
from ibero.core.audit import AuditSnapshot
from ibero.core.reproducibility import simulation_source_hash
from ibero.core.task_loading import load_task_spec
from ibero.control.latch import LatchArmController
from ibero.mechanisms.snap_latch import LatchObserver
from ibero.robots.g1_upperbody import ARM_JOINTS
from ibero.robots.g1_upperbody_2f85 import vendored_robot_asset_hash

DEFAULT_SCENE = Path(__file__).resolve().parents[3] / "scenes/latch_release"


class LatchReleaseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    scene_kind = "latch_release"

    def __init__(self, scene_path=DEFAULT_SCENE, render_mode=None):
        self.scene = SceneLoader().validate(scene_path)
        if self.scene.config["kind"] != self.scene_kind:
            raise ValueError(f"Expected {self.scene_kind} recipe")
        self.render_mode = render_mode
        self.control_dt = 1 / self.scene.config["physics"]["control_hz"]
        self.max_episode_steps = round(
            self.scene.constraints["task"]["max_seconds"] / self.control_dt
        )
        self.task = load_task_spec(self.scene)
        self.action_space = spaces.Box(-1, 1, (14,), dtype=np.float32)
        self._renderer = None
        self.reset(seed=0)
        self.observation_space = spaces.Dict(
            {
                "qpos": spaces.Box(-np.inf, np.inf, (self.model.nq,), dtype=np.float64),
                "qvel": spaces.Box(-np.inf, np.inf, (self.model.nv,), dtype=np.float64),
                "latch": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        cfg = copy.deepcopy(self.scene.config)
        jitter = (
            self.np_random.uniform(-1, 1, 3) * cfg["initialization"]["origin_jitter_m"]
        )
        cfg["initialization"]["origin_m"] = (
            np.asarray(cfg["initialization"]["origin_m"]) + jitter
        ).tolist()
        j = cfg["initialization"]["friction_jitter"]
        cfg["mechanism"]["friction"] += float(self.np_random.uniform(-j, j))
        self.resolved_config = cfg
        self.seed_value = seed
        cell = SceneCompiler().compile(replace(self.scene, config=cfg))
        self.model, self.handles = cell.model, cell.handles
        self.data = mujoco.MjData(self.model)
        for name, value in zip(ARM_JOINTS, cfg["robot"]["arm_qpos"]):
            joint = self.model.joint(name)
            if not joint.range[0] <= value <= joint.range[1]:
                raise ValueError(f"Reset joint outside limits: {name}")
            self.data.joint(name).qpos[0] = value
        mujoco.mj_forward(self.model, self.data)
        self.controller = LatchArmController(self.model, self.handles, self.control_dt)
        self.controller.reset(self.data)
        self.data.ctrl[self.handles.gripper_actuator_ids[0]] = 0
        material = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom(i).name.startswith(("tongue_", "latch_hook"))
        ]
        safety = self.scene.constraints["safety"]
        self.observer = LatchObserver(
            self.model,
            cell.latch_names,
            audited_geom_ids=material,
            max_force_n=safety["max_force_n"],
            max_deflection_m=safety["max_deflection_m"],
        )
        self.plug = self.model.body("latch_plug").id
        self.socket = self.model.body("latch_socket").id
        self.shell = self.model.geom("plug_shell").id
        self.pad_groups = [
            {self.model.geom(f"right_{side}_pad{i}").id for i in (1, 2)}
            for side in ("left", "right")
        ]
        self.robot_geoms = set()
        self.gripper_geoms = set()
        pelvis = self.model.body("pelvis").id
        grip_base = self.model.body("right_base").id
        for gid in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[gid])
            while body > 0:
                if body == pelvis:
                    self.robot_geoms.add(gid)
                if body == grip_base:
                    self.gripper_geoms.add(gid)
                body = int(self.model.body_parentid[body])
        self.elapsed_steps = 0
        self._robot_mask = np.array(
            [i in self.robot_geoms for i in range(self.model.ngeom)]
        )
        self._gripper_mask = np.array(
            [i in self.gripper_geoms for i in range(self.model.ngeom)]
        )
        self._pad_mask = np.array(
            [any(i in p for p in self.pad_groups) for i in range(self.model.ngeom)]
        )
        self.invalid_reason = None
        self.events = []
        self._done = False
        self._review_only = False
        self._replay_restored = False
        self._reset_materials(cell)
        self.task.reset()
        self.last_state = self.observer.observe(self.data)
        self._measure_contacts()
        self.last_info = self._info()
        return self._observation(), self.last_info

    def _measure_contacts(self):
        """夹持与舌片载荷分开；自碰撞、指垫穿透和插座反力仍单独审计。"""
        touch = [False, False]
        self.contact_pairs = set()
        self.socket_force_world = np.zeros(3)
        force = np.zeros(6)
        self.contact_details = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            # Flex 接触的 geom id 为 -1，不能负索引到最后一个刚体 geom。
            names = [
                self.model.geom(g).name
                if g >= 0
                else f"flex:{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_FLEX, int(c.flex[j]))}"
                for j, g in enumerate((a, b))
            ]
            self.contact_pairs.add(tuple(names))
            mujoco.mj_contactForce(self.model, self.data, i, force)
            self.contact_details.append(
                {
                    "pair": names,
                    "force_n": float(np.linalg.norm(force[:3])),
                    "penetration_m": max(0, -float(c.dist)),
                }
            )
            for side, pads in enumerate(self.pad_groups):
                if (a in pads and b == self.shell) or (b in pads and a == self.shell):
                    touch[side] |= force[0] > 0.1
                    if -c.dist > self.scene.constraints["safety"]["grip_penetration_m"]:
                        self.invalid_reason = "grip_penetration"
            if (
                a in self.robot_geoms
                and b in self.robot_geoms
                and not {a, b} <= self.gripper_geoms
            ):
                if (
                    -c.dist
                    > self.scene.constraints["safety"]["self_collision_penetration_m"]
                ):
                    self.invalid_reason = "robot_self_collision"
            body_a, body_b = [
                int(self.model.geom_bodyid[g]) if g >= 0 else -1 for g in (a, b)
            ]
            world = c.frame.reshape(3, 3).T @ force[:3]
            if body_a == self.socket:
                self.socket_force_world -= world
            if body_b == self.socket:
                self.socket_force_world += world
        self.grasped = bool(all(touch))

    def step(self, action):
        if self._done or self._review_only or self._replay_restored:
            raise RuntimeError("Reset before stepping a finished/replayed episode")
        self.controller.apply(self.data, action)
        for _ in range(round(self.control_dt / self.model.opt.timestep)):
            self._before_substep()
            mujoco.mj_step(self.model, self.data)
            # 接触力、xpos 均对应本子步积分前状态；标记正确时刻，避免重复求解。
            self.last_state = self.observer.observe(
                self.data, sample_time_s=self.data.time - self.model.opt.timestep
            )
            # 每子步检查碰撞几何，不重复计算只在控制帧消费的全部坐标力矩。
            pairs = self.data.contact.geom
            if len(pairs):
                a, b = pairs.T
                valid_a, valid_b = a >= 0, b >= 0
                a, b = np.maximum(a, 0), np.maximum(b, 0)
                penetration = -self.data.contact.dist
                forbidden = (
                    self._robot_mask[a]
                    & self._robot_mask[b]
                    & valid_a
                    & valid_b
                    & ~(self._gripper_mask[a] & self._gripper_mask[b])
                )
                grip = (
                    (
                        (self._pad_mask[a] & (b == self.shell))
                        | (self._pad_mask[b] & (a == self.shell))
                    )
                    & valid_a
                    & valid_b
                )
                if np.any(
                    forbidden
                    & (
                        penetration
                        > self.scene.constraints["safety"][
                            "self_collision_penetration_m"
                        ]
                    )
                ):
                    self.invalid_reason = "robot_self_collision"
                if np.any(
                    grip
                    & (
                        penetration
                        > self.scene.constraints["safety"]["grip_penetration_m"]
                    )
                ):
                    self.invalid_reason = "grip_penetration"
            if self.observer.invalid_reason:
                self.invalid_reason = self.observer.invalid_reason
            self._after_substep()
            if self.invalid_reason:
                break
        self.elapsed_steps += 1
        mujoco.mj_forward(self.model, self.data)
        self.last_state = self.observer.observe(self.data)
        if self.observer.invalid_reason:
            self.invalid_reason = self.observer.invalid_reason
        self._measure_contacts()
        self._after_substep()
        if self.invalid_reason and not self.events:
            self.events.append(
                {"time_s": float(self.data.time), "type": self.invalid_reason}
            )
        audit = AuditSnapshot(
            False,
            self.invalid_reason == "robot_self_collision",
            0,
            0,
            tuple(self.events),
        )
        result = self.task.evaluate(self._task_state(), audit)
        self._done = result.terminated or result.truncated
        info = self._info()
        info.update(
            success=result.success,
            stage=result.stage,
            failure_reason=result.failure_reason,
            task_metrics=dict(result.metrics),
        )
        self.last_info = info
        return (
            self._observation(),
            result.reward,
            result.terminated,
            result.truncated,
            info,
        )

    def _task_state(self):
        task = self.scene.constraints["task"]
        frame = self.data.xmat[self.socket].reshape(3, 3)
        position = frame.T @ (self.data.xpos[self.plug] - self.data.xpos[self.socket])
        relative = frame.T @ self.data.xmat[self.plug].reshape(3, 3)
        extra = dict(
            self.last_state,
            invalid_reason=self.invalid_reason,
            withdrawal_m=-float(position[0]),
            required_withdrawal_m=task["withdrawal_distance_m"],
            required_hold_seconds=task["hold_seconds"],
            pose_error_rad=float(
                np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
            ),
            pose_tolerance_rad=task["pose_tolerance_rad"],
            control_dt=self.control_dt,
        )
        extra.update(self._material_task_state())
        return TaskState(
            {"plug": self.data.xpos[self.plug].copy()},
            {},
            {},
            {"right": self.grasped},
            frozenset(self.contact_pairs),
            0,
            self.elapsed_steps,
            self.max_episode_steps,
            extra,
        )

    def _observation(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "latch": np.array(
                [
                    self.last_state[k]
                    for k in ("clearance_m", "deflection_m", "contact_force_n")
                ]
            ),
        }

    # 小范围组合点只用于当前真实线束对象；不建立任意多引擎生命周期框架。
    def _reset_materials(self, cell):
        pass

    def _before_substep(self):
        pass

    def _after_substep(self):
        pass

    def _material_task_state(self):
        return {}

    def _info(self):
        return dict(
            self._task_state().extra,
            grasped=self.grasped,
            success=False,
            socket_force_world_n=self.socket_force_world.tolist(),
            events=list(self.events),
            contacts=self.contact_details,
            wrist_sensors={
                side: {
                    kind: self.data.sensor(f"{side}_wrist_ft_{kind}")
                    .data.copy()
                    .tolist()
                    for kind in ("force", "torque")
                }
                for side in ("left", "right")
            },
        )

    def manifest(self):
        return {
            "scene_hash": self.scene.scene_hash,
            "source_hash": simulation_source_hash(),
            "asset_hash": vendored_robot_asset_hash(),
            "mujoco_version": mujoco.__version__,
            "seed": self.seed_value,
            "resolved_config": self.resolved_config,
            "parameter_status": "provisional",
        }

    def render(self, camera="front"):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
