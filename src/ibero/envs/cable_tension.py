"""The first executable IBERO environment: cooperative cable tension."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ibero.control import BimanualResolvedRateController
from ibero.robots.g1_upperbody import build_g1_cable_model


class CableTensionEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Hold a cable in a safe force band using two G1 end effectors.

    Actions are normalized end-effector deltas in the order
    ``[left xyz/rpy/grip, right xyz/rpy/grip]``.  The task intentionally
    attaches hard cable terminals to tool frames: Core-0 tests cooperative
    force control before it tests soft-object grasping.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    control_dt = 0.05
    physics_substeps = 25
    max_episode_steps = 200
    target_tension_range = (3.0, 7.0)
    damage_tension = 15.0
    target_hold_steps = 12

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.model, self.handles = build_g1_cable_model()
        self.data = mujoco.MjData(self.model)
        self.controller = BimanualResolvedRateController(
            self.model, self.handles, self.control_dt
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(-50.0, 50.0, shape=(28,), dtype=np.float32),
                "ee_pose": spaces.Box(-5.0, 5.0, shape=(14,), dtype=np.float32),
                "wrench": spaces.Box(-1_000.0, 1_000.0, shape=(12,), dtype=np.float32),
                "cable": spaces.Box(-1_000.0, 1_000.0, shape=(2,), dtype=np.float32),
            }
        )
        self._renderer: mujoco.Renderer | None = None
        self._render_camera = mujoco.MjvCamera()
        self._render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._render_camera.lookat = [0.15, 0.0, 0.55]
        self._render_camera.distance = 1.75
        self._render_camera.azimuth = 0.0
        self._render_camera.elevation = -12.0
        self._step_count = 0
        self._in_band_steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._set_ready_pose()
        self.controller.reset(self.data)
        self._update_cable_visual()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._in_band_steps = 0
        observation = self._observation()
        info = self._info(is_success=False)
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.controller.apply(self.data, action)
        for _ in range(self.physics_substeps):
            mujoco.mj_step(self.model, self.data)
        self._update_cable_visual()
        mujoco.mj_forward(self.model, self.data)
        self._step_count += 1

        tension = self._cable_tension()
        in_band = (
            self.target_tension_range[0] <= tension <= self.target_tension_range[1]
        )
        self._in_band_steps = self._in_band_steps + 1 if in_band else 0
        damage = tension >= self.damage_tension
        collision = self._has_self_collision()
        is_success = self._in_band_steps >= self.target_hold_steps and not collision
        terminated = bool(is_success or damage)
        truncated = bool(self._step_count >= self.max_episode_steps)
        reward = self._reward(tension, damage, collision, is_success)
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            self._info(is_success),
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self._render_camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _set_ready_pose(self) -> None:
        # A symmetric, forward-facing posture.  The small seed perturbation
        # provides reproducible initial-state diversity without hiding task
        # semantics in a random placement sampler.
        ready = np.asarray(
            [
                -0.80,
                0.45,
                0.00,
                1.00,
                0.00,
                -0.20,
                0.00,
                -0.80,
                -0.45,
                0.00,
                1.00,
                0.00,
                -0.20,
                0.00,
            ],
            dtype=np.float64,
        )
        noise = self.np_random.uniform(-0.015, 0.015, size=ready.shape)
        arm_joint_ids = self.model.actuator_trnid[self.handles.arm_actuator_ids, 0]
        arm_qpos = self.model.jnt_qposadr[arm_joint_ids]
        self.data.qpos[arm_qpos] = ready + noise
        mujoco.mj_forward(self.model, self.data)

    def _update_cable_visual(self) -> None:
        """Place kinematic cable capsules along a deterministic sag curve."""

        left = self.data.site_xpos[self.handles.left_anchor_site_id].copy()
        right = self.data.site_xpos[self.handles.right_anchor_site_id].copy()
        count = len(self.handles.cable_mocap_ids)
        previous = left
        for index, mocap_id in enumerate(self.handles.cable_mocap_ids):
            t1 = (index + 1) / count
            point = (1.0 - t1) * left + t1 * right
            point[2] -= 0.055 * 4.0 * t1 * (1.0 - t1)
            direction = point - previous
            direction /= np.linalg.norm(direction)
            self.data.mocap_pos[mocap_id] = (previous + point) / 2.0
            self.data.mocap_quat[mocap_id] = self._quat_from_z(direction)
            previous = point

    @staticmethod
    def _quat_from_z(direction: np.ndarray) -> np.ndarray:
        """Return a scalar-first quaternion rotating local +z onto direction."""

        source = np.asarray([0.0, 0.0, 1.0])
        dot = float(np.clip(source @ direction, -1.0, 1.0))
        if dot < -0.999999:
            return np.asarray([0.0, 1.0, 0.0, 0.0])
        axis = np.cross(source, direction)
        quat = np.asarray([1.0 + dot, axis[0], axis[1], axis[2]])
        return quat / np.linalg.norm(quat)

    def _cable_tension(self) -> float:
        tendon_id = self.handles.cable_tendon_id
        rest_length = self.model.tendon_lengthspring[tendon_id, 0]
        extension = self.data.ten_length[tendon_id] - rest_length
        force = (
            self.model.tendon_stiffness[tendon_id] * extension
            + self.model.tendon_damping[tendon_id] * self.data.ten_velocity[tendon_id]
        )
        # A cable reports tensile load only.  Compression is not a damage mode.
        return float(max(0.0, force))

    def _has_self_collision(self) -> bool:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            left = self._is_side_geom(contact.geom1, "left")
            right = self._is_side_geom(contact.geom2, "right")
            if (left and right) or (
                self._is_side_geom(contact.geom2, "left")
                and self._is_side_geom(contact.geom1, "right")
            ):
                return True
        return False

    def _is_side_geom(self, geom_id: int, side: str) -> bool:
        body_id = self.model.geom_bodyid[geom_id]
        while body_id > 0:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and name.startswith(f"{side}_"):
                return True
            body_id = self.model.body_parentid[body_id]
        return False

    def _observation(self) -> dict[str, np.ndarray]:
        arm_joint_ids = self.model.actuator_trnid[self.handles.arm_actuator_ids, 0]
        qpos_adr = self.model.jnt_qposadr[arm_joint_ids]
        qvel_adr = self.model.jnt_dofadr[arm_joint_ids]
        ee_pose = []
        for site_id in (
            self.handles.left_anchor_site_id,
            self.handles.right_anchor_site_id,
        ):
            ee_pose.extend(self.data.site_xpos[site_id])
            ee_pose.extend(self.data.site_xmat[site_id].reshape(3, 3)[:, 2])
            ee_pose.append(1.0)
        wrench = []
        for body_id in (
            self.handles.left_wrist_body_id,
            self.handles.right_wrist_body_id,
        ):
            # cfrc_ext uses [torque, force] in the body frame.  IBERO presents
            # the conventional [Fx, Fy, Fz, Tx, Ty, Tz] order.
            torque_force = self.data.cfrc_ext[body_id]
            wrench.extend(torque_force[3:])
            wrench.extend(torque_force[:3])
        cable_length = float(
            np.linalg.norm(
                self.data.site_xpos[self.handles.right_anchor_site_id]
                - self.data.site_xpos[self.handles.left_anchor_site_id]
            )
        )
        return {
            "proprio": np.concatenate(
                (self.data.qpos[qpos_adr], self.data.qvel[qvel_adr])
            ).astype(np.float32),
            "ee_pose": np.asarray(ee_pose, dtype=np.float32),
            "wrench": np.asarray(wrench, dtype=np.float32),
            "cable": np.asarray(
                [self._cable_tension(), cable_length], dtype=np.float32
            ),
        }

    def _reward(
        self, tension: float, damage: bool, collision: bool, success: bool
    ) -> float:
        target = sum(self.target_tension_range) / 2.0
        reward = 1.0 - min(abs(tension - target) / target, 1.0)
        if damage:
            reward -= 2.0
        if collision:
            reward -= 1.0
        if success:
            reward += 2.0
        return float(reward)

    def _info(self, is_success: bool) -> dict[str, Any]:
        tension = self._cable_tension()
        collision = self._has_self_collision()
        return {
            "is_success": bool(is_success),
            "audit": {
                "cable_tension_n": tension,
                "damage_flags": {"cable_over_tension": tension >= self.damage_tension},
                "self_collision_flags": {"left_right": collision},
                "in_band_steps": self._in_band_steps,
            },
        }
