from __future__ import annotations

import copy

import gymnasium.utils.env_checker
import mujoco
import numpy as np

import ibero
from ibero.baselines import CableHandoverScript
from ibero.calibrate import wrist_static_load_check
from ibero.core.scene_compiler import SceneCompiler
from ibero.core.scene_loader import SceneLoader
from ibero.envs.cable_handover import DEFAULT_SCENE_PATH
from ibero.multiview import CableHandoverMultiView
from ibero.robots.g1_upperbody_2f85 import build_g1_handover_model


def _run_baseline(env: ibero.CableHandoverEnv, seed: int) -> dict:
    env.reset(seed=seed)
    baseline = CableHandoverScript()
    for _ in range(env.max_episode_steps):
        _, _, terminated, truncated, info = env.step(baseline.action(env))
        if terminated or truncated:
            return info
    raise AssertionError("Cable handover did not finish at the episode limit")


def test_scene_recipe_is_strict_and_scene_level_material_is_flex() -> None:
    scene = SceneLoader().validate(DEFAULT_SCENE_PATH)
    cable = scene.config["materials"]["cable"]
    assert cable["representation"] == "flex"
    assert cable["length_m"] > 0
    assert cable["outer_diameter_m"] > 0
    assert cable["parameter_source"]["length_m"] == "provisional"
    assert len(scene.scene_hash) == 64
    compiled = SceneCompiler().compile(scene)
    assert compiled.model.nflex == 1
    assert compiled.cable_parameters.representation == "flex"
    body_names = {
        mujoco.mj_id2name(compiled.model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(compiled.model.nbody)
    }
    assert "left_hip_pitch_link" not in body_names
    assert "right_hip_pitch_link" not in body_names
    assert compiled.model.ncam == 3


def _mesh_names_used_by_geoms(model: mujoco.MjModel) -> set[str]:
    """Only count visual/collision meshes actually used by a geom, not unused assets."""

    return {
        name
        for geom_id in range(model.ngeom)
        if model.geom_dataid[geom_id] >= 0
        if (
            name := mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom_id])
            )
        )
        is not None
    }


def test_end_effector_selection_never_stacks_g1_hand_visual_and_2f85() -> None:
    scene = SceneLoader().validate(DEFAULT_SCENE_PATH)
    robotiq_model, robotiq_handles, _ = build_g1_handover_model(
        scene.config, scene.constraints
    )
    native_meshes = {"left_rubber_hand", "right_rubber_hand"}
    assert robotiq_handles.end_effector_type == "robotiq_2f85"
    assert not (_mesh_names_used_by_geoms(robotiq_model) & native_meshes)
    assert len(robotiq_handles.gripper_actuator_ids) == 2

    native_config = copy.deepcopy(dict(scene.config))
    native_config["robot"]["end_effector"]["type"] = "g1_native_hand"
    native_model, native_handles, _ = build_g1_handover_model(
        native_config, scene.constraints
    )
    assert native_handles.end_effector_type == "g1_native_hand"
    assert native_meshes <= _mesh_names_used_by_geoms(native_model)
    assert len(native_handles.gripper_actuator_ids) == 0
    assert mujoco.mj_name2id(native_model, mujoco.mjtObj.mjOBJ_BODY, "left_base") < 0


def test_four_view_renderer_exposes_all_review_views() -> None:
    env = ibero.make("ibero/CableHandover-v0")
    try:
        env.reset(seed=7)
        views = CableHandoverMultiView(env)
        try:
            image = views.render()
            assert image.shape == (480, 640, 3)
            assert image.dtype == np.uint8
            assert views.labels == ("总览", "左末端", "右末端", "端头与目标区")
        finally:
            views.close()
    finally:
        env.close()


def test_reset_is_seed_deterministic_and_records_scene_randomization() -> None:
    env = ibero.make("ibero/CableHandover-v0")
    try:
        first, _ = env.reset(seed=11)
        first_target = env.data.site_xpos[env.handles.target_region_site_id].copy()
        second, _ = env.reset(seed=11)
        second_target = env.data.site_xpos[env.handles.target_region_site_id].copy()
        for key in first:
            np.testing.assert_allclose(first[key], second[key], atol=1e-6)
        np.testing.assert_allclose(first_target, second_target, atol=1e-8)
        _, _ = env.reset(seed=12)
        third_target = env.data.site_xpos[env.handles.target_region_site_id].copy()
        assert not np.allclose(first_target, third_target)
    finally:
        env.close()


def test_initial_grasp_is_real_contact_and_wrist_ft_static_check_passes() -> None:
    env = ibero.make("ibero/CableHandover-v0")
    try:
        env.reset(seed=0)
        assert env._grasped("left")
        assert env._terminal_pad_contact("left")
        assert not env._grasped("right")
        check = wrist_static_load_check(env)
        assert check["passed"]
        assert float(check["relative_magnitude_error"]) <= 0.05
    finally:
        env.close()


def test_scripted_contact_handover_succeeds_across_fixed_seeds() -> None:
    env = ibero.make("ibero/CableHandover-v0")
    try:
        for seed in range(3):
            info = _run_baseline(env, seed)
            assert info["is_success"]
            assert info["failure_reason"] is None
            assert not info["audit"]["damage_flags"]["cable_over_tension"]
            assert not info["audit"]["self_collision_flags"]["left_right"]
    finally:
        env.close()


def test_flex_safe_hold_is_stable_for_ten_simulated_seconds() -> None:
    env = ibero.make("ibero/CableHandover-v0")
    try:
        env.reset(seed=4)
        action = np.zeros(14, dtype=np.float32)
        action[6] = 1.0
        action[13] = -1.0
        for _ in range(200):
            observation, _, terminated, truncated, info = env.step(action)
            assert not terminated
            assert not truncated
            assert all(np.isfinite(value).all() for value in observation.values())
        assert not info["audit"]["damage_flags"]["cable_over_tension"]
    finally:
        env.close()


def test_handover_environment_obeys_gymnasium_contract() -> None:
    env = ibero.make("ibero/CableHandover-v0")
    try:
        gymnasium.utils.env_checker.check_env(env, skip_render_check=True)
    finally:
        env.close()
