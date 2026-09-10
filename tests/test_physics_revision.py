"""验收物理含义与故障路径，而非只检查模块能导入。"""

import copy
from dataclasses import replace
import mujoco
import numpy as np
import pytest

import ibero
from ibero.core.scene_loader import SceneLoader, SceneValidationError
from ibero.envs.cable_handover import DEFAULT_SCENE_PATH
from ibero.materials.cable import CableParameters
from ibero.materials.bench import axial_fixture, bending_gradient_check, sag_fixture
from ibero.verify_core01 import sensor_fixture, run_episode
from ibero.review import Trace


@pytest.fixture(scope="module")
def scene():
    return SceneLoader().validate(DEFAULT_SCENE_PATH.parent / "cable_stretch")


def test_mount_axis_is_aligned_with_wrist_forward():
    env = ibero.make("ibero/CableStretch-v0")
    try:
        env.reset(seed=7)
        for side in ("left", "right"):
            wrist = env.model.body(side + "_wrist_yaw_link").id
            pinch = env.model.site(side + "_pinch").id
            local = (
                env.data.xmat[wrist].reshape(3, 3).T
                @ env.data.site_xmat[pinch].reshape(3, 3)[:, 2]
            )
            np.testing.assert_allclose(local, [1, 0, 0], atol=1e-8)
    finally:
        env.close()


def test_material_force_law_mass_and_resolution(scene):
    p = CableParameters.from_scene(scene.config["materials"]["cable"])
    for segments in (6, 12, 24):
        result = axial_fixture(replace(p, segments=segments))
        assert result["relative_error"] < 0.02
        assert abs(result["mass_kg"] - result["expected_mass_kg"]) < 1e-10
    assert bending_gradient_check()["gradient_max_error"] < 1e-6
    low = sag_fixture(p, 0.0002)
    high = sag_fixture(p, 0.002)
    assert low["sag_m"] > high["sag_m"] > 0
    assert max(low["load_error"], high["load_error"]) < 0.05


def test_wrench_axes_and_moment_arm():
    assert max(row["max_error"] for row in sensor_fixture()) < 1e-8


@pytest.mark.parametrize(
    "change",
    [{"damping": float("nan")}, {"segments": 2}, {"representation": "capsule_chain"}],
)
def test_invalid_material_configuration_fails_before_simulation(scene, change):
    config = copy.deepcopy(scene.config)
    config["materials"]["cable"].update(change)
    with pytest.raises(SceneValidationError):
        SceneLoader()._validate_config(config)
        SceneLoader()._validate_physics(config, scene.constraints)


def test_stretch_recovers_after_disturbance_and_preserves_trace(tmp_path):
    from ibero.control.tension import CableStretchScript

    env = ibero.make("ibero/CableStretch-v0")
    try:
        _, info = env.reset(seed=7)
        script = CableStretchScript()
        trace = Trace(env)
        trace.append(info)
        initial = env.data.qpos.copy()
        for _ in range(env.max_episode_steps):
            _, _, terminated, truncated, info = env.step(script.action(env))
            trace.append(info)
            if terminated or truncated:
                break
        assert info["is_success"] and info["metrics"]["hold_seconds"] >= 2
        assert info["metrics"]["disturbance_finished"]
        assert info["audit"]["peak_cable_tension_n"] > 1
        path = tmp_path / "episode.npz"
        trace.save(path)
        loaded = Trace(env).load(path)
        loaded.restore(0)
        np.testing.assert_allclose(env.data.qpos, initial, atol=0)
        final = loaded.restore(len(loaded.states) - 1)
        assert final["is_success"] and env.data.time > 6
        assert len(loaded.infos) == len(trace.infos)
        with pytest.raises(RuntimeError, match="read-only"):
            env.step(np.zeros(14))
        manifest = env.rollout_manifest(final)
        assert manifest["replay"] and manifest["result"]["is_success"]
    finally:
        env.close()


@pytest.mark.parametrize(
    "fault,expected", [("drop", "dropped"), ("overpull", "cable_damage")]
)
def test_contact_release_and_overload_are_failures(fault, expected):
    env = ibero.make("ibero/CableStretch-v0")
    try:
        result = run_episode(env, fault=fault)
        assert not result["success"]
        assert result["failure_reason"] == expected
    finally:
        env.close()


def test_invalid_actions_are_rejected():
    env = ibero.make("ibero/CableStretch-v0")
    try:
        env.reset(seed=7)
        with pytest.raises(ValueError):
            env.step(np.full(14, np.nan))
    finally:
        env.close()


def test_workcell_collision_is_detected_and_latched():
    env = ibero.make("ibero/CableHandover-v0")
    try:
        env.reset(seed=7)
        # 故障夹具故意移动托台使其与夹爪相交；正常策略不修改工装位置。
        pad = env.handles.left_pad_geom_ids[0]
        env.model.body_pos[env.handles.target_receiver_body_id] = env.data.geom_xpos[
            pad
        ] - np.array([0, 0, -0.021])
        mujoco.mj_forward(env.model, env.data)
        assert env._has_disallowed_self_collision()
        assert any("target_receiver" in pair for pair in env._collision_pairs)
        env.auditor.observe(cable_tension_n=0, self_collision=True, dt=0.0002)
        result = env.task.evaluate(env._task_state(), env.auditor.snapshot())
        assert result.failure_reason == "self_collision" and not result.success
        env.auditor.observe(cable_tension_n=0, self_collision=False, dt=0.0002)
        assert env.auditor.snapshot().self_collision
    finally:
        env.close()


def test_material_direction_rotates_terminal_and_preserves_arc_length(scene):
    from ibero.materials.cable import _flex_subspec

    p = CableParameters.from_scene(scene.config["materials"]["cable"])
    for direction in ([1.0, 0, 0], [0, 0, 1.0]):
        model = _flex_subspec(p, np.array(direction)).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        long_axis = data.geom_xmat[model.geom("cable_terminal_geom").id].reshape(3, 3)[
            :, 1
        ]
        np.testing.assert_allclose(long_axis, -np.array(direction), atol=1e-10)
        assert abs(model.flexedge_length0.sum() - p.length_m) < 1e-10


def test_extreme_bending_stiffness_rejected_by_timestep_bound(scene):
    config = copy.deepcopy(scene.config)
    config["materials"]["cable"]["bending_stiffness_nm2"] = 100.0
    with pytest.raises(SceneValidationError, match="material bound"):
        SceneLoader()._validate_physics(config, scene.constraints)
