"""机器人工具/场景/任务边界；完整动力学成功率由长测单独验收。"""

import copy
from dataclasses import replace
import mujoco
import numpy as np
import pytest
from ibero.envs.latch_release import LatchReleaseEnv
from ibero.core.industrial_config import validate_industrial
from ibero.robots.g1_industrial import solve_reset_pose


@pytest.fixture(scope="module")
def env():
    obj = LatchReleaseEnv()
    yield obj
    obj.close()


def test_real_free_object_and_original_robot_limits(env):
    assert env.model.joint("plug_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    with pytest.raises(KeyError):
        env.model.joint("plug_slide")
    assert len(env.handles.gripper_actuator_ids) == 1
    np.testing.assert_allclose(
        env.model.jnt_actfrcrange[env.model.joint("right_wrist_pitch_joint").id],
        [-5, 5],
    )
    np.testing.assert_allclose(
        env.model.jnt_actfrcrange[env.model.joint("left_shoulder_pitch_joint").id],
        [-25, 25],
    )
    assert all(
        not env.model.geom(i).name.startswith("left_fingers")
        for i in range(env.model.ngeom)
    )


def test_seed_changes_real_geometry_and_friction(env):
    _, a = env.reset(seed=100)
    first = copy.deepcopy(env.resolved_config)
    env.reset(seed=101)
    second = env.resolved_config
    assert first["initialization"]["origin_m"] != second["initialization"]["origin_m"]
    assert first["mechanism"]["friction"] != second["mechanism"]["friction"]
    _, b = env.reset(seed=100)
    assert first == env.resolved_config
    assert a["clearance_m"] == b["clearance_m"]
    assert env.elapsed_steps == 0 and not env.invalid_reason and not env.events


@pytest.mark.parametrize("bad", ["tool", "jitter", "unknown", "quaternion", "force"])
def test_unsupported_robot_recipe_rejected(env, bad):
    c = copy.deepcopy(env.scene.config)
    if bad == "tool":
        c["robot"]["left_tool"] = "hidden_gripper"
    elif bad == "jitter":
        c["initialization"]["origin_jitter_m"] = [-0.001, 0, 0]
    elif bad == "unknown":
        c["control"]["unlock_weld"] = True
    elif bad == "quaternion":
        c["initialization"]["quaternion_wxyz"] = [0, 0, 0, 0]
    else:
        c["control"]["max_pull_force_n"] = 10
    with pytest.raises(ValueError):
        validate_industrial(c, env.scene.constraints)


def test_task_requires_full_withdrawal_grasp_hold_and_no_damage(env):
    env.reset(seed=0)
    task = env.task
    state = env._task_state()
    x = dict(state.extra, released=True, latch_state="released", withdrawal_m=0.012)
    state = replace(state, extra=x, grasps={"right": True})
    assert not task.evaluate(state, None).success
    x = dict(x, withdrawal_m=0.08, pose_error_rad=0)
    state = replace(state, extra=x)
    for _ in range(24):
        assert not task.evaluate(state, None).success
    assert task.evaluate(state, None).success
    damaged = replace(state, extra=dict(x, invalid_reason="force_out_of_model_range"))
    result = task.evaluate(damaged, None)
    assert not result.success and result.terminated


def test_press_path_kinematic_self_collision_check(env):
    env.reset(seed=0)
    site = "left_press_tcp"
    initial = env.data.site(site).xpos.copy()
    rotation = env.data.site(site).xmat.reshape(3, 3).copy()
    for dz in (0, 0.005, 0.01, 0.0165):
        target = initial - [0, 0, dz]
        solve_reset_pose(env.model, env.data, "left", site, target, rotation)
        for c in env.data.contact:
            a, b = int(c.geom1), int(c.geom2)
            if a in env.robot_geoms and b in env.robot_geoms:
                assert -c.dist <= 0.0001
    env.reset(seed=0)


def test_trace_roundtrip_and_replay_cannot_resume(env, tmp_path):
    from ibero.review import Trace

    env.reset(seed=103)
    trace = Trace(env)
    trace.append(env.last_info)
    expected = env.data.qpos.copy()
    path = tmp_path / "trace.npz"
    trace.save(path)
    env.reset(seed=0)
    loaded = Trace(env).load(path)
    assert loaded.restore(0)["replay"]
    np.testing.assert_array_equal(env.data.qpos, expected)
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(np.zeros(14))
    env.reset(seed=0)
    assert not env._replay_restored


def test_trace_rejects_different_recipe(env, tmp_path):
    from ibero.review import Trace

    env.reset(seed=0)
    trace = Trace(env)
    trace.append(env.last_info)
    path = tmp_path / "trace.npz"
    trace.save(path)
    original = env.scene
    try:
        env.scene = replace(original, config_hash="different-material")
        with pytest.raises(ValueError, match="mismatch"):
            Trace(env).load(path)
    finally:
        env.scene = original
