"""组合材料质量、连接、力通道和任务因果；长程成功率独立计数。"""

from dataclasses import replace
import copy
import mujoco
import numpy as np
import pytest
from ibero.envs.harness_unplug import HarnessUnplugEnv
from ibero.core.industrial_config import validate_industrial


@pytest.fixture(scope="module")
def env():
    e = HarnessUnplugEnv()
    yield e
    e.close()


def test_one_free_plug_mass_arc_length_attachment(env):
    env.reset(seed=0)
    assert env.model.nflex == 1
    assert env.model.joint("plug_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert env.model.body("latch_plug").mass[0] == pytest.approx(0.06)
    assert env.model.body_mass[env.mechanics.bodies].sum() == pytest.approx(0.009)
    assert env.model.flexedge_length0.sum() == pytest.approx(0.3)
    eq = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "harness_material_attachment"
    )
    assert env.model.eq_type[eq] == mujoco.mjtEq.mjEQ_CONNECT
    assert env.model.eq_obj1id[eq] == env.plug
    assert env.model.eq_obj2id[eq] == env.mechanics.bodies[0]
    assert not any(env.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    np.testing.assert_allclose(
        env.data.flexvert_xpos[0],
        env.data.xpos[env.plug]
        + env.data.xmat[env.plug].reshape(3, 3) @ [-0.032, 0, -0.008],
        atol=1e-12,
    )


def test_material_forces_act_only_on_nodes_and_sum_to_zero(env):
    env.reset(seed=1)
    ctrl, passive = env.data.ctrl.copy(), env.data.qfrc_passive.copy()
    env._before_substep()
    np.testing.assert_array_equal(env.data.ctrl, ctrl)
    np.testing.assert_array_equal(env.data.qfrc_passive, passive)
    force = env.data.qfrc_applied[env.mechanics.indices]
    assert np.linalg.norm(force) > 0
    np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-12)
    other = np.ones(env.model.nv, dtype=bool)
    other[env.mechanics.indices.ravel()] = False
    assert np.all(env.data.qfrc_applied[other] == 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("length_m", True),
        ("damping", float("nan")),
        ("segments", 100),
        ("source", "measured"),
    ],
)
def test_invalid_cable_parameters_rejected(env, field, value):
    c = copy.deepcopy(env.scene.config)
    c["materials"]["cable"][field] = value
    with pytest.raises(ValueError):
        validate_industrial(c, env.scene.constraints)


def test_place_requires_prior_real_withdrawal_and_stable_support(env):
    env.reset(seed=0)
    state = env._task_state()
    x = dict(
        state.extra,
        receiver_supported=True,
        receiver_contains_plug=True,
        plug_speed_m_s=0,
        plug_angular_speed_rad_s=0,
        released=True,
    )
    state = replace(state, extra=x)
    for _ in range(60):
        assert not env.task.evaluate(state, None).success
    pulled = replace(
        state,
        extra=dict(x, withdrawal_m=0.08, pose_error_rad=0),
        grasps={"right": True},
    )
    assert not env.task.evaluate(pulled, None).success
    for _ in range(49):
        assert not env.task.evaluate(state, None).success
    assert env.task.evaluate(state, None).success
    invalid = replace(state, extra=dict(x, invalid_reason="cable_over_tension"))
    assert not env.task.evaluate(invalid, None).success
    env.reset(seed=0)
    assert not env.task.withdrawn and env.peak_tension_n == 0
