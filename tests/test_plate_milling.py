"""机器人加工不靠隐藏状态修正：装配能力、命令边界与实际探针。"""

import copy
from pathlib import Path
import mujoco
import numpy as np
import pytest
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.core.plate_milling_config import validate_plate_milling
from ibero.core.scene_loader import SceneLoader
from ibero.core.stock_trace import StockTrace
from ibero.control.milling_bench import MillingBenchScript, bench_target
from ibero.materials.stock import VoxelStock
from ibero.materials.parameters import StockParameters
from ibero.benches.stock_probe import inspect_machined_stock
from ibero.processes.tools import ToolPose, EndMillGeometry, swept_cells

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def env():
    return PlateMillingEnv()


def test_spindle_does_not_inherit_g1_arm_friction_or_armature(env):
    m = env.model
    spindle = m.joint("mill_spindle")
    assert m.dof_frictionloss[spindle.dofadr][0] == 0
    assert m.dof_armature[spindle.dofadr][0] == 0
    wrist = m.joint("left_wrist_pitch_joint")
    assert m.dof_frictionloss[wrist.dofadr][0] == pytest.approx(0.3)
    assert m.dof_armature[wrist.dofadr][0] == pytest.approx(0.01)
    np.testing.assert_array_equal(m.jnt_actfrcrange[wrist.id], [-5, 5])
    assert m.jnt_actfrcrange[m.joint("left_shoulder_pitch_joint").id, 1] == 25
    assert m.body("mill_rotor").mass[0] == pytest.approx(0.02)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_press_tool") == -1


def test_servo_only_changes_original_commands_and_rotation_is_physical(env):
    q, v = env.data.qpos.copy(), env.data.qvel.copy()
    arms = env.model.jnt_actfrcrange.copy()
    env.servo.apply(env.data, env.start + [0.0001, 0, 0])
    np.testing.assert_array_equal(env.data.qpos, q)
    np.testing.assert_array_equal(env.data.qvel, v)
    np.testing.assert_array_equal(env.model.jnt_actfrcrange, arms)
    assert not env.data.qfrc_applied.any()
    for _ in range(20):
        info = env.step(env.start, 6000)
        assert not info["invalid_reason"]
    assert info["actual_rpm"] > 5700
    assert env.stock.version == 0
    assert info["peak_actual_joint_limit_fraction"] <= 1


def test_controller_waits_for_actual_spindle_and_transforms_local_waypoints(env):
    p = MillingBenchScript(env, bench_target("slot"))
    p.command(env)
    assert not p.spindle_ready
    np.testing.assert_allclose(
        p.waypoints[0][0], env.stock.origin + [-0.004, 0, 0.0035]
    )
    # 纯控制器反事实：经过等待时间但转速仍是零，不准推进名义进刀目标。
    env.data.time = 2.0
    initial = p.nominal.copy()
    target, rpm = p.command(env)
    assert p.index == 0 and not p.spindle_ready and rpm == 6000
    np.testing.assert_array_equal(target, initial)


def test_plate_config_rejects_fake_gravity_and_unbounded_axis_approximation():
    scene = SceneLoader().validate(ROOT / "scenes/plate_milling")
    for field, value in [
        ("axis_tolerance_rad", 0.02),
        ("holder_angular_limit_rad_s", 1.0),
    ]:
        cfg = copy.deepcopy(scene.config)
        cfg["robot"][field] = value
        with pytest.raises(ValueError):
            validate_plate_milling(cfg, scene.constraints)
    cfg = copy.deepcopy(scene.config)
    cfg["physics"]["gravity_m_s2"] = [0, 0, 0]
    with pytest.raises(ValueError):
        validate_plate_milling(cfg, scene.constraints)


def test_task_cannot_succeed_for_solid_stock_or_spinning_tool(env):
    assert not env.evaluate_task(env.target)["result"]["success"]
    with pytest.raises(ValueError, match="declared"):
        env.evaluate_task(bench_target("slot"))


def test_robot_trace_reset_seed_and_readonly_replay(env, tmp_path):
    env.reset(seed=3)
    q = env.data.qpos.copy()
    env.reset(seed=3)
    np.testing.assert_array_equal(q, env.data.qpos)
    env.reset(seed=4)
    assert not np.array_equal(q, env.data.qpos)
    t = StockTrace(env)
    t.append(env.last_info)
    t.append(env.step(env.start, 6000))
    t.save(tmp_path / "robot.npz")
    loaded = StockTrace(env).load(tmp_path / "robot.npz")
    loaded.restore(1)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.start, 0)
    env.reset(seed=4)
    assert env.stock.version == 0 and not env._replay_restored


def test_different_p4_target_replay_rejected_before_changing_env(env, tmp_path):
    trace = StockTrace(env)
    trace.append(env.last_info)
    trace.save(tmp_path / "hole.npz")
    other = PlateMillingEnv(shape="slot")
    q = other.data.qpos.copy()
    assert env.manifest()["scene_hash"] != other.manifest()["scene_hash"]
    with pytest.raises(ValueError, match="mismatch"):
        StockTrace(other).load(tmp_path / "hole.npz")
    np.testing.assert_array_equal(q, other.data.qpos)


def test_invalid_servo_gains_fail_before_commands(env):
    from ibero.control.milling import MillingArmServo

    ctrl = env.data.ctrl.copy()
    for gain in (float("nan"), -1, True):
        with pytest.raises(ValueError):
            MillingArmServo(env.model, env.data, position_kp=gain)
        np.testing.assert_array_equal(ctrl, env.data.ctrl)


def test_translated_actual_hole_probe_and_material_are_consistent():
    stock = VoxelStock(
        StockParameters(size_m=(0.024, 0.02, 0.004)), 0.001, origin=[0.45, 0.16, 0.85]
    )
    tool = EndMillGeometry(0.004, 0.012, 0.005, 0.02)
    pose = ToolPose(tuple(stock.origin + [0, 0, -0.003]))
    event = stock.prepare_removal(swept_cells(stock, tool, pose, pose), "probe-test")
    stock.commit(event)
    expected = stock.state_hash()
    report = inspect_machined_stock(stock, bench_target("through_hole"))
    assert report["passed"] and stock.state_hash() == expected
    assert report["world_origin_m"] == [0.45, 0.16, 0.85]
