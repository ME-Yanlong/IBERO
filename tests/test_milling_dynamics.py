"""力必须改变真实动力学并到达 F/T；隐式预测不能写回物体状态。"""

import mujoco
import numpy as np
import pytest
from ibero.benches.milling import MillingFixture
from ibero.materials.parameters import StockParameters
from ibero.processes.tools import EndMillGeometry
from ibero.processes.milling_forces import MillingCoefficients, MillingLimits
from ibero.core.loads import PhysicalLoads


def fixture(*, max_force=50, start=(-0.025, 0, 0.025), ft_quaternion=(1, 0, 0, 0)):
    c = MillingCoefficients(
        1.8e9,
        0.6e9,
        0.2e9,
        100,
        50,
        20,
        1.8e9,
        0.4e9,
        "AISI1045-provisional",
        "provisional",
    )
    limits = MillingLimits(1000, 12000, 0.0001, 0.006, max_force, 0.15, 100, 2, True)
    return MillingFixture(
        StockParameters(size_m=(0.006, 0.006, 0.002)),
        0.002,
        EndMillGeometry(0.004, 0.012, 0.005, 0.02),
        c,
        limits,
        start=start,
        ft_quaternion=ft_quaternion,
    )


def test_body_loads_do_not_repeat_and_preserve_additive_external_force():
    e = fixture()
    b = e.process.tool_body
    loads = PhysicalLoads(e.model)
    e.data.xfrc_applied[b, 0] = 2
    loads.begin(e.data)
    loads.add_wrench(e.data, b, [1, 0, 0], [0, 0, 0], e.data.xipos[b])
    loads.commit(e.data)
    assert e.data.xfrc_applied[b, 0] == 3
    e.data.xfrc_applied[b, 0] += 0.5
    loads.begin(e.data)
    loads.commit(e.data)
    assert e.data.xfrc_applied[b, 0] == 2.5
    assert not e.data.qfrc_applied.any()


def test_quasistatic_ft_reports_applied_force_and_moment_arm():
    e = fixture()
    force, torque = np.array([1.0, -2.0, 3.0]), np.array([0.01, 0.02, -0.025])
    for _ in range(3000):
        e.loads.begin(e.data)
        point = e.data.site("mill_tip").xpos.copy()
        e.loads.add_wrench(e.data, e.process.tool_body, force, torque, point)
        e.loads.commit(e.data)
        mujoco.mj_step(e.model, e.data)
        mujoco.mj_forward(e.model, e.data)
    lever = e.data.site("mill_tip").xpos - e.data.site("mill_ft_site").xpos
    np.testing.assert_allclose(e.data.sensor("mill_force").data, -force, atol=1e-6)
    np.testing.assert_allclose(
        e.data.sensor("mill_torque").data, -(torque + np.cross(lever, force)), atol=1e-6
    )
    assert np.linalg.norm(e.data.qvel[:3]) < 1e-6


def test_implicit_force_solves_stiff_drag_without_overwriting_physics():
    e = fixture()
    # 单独的数值辨识初态，取消平移伺服，仅检查质量—速度阻力的闭式隐式解。
    for name in ("mill_axis_0", "mill_axis_1", "mill_axis_2"):
        i = e.model.actuator(name).id
        e.model.actuator_gainprm[i] = 0
        e.model.actuator_biasprm[i] = 0
    e.data.qvel[2] = -0.0002
    e.data.joint("mill_spindle").qvel[0] = 600
    e.data.ctrl[e.model.actuator("mill_motor").id] = 600
    mujoco.mj_forward(e.model, e.data)
    q, v = e.data.qpos.copy(), e.data.qvel.copy()
    damping = 16000.0

    def law(u):
        return np.array([0, 0, -damping * u[2], 0, 0, 0])

    e.loads.begin(e.data)
    point = e.data.site("mill_tip").xpos.copy()
    wrench, response, error = e.process.coupling.solve(
        e.data, e.loads.generalized, e.loads.external_body, point, law
    )
    np.testing.assert_array_equal(e.data.qpos, q)
    np.testing.assert_array_equal(e.data.qvel, v)
    assert error < 1e-8
    expected = -0.0002 / (1 + e.model.opt.timestep * damping / 0.6)
    assert response[2] == pytest.approx(expected, rel=1e-8)
    e.loads.add_wrench(e.data, e.process.tool_body, wrench[:3], wrench[3:], point)
    e.loads.commit(e.data)
    mujoco.mj_step(e.model, e.data)
    assert e.data.qvel[2] == pytest.approx(expected, rel=1e-8)


def test_overload_prevents_both_material_commit_and_integration():
    e = fixture(max_force=0.01, start=(0, 0, 0.0011))
    # reset 级设定已有转速与微小进给，之后 step 不能通过瞬移绕过超载。
    e.data.joint("mill_spindle").qvel[0] = 6000 * np.pi / 30
    e.data.qvel[2] = -0.0002
    mujoco.mj_forward(e.model, e.data)
    q, h = e.data.qpos.copy(), e.stock.state_hash()
    info = e.step((0, 0, 0), 6000, substeps=1)
    assert info["invalid_reason"] == "cutting_force_overload"
    np.testing.assert_array_equal(e.data.qpos, q)
    assert e.stock.state_hash() == h and e.data.time == 0
    with pytest.raises(RuntimeError, match="reset"):
        e.step((0, 0, 0), 6000)


def test_faulted_cut_rolls_back_and_requires_reset():
    e = fixture(start=(0, 0, 1e-8))
    e.data.joint("mill_spindle").qvel[0] = 6000 * np.pi / 30
    e.data.qvel[2] = -0.0002
    mujoco.mj_forward(e.model, e.data)
    q, h = e.data.qpos.copy(), e.stock.state_hash()
    with pytest.raises(RuntimeError, match="Injected"):
        e.step((0, 0, -0.001), 6000, substeps=1, fault_at="after_forward")
    np.testing.assert_array_equal(e.data.qpos, q)
    assert e.stock.state_hash() == h and e.data.time == 0
    assert not e.loads._open
    e.binding.ensure_consistent()
    with pytest.raises(RuntimeError, match="reset"):
        e.step((0, 0, -0.001), 6000)
    e.reset(seed=0)
    assert not e._done and e.stock.occupied.all()


def test_stopped_tool_is_solid_and_cannot_remove_stock():
    e = fixture(start=(0, 0, 0.0011))
    peak_contacts = 0
    for _ in range(30):
        info = e.step((0, 0, -0.004), 0)
        peak_contacts = max(peak_contacts, e.data.ncon)
    assert peak_contacts > 0
    assert e.data.site("mill_tip").xpos[2] > 0
    assert info["mode"] == "solid_contact" and e.stock.version == 0


def test_air_cut_is_zero_load_and_idempotent_material():
    e = fixture()
    for _ in range(10):
        info = e.step(e.start + [0.001, 0, 0], 6000)
    assert info["mode"] == "air_cut" and info["invalid_reason"] is None
    assert e.stock.version == 0 and np.linalg.norm(info["force_world_n"]) == 0


def test_noncutting_housing_retains_contact_when_blade_mask_changes():
    # 台架级构型诊断：已空的刀刃区域不消除更粗刀柄外圈的实体。
    e = fixture(start=(0.006, 0, -0.012))
    housing = e.model.geom("mill_housing").id
    e.model.geom_contype[e.process.blade_geom] = 2
    mujoco.mj_forward(e.model, e.data)
    assert e.model.geom_contype[housing] == 1
    assert any(housing in (c.geom1, c.geom2) for c in e.data.contact)


def test_milling_replay_restores_blade_mode_as_well_as_voxels(tmp_path):
    from ibero.core.stock_trace import StockTrace

    e = fixture()
    trace = StockTrace(e)
    trace.append(e.last_info)
    for _ in range(5):
        info = e.step(e.start, 6000)
    assert e.model.geom_contype[e.process.blade_geom] == 2
    trace.append(info)
    path = tmp_path / "milling.npz"
    trace.save(path)
    loaded = StockTrace(e).load(path)
    assert e.model.geom_contype[e.process.blade_geom] == 1
    loaded.restore(1)
    assert e.model.geom_contype[e.process.blade_geom] == 2
    with pytest.raises(RuntimeError, match="reset"):
        e.step(e.start, 6000)
    loaded.restore(0)
    assert e.model.geom_contype[e.process.blade_geom] == 1


def test_rotated_ft_and_equal_opposite_world_wrench():
    e = fixture(ft_quaternion=(np.sqrt(0.5), 0, 0, np.sqrt(0.5)))
    # 传感器局部坐标旋转 90 度，不改变刚体构型；检查轴交换及力矩符号。
    ft = e.model.site("mill_ft_site").id
    mujoco.mj_forward(e.model, e.data)
    force, torque = np.array([1.0, 2.0, -3.0]), np.array([0.03, -0.02, 0.01])
    for _ in range(3000):
        point = e.data.site("mill_tip").xpos.copy() + [0.001, -0.002, 0.003]
        e.loads.begin(e.data)
        e.loads.add_wrench(e.data, e.process.tool_body, force, torque, point)
        e.loads.add_wrench(e.data, e.process.stock_body, -force, -torque, point)
        e.loads.commit(e.data)
        np.testing.assert_allclose(
            e.data.xfrc_applied[:, :3].sum(axis=0), 0, atol=1e-12
        )
        net_moment = e.data.xfrc_applied[:, 3:] + np.cross(
            e.data.xipos, e.data.xfrc_applied[:, :3]
        )
        np.testing.assert_allclose(net_moment.sum(axis=0), 0, atol=1e-12)
        mujoco.mj_step(e.model, e.data)
        mujoco.mj_forward(e.model, e.data)
    rotation = e.data.site_xmat[ft].reshape(3, 3)
    np.testing.assert_allclose(rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    lever = point - e.data.site_xpos[ft]
    np.testing.assert_allclose(
        e.data.sensor("mill_force").data, -rotation.T @ force, atol=1e-6
    )
    np.testing.assert_allclose(
        e.data.sensor("mill_torque").data,
        -rotation.T @ (torque + np.cross(lever, force)),
        atol=1e-6,
    )


def test_repeated_reset_clears_material_loads_and_preserves_handles():
    import gc
    import tracemalloc

    e = fixture()
    identities = (id(e.model), id(e.data), id(e.stock), id(e.binding))
    geom_ids = e.binding.geom_ids.copy()
    tracemalloc.start()
    samples = []
    try:
        for index in range(270):
            e.binding.commit(
                e.stock.prepare_removal([0, 1], f"reset-cycle-{index}"), e.data
            )
            e.loads.begin(e.data)
            e.loads.add_wrench(
                e.data,
                e.process.tool_body,
                [1, 2, 3],
                [0, 0, 0.01],
                e.data.site("mill_tip").xpos,
            )
            e.loads.commit(e.data)
            e.process.sequence = 1
            e.process.invalid_reason = "test-reset"
            e.model.geom_contype[e.process.blade_geom] = 2
            e.reset(seed=0)
            assert not e.stock.events and not e.stock._events_by_id
            assert e.stock.version == 0 and e.stock.occupied.all()
            assert not e.data.qfrc_applied.any() and not e.data.xfrc_applied.any()
            assert not e.loads._last_body.any() and not e.loads._open
            assert e.process.sequence == 0 and e.process.invalid_reason is None
            assert e.model.geom_contype[e.process.blade_geom] == 1
            assert identities == (id(e.model), id(e.data), id(e.stock), id(e.binding))
            np.testing.assert_array_equal(e.binding.geom_ids, geom_ids)
            e.binding.ensure_consistent()
            if index >= 20 and index % 10 == 0:
                gc.collect()
                samples.append(tracemalloc.get_traced_memory()[0])
        # 预热后 250 回合的受跟踪活跃内存界；不是对全部原生内存的无限期无泄漏保证。
        assert max(samples) - min(samples) < 1024 * 1024
    finally:
        tracemalloc.stop()


def test_implicit_face_activation_boundary_has_balanced_solution():
    e = fixture()
    e.data.ctrl[e.model.actuator("mill_axis_2").id] = -1 / 5000
    e.data.joint("mill_spindle").qvel[0] = 600
    e.data.ctrl[e.model.actuator("mill_motor").id] = 600
    mujoco.mj_forward(e.model, e.data)

    def piecewise_wrench(u):
        down = max(-u[2], 0)
        return np.array([0, 0, 1 + 16000 * down, 0, 0, -0.03 - 144 * down])

    e.loads.begin(e.data)
    point = e.data.site("mill_tip").xpos.copy()
    w, u, error = e.process.coupling.solve(
        e.data, e.loads.generalized, e.loads.external_body, point, piecewise_wrench
    )
    assert abs(u[2]) < 1e-9 and error < 1e-8
    e.loads.add_wrench(e.data, e.process.tool_body, w[:3], w[3:], point)
    e.loads.commit(e.data)
    mujoco.mj_step(e.model, e.data)
    assert abs(e.data.qvel[2]) < 1e-9


def test_invalid_commands_and_trace_frames_are_rejected_before_mutation():
    from ibero.core.stock_trace import StockTrace

    e = fixture()
    trace = StockTrace(e)
    for bad in (float("nan"), float("inf"), -1, True):
        with pytest.raises(ValueError):
            e.step(e.start, bad)
    assert e.data.time == 0 and e.stock.version == 0
    for info in ({"bad": float("nan")}, [], None):
        with pytest.raises(ValueError):
            trace.append(info)
    assert not trace.states and not trace.infos and not trace.collision_modes
    for seed in (True, -1, 2**32):
        with pytest.raises(ValueError):
            e.reset(seed=seed)
