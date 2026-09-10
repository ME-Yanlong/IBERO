"""力必须改变真实动力学并到达 F/T；隐式预测不能写回物体状态。"""

import mujoco
import numpy as np
import pytest
from ibero.benches.milling import MillingFixture
from ibero.materials.parameters import StockParameters
from ibero.processes.tools import EndMillGeometry
from ibero.processes.milling_forces import MillingCoefficients, MillingLimits
from ibero.core.loads import PhysicalLoads


def fixture(*, max_force=50, start=(-0.025, 0, 0.025)):
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
