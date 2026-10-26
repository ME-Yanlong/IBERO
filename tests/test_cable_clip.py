"""侧向线夹台架：真实线段压入、低载保持、超载拉出。"""

import mujoco
import numpy as np
import pytest

from ibero.mechanisms.cable_clip import (
    CableClipObserver,
    CableClipParameters,
    add_cable_clip,
)


def _bench():
    params = CableClipParameters()
    spec = mujoco.MjSpec()
    spec.modelname = "cable_clip_bench"
    spec.option.timestep = 2.5e-5
    spec.option.gravity = [0, 0, 0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    names = add_cable_clip(spec, params, name="test_clip")
    wire = spec.worldbody.add_body(name="wire", pos=[0, 0, 0.034])
    # 台架只允许法向位移；机器人任务必须改用自由 Flex 线段。
    wire.add_joint(
        name="wire_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        damping=0.3,
    )
    wire.add_geom(
        name="wire_capsule",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, -0.015, 0, 0, 0.015, 0],
        size=[params.wire_radius_m],
        mass=0.006,
        friction=[params.friction, 0.005, 0.0001],
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    observer = CableClipObserver(model, names, params)
    dof = model.jnt_dofadr[model.joint("wire_slide").id]
    return params, model, data, observer, dof


def _pulse(model, data, dof, force_n, steps):
    for _ in range(steps):
        data.qfrc_applied[dof] = force_n
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def test_clip_rejects_non_capture_geometry():
    with pytest.raises(ValueError, match="mouth must be narrower"):
        CableClipParameters(mouth_half_width_m=0.006)
    with pytest.raises(ValueError, match="seat wider"):
        CableClipParameters(seat_half_width_m=0.004)
    with pytest.raises(ValueError, match="fit behind the lip"):
        CableClipParameters(seat_depth_m=0.023)


def test_clip_builds_local_frame_and_deforms_on_contact():
    params, model, data, observer, dof = _bench()
    before = observer.observe(data)
    assert before["jaw_recovered"]
    assert before["jaw_gap_m"] == pytest.approx(2 * params.mouth_half_width_m)
    assert np.allclose(before["rotation_world"], np.eye(3))
    _pulse(model, data, dof, -8, 200)
    entering = observer.observe(data)
    assert not entering["jaw_recovered"]
    assert entering["jaw_gap_m"] > before["jaw_gap_m"]
    assert entering["max_penetration_m"] < entering["penetration_limit_m"]


def test_wire_reaches_seat_retains_under_small_pull_and_can_escape():
    params, model, data, observer, dof = _bench()
    _pulse(model, data, dof, -8, 1600)
    _pulse(model, data, dof, 0, 400)
    seated = observer.observe(data)
    z = float(data.xpos[model.body("wire").id, 2])
    assert params.seat_z_min_m <= z <= params.seat_z_max_m
    assert seated["jaw_recovered"]
    assert seated["max_penetration_m"] < seated["penetration_limit_m"]
    # 小拉力保持在唇边后；同一个模型加大拉力又能撑开夹臂而退出。
    _pulse(model, data, dof, 0.3, 500)
    z_retained = float(data.xpos[model.body("wire").id, 2])
    assert z_retained < params.seat_z_max_m
    _pulse(model, data, dof, 4, 1800)
    z_escaped = float(data.xpos[model.body("wire").id, 2])
    assert z_escaped > params.lip_z_m + params.wire_radius_m


def _flex_bench(params=None, *, initial_z=0.034):
    """使用真实 Flex 元素构建可控法向力台架。"""
    params = params or CableClipParameters()
    spec = mujoco.MjSpec()
    spec.modelname = "cable_clip_flex_bench"
    spec.option.timestep = 2.5e-5
    spec.option.gravity = [0, 0, 0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    names = add_cable_clip(spec, params, name="flex_clip")
    vertices = []
    for index in range(7):
        body = spec.worldbody.add_body(
            name=f"flex_vertex_{index}", pos=[0, (index - 3) * 0.006, initial_z]
        )
        vertices.append(body.name)
        for axis_index, axis in enumerate(np.eye(3)):
            body.add_joint(
                name=f"flex_vertex_{index}_{axis_index}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=axis,
                damping=0.03,
            )
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.002],
            mass=0.006 / 7,
            contype=0,
            conaffinity=0,
            rgba=[0, 0, 0, 0],
        )
    spec.add_flex(
        name="test_wire_flex",
        dim=1,
        radius=params.wire_radius_m,
        vertbody=vertices,
        vert=np.zeros(21),
        elem=np.column_stack((np.arange(6), np.arange(1, 7))).ravel(),
        edgestiffness=1000,
        edgedamping=0.1,
        contype=2,
        conaffinity=1,
        selfcollide=mujoco.mjtFlexSelf.mjFLEXSELF_NONE,
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
        friction=[params.friction, 0.005, 0.0001],
        rgba=[0.9, 0.2, 0.05, 1],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    observer = CableClipObserver(model, names, params)
    dofs = [
        model.jnt_dofadr[model.joint(f"flex_vertex_{index}_2").id]
        for index in range(7)
    ]
    mujoco.mj_forward(model, data)
    return params, model, data, observer, dofs


def test_flex_wire_contacts_jaws_and_releases_without_a_lock_switch():
    """真实 1D Flex 圆线与线夹接触；刚性胶囊台架不能替代此检查。"""
    params, model, data, observer, dofs = _flex_bench()
    saw_opening = False
    for force_n, steps in ((-8, 1600), (0, 400), (0.3, 500)):
        for _ in range(steps):
            data.qfrc_applied[dofs] = force_n / len(dofs)
            mujoco.mj_step(model, data)
            saw_opening |= observer.observe(data)["jaw_gap_m"] > 0.009
        mujoco.mj_forward(model, data)
    retained = observer.observe(data)
    assert saw_opening
    assert retained["jaw_recovered"]
    assert retained["max_penetration_m"] < retained["penetration_limit_m"]
    assert data.flexvert_xpos[:, 2].mean() < params.seat_z_max_m
    for _ in range(1800):
        data.qfrc_applied[dofs] = 4 / len(dofs)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert data.flexvert_xpos[:, 2].mean() > params.lip_z_m + params.wire_radius_m


def test_soft_clip_retains_a_wire_resting_close_to_lip():
    params, model, data, observer, dofs = _flex_bench(
        CableClipParameters(young_pa=2e8), initial_z=0.021
    )
    # 整个圆截面不必完全退到唇边背面；真实唇边仍会阻止小幅外拉。
    assert params.seat_z_max_m == pytest.approx(0.024)
    for _ in range(1600):
        data.qfrc_applied[dofs] = 0.1 / len(dofs)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    retained = observer.observe(data)
    z = float(data.flexvert_xpos[:, 2].mean())
    assert 0.021 < z < params.seat_z_max_m
    assert retained["jaw_recovered"]
    assert retained["jaw_gap_m"] < 2 * retained["wire_radius_m"]
    # 增大同向拉力，夹臂再次张开并让线束真正越过唇边。
    for _ in range(1600):
        data.qfrc_applied[dofs] = 0.5 / len(dofs)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert data.flexvert_xpos[:, 2].mean() > params.lip_z_m + params.wire_radius_m


def test_soft_clip_accepts_flex_with_lower_normal_force():
    params, model, data, observer, dofs = _flex_bench(
        CableClipParameters(young_pa=2e8)
    )
    peak_penetration = 0.0
    for step in range(2000):
        data.qfrc_applied[dofs] = -0.4 / len(dofs)
        mujoco.mj_step(model, data)
        if step % 20 == 0:
            peak_penetration = max(
                peak_penetration, observer.observe(data)["max_penetration_m"]
            )
    for _ in range(400):
        data.qfrc_applied[dofs] = 0
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    state = observer.observe(data)
    assert data.flexvert_xpos[:, 2].mean() < params.seat_z_max_m
    assert state["jaw_recovered"]
    assert state["jaw_gap_m"] < 2 * params.wire_radius_m
    assert peak_penetration < state["penetration_limit_m"]
