"""独立材料试验台：固定端夹具与解析拉伸/弯曲检查，不依赖机器人脚本。"""

from dataclasses import replace
import mujoco
import numpy as np

from ibero.materials.cable import _flex_subspec
from ibero.materials.mechanics import CableMechanics, bending_energy_force


def axial_fixture(params, extension=0.01, duration=0.8):
    # 试验夹具直接固定两个端头，只在此材料校准工具中使用。任务抓取不使用。
    p = replace(params, initial_span_m=params.length_m + params.terminal_length_m)
    spec = _flex_subspec(p, np.array([1.0, 0, 0]))
    for joint in list(spec.joints):
        if joint.name.startswith(("cable_terminal_", "cable_tail_")):
            spec.delete(joint)
    for geom in spec.geoms:
        geom.contype = geom.conaffinity = 0
    spec.option.timestep = 0.0001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity = [0, 0, 0]
    model = spec.compile()
    data = mujoco.MjData(model)
    mechanics = CableMechanics(model, p)
    tail = model.body("cable_tail_terminal").id
    nominal = model.body_pos[tail].copy()
    mujoco.mj_forward(model, data)
    # 逐步移动固定夹具，防止瞬时位移冲击污染静态刚度读数。
    for step in range(int(duration / model.opt.timestep)):
        model.body_pos[tail, 0] = nominal[0] + extension * min(
            1, step * model.opt.timestep / 0.2
        )
        mechanics.apply(data)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    state = mechanics.state(data)
    expected = p.axial_stiffness_n_m * extension
    measured = np.mean(state["endpoint_tension_n"])
    return {
        "segments": p.segments,
        "extension_m": extension,
        "expected_n": expected,
        "measured_n": float(measured),
        "relative_error": float(abs(measured - expected) / max(abs(expected), 1e-9)),
        "mass_kg": float(model.body_mass[mechanics.bodies].sum()),
        "expected_mass_kg": p.length_m * p.linear_density_kg_m,
        "state": state,
    }


def bending_gradient_check():
    points = np.array([[0.0, 0, 0], [0.02, 0, 0.005], [0.04, 0, 0], [0.06, 0.002, 0]])
    energy, force = bending_energy_force(points, 0.0002, 0.02)
    numerical = np.zeros_like(points)
    for i in range(len(points)):
        for j in range(3):
            a = points.copy()
            b = points.copy()
            a[i, j] += 1e-7
            b[i, j] -= 1e-7
            numerical[i, j] = (
                -(
                    bending_energy_force(a, 0.0002, 0.02)[0]
                    - bending_energy_force(b, 0.0002, 0.02)[0]
                )
                / 2e-7
            )
    return {
        "energy_j": energy,
        "gradient_max_error": float(np.max(abs(force - numerical))),
        "net_force_n": float(np.linalg.norm(force.sum(axis=0))),
        "net_torque_nm": float(np.linalg.norm(np.cross(points, force).sum(axis=0))),
    }


def sag_fixture(params, rigidity, seconds=1.5):
    """两端固定、全重力悬垂；支撑反力应等于线束自重，EI 增大应减少下垂。"""
    p = replace(
        params,
        initial_span_m=params.length_m + params.terminal_length_m,
        bending_stiffness_nm2=rigidity,
    )
    spec = _flex_subspec(p, np.array([1.0, 0, 0]))
    for joint in list(spec.joints):
        if joint.name.startswith(("cable_terminal_", "cable_tail_")):
            spec.delete(joint)
    for geom in spec.geoms:
        geom.contype = geom.conaffinity = 0
    spec.option.timestep = 0.0001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = spec.compile()
    data = mujoco.MjData(model)
    mechanics = CableMechanics(model, p)
    mujoco.mj_forward(model, data)
    reference = data.flexvert_xpos[0, 2]
    samples = []
    for step in range(int(seconds / model.opt.timestep)):
        mechanics.apply(data)
        mujoco.mj_step(model, data)
        if step > int((seconds - 0.3) / model.opt.timestep):
            vertical = 0.0
            for name in ("terminal_to_flex", "tail_to_flex"):
                eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
                rows = np.flatnonzero(
                    (
                        data.efc_type[: data.nefc]
                        == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                    )
                    & (data.efc_id[: data.nefc] == eq)
                )
                vertical += data.efc_force[rows[2]]
            samples.append(
                (
                    abs(vertical),
                    reference
                    - float(data.flexvert_xpos[len(mechanics.bodies) // 2, 2]),
                )
            )
    load, sag = np.mean(samples, axis=0)
    return {
        "rigidity_nm2": rigidity,
        "sag_m": float(sag),
        "support_force_n": float(load),
        "expected_weight_n": p.length_m * p.linear_density_kg_m * 9.81,
        "load_error": float(
            abs(load - p.length_m * p.linear_density_kg_m * 9.81)
            / (p.length_m * p.linear_density_kg_m * 9.81)
        ),
    }
