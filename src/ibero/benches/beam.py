"""独立弹性梁验证：解析解是参考，MuJoCo 只产生被比较的测量。"""

import time
import mujoco
import numpy as np

from ibero.materials.elastic_beam import add_elastic_beam
from ibero.materials.parameters import BeamParameters


def beam_static_test(*, segments=8, timestep=0.00001, load_n=0.01):
    p = BeamParameters()
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = timestep
    spec.option.gravity = [0, 0, 0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    root = spec.worldbody.add_body(name="fixed_root")
    last, names = add_elastic_beam(root, p, segments)
    for geom in spec.geoms:
        geom.contype = geom.conaffinity = 0
    model = spec.compile()
    data = mujoco.MjData(model)
    tip = model.site(names.tip_site).id
    body = model.body(last.name).id
    samples = []
    for step in range(round(0.16 / timestep)):
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = 0
        force = -load_n * min(1.0, data.time / 0.02) if data.time < 0.08 else 0.0
        mujoco.mj_applyFT(
            model,
            data,
            [0, 0, force],
            [0, 0, 0],
            data.site_xpos[tip],
            body,
            data.qfrc_applied,
        )
        mujoco.mj_step(model, data)
        if step % max(1, round(0.001 / timestep)) == 0:
            mujoco.mj_forward(model, data)
            samples.append(
                {
                    "time_s": float(data.time),
                    "load_n": -force,
                    "deflection_m": float(-data.site_xpos[tip, 2]),
                }
            )
    measured = np.mean(
        [r["deflection_m"] for r in samples if 0.06 < r["time_s"] < 0.079]
    )
    expected = load_n * p.length_m**3 / (3 * p.rigidity_nm2)
    residual = abs(samples[-1]["deflection_m"]) / expected
    return {
        "reference": "Euler-Bernoulli small-deflection end load",
        "segments": segments,
        "timestep_s": timestep,
        "expected_deflection_m": expected,
        "measured_deflection_m": float(measured),
        "stiffness_relative_error": float(
            abs(load_n / measured - p.tip_stiffness_n_m) / p.tip_stiffness_n_m
        ),
        "residual_fraction": float(residual),
        "trace": samples,
    }


def flex_candidate_test(
    *, timestep=0.00001, seconds=0.05, integrator="implicitfast", damping_s=None
):
    """同尺寸三维连续体候选；完整保留不稳定/锁定/精度差的结果，不作为默认。"""
    p = BeamParameters()
    # 允许无阻尼诊断以区分积分不稳定与空间离散误差，但明确改变物理假设，不能据此替换默认。
    damping = p.relaxation_time_s if damping_s is None else damping_s
    if integrator not in {"implicitfast", "implicit", "Euler"} or damping < 0:
        raise ValueError("Unsupported diagnostic integrator/damping")
    xml = f'''<mujoco><option timestep="{timestep}" gravity="0 0 0" integrator="{integrator}"/>
    <worldbody><flexcomp name="beam" type="grid" count="9 3 3" spacing=".0075 .006 .001"
    pos=".03 0 0" dim="3" mass="{p.mass_kg}" radius=".0001">
    <elasticity young="{p.young_pa}" poisson=".3" damping="{damping}"/>
    <contact contype="0" conaffinity="0"/><pin id="0 1 2 3 4 5 6 7 8"/>
    </flexcomp></worldbody></mujoco>'''
    started = time.monotonic()
    report = {
        "representation": "3d_flex_candidate",
        "xml": xml,
        "timestep_s": timestep,
        "integrator": integrator,
        "damping_s": damping,
        "changed_material_assumption": damping != p.relaxation_time_s,
        "reference_status": "analytical_not_measured",
        "passed": False,
    }
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        tip_vertices = np.flatnonzero(np.isclose(data.flexvert_xpos[:, 0], 0.06))
        if len(tip_vertices) != 9:
            raise ValueError("Unexpected Flex vertex ordering/geometry")
        tip_bodies = model.flex_vertbodyid[tip_vertices]
        for i in range(round(seconds / timestep)):
            data.xfrc_applied[:] = 0
            data.xfrc_applied[tip_bodies, 2] = (
                -0.01 / len(tip_bodies) * min(1.0, data.time / 0.01)
            )
            mujoco.mj_step(model, data)
            if any(
                data.warning[j].number
                for j in (
                    mujoco.mjtWarning.mjWARN_BADQPOS,
                    mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC,
                )
            ):
                raise FloatingPointError(f"Numerical warning at physical step {i}")
        mujoco.mj_forward(model, data)
        measured = float(-data.flexvert_xpos[tip_vertices, 2].mean())
        expected = 0.01 / p.tip_stiffness_n_m
        error = abs(measured - expected) / expected
        report.update(
            measured_deflection_m=measured,
            expected_deflection_m=expected,
            relative_error=error,
            passed=bool(error <= 0.05),
            dofs=model.nv,
        )
    except (ValueError, FloatingPointError) as error:
        report["failure"] = str(error)
    report["wall_seconds"] = time.monotonic() - started
    return report
