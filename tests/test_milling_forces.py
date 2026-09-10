"""独立闭式平均值与六维换算，不使用力函数自身作为参考。"""

from dataclasses import replace
import math
import numpy as np
import pytest
from ibero.processes.milling_forces import (
    MillingCoefficients,
    MillingLimits,
    mean_side_wrench,
    mean_face_wrench,
    capacity_reason,
)


@pytest.fixture
def coefficients():
    return MillingCoefficients(
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


def test_full_slot_average_against_closed_form(coefficients):
    c = coefficients
    radius, rpm, teeth, depth, feed = 0.004, 6000, 2, 0.001, 0.001
    fz = feed / (rpm / 60 * teeth)
    force, torque = mean_side_wrench(
        c,
        radius_m=radius,
        rpm=rpm,
        teeth=teeth,
        velocity_tool=[feed, 0, 0],
        axial_lengths_m=np.ones(256) * depth,
    )
    expected = [
        -teeth * depth * (c.radial_pa * fz / 4 + c.radial_edge_n_m / math.pi),
        -teeth * depth * (c.tangential_pa * fz / 4 + c.tangential_edge_n_m / math.pi),
        teeth * depth * (c.axial_pa * fz / math.pi + c.axial_edge_n_m / 2),
    ]
    expected_torque_z = (
        -radius
        * teeth
        * depth
        * (c.tangential_pa * fz / math.pi + c.tangential_edge_n_m / 2)
    )
    np.testing.assert_allclose(force, expected, rtol=0.001)
    assert torque[2] == pytest.approx(expected_torque_z, rel=0.001)
    assert force[0] < 0 and torque[2] < 0


def test_face_cut_integral_and_noncutting_directions(coefficients):
    c = coefficients
    f, t = mean_face_wrench(
        c,
        radius_m=0.004,
        rpm=6000,
        teeth=2,
        axial_velocity_m_s=-0.0002,
        engagement_fraction=1,
    )
    np.testing.assert_allclose(f, [0, 0, c.face_axial_pa * 1e-6 * 0.004 * 2])
    np.testing.assert_allclose(t, [0, 0, -c.face_tangent_pa * 1e-6 * 0.004**2])
    for rpm, velocity, fraction in [
        (0, -0.0002, 1),
        (6000, 0.0002, 1),
        (6000, -0.0002, 0),
    ]:
        f, t = mean_face_wrench(
            c,
            radius_m=0.004,
            rpm=rpm,
            teeth=2,
            axial_velocity_m_s=velocity,
            engagement_fraction=fraction,
        )
        assert not f.any() and not t.any()


def test_capacity_never_clips_requested_load():
    limits = MillingLimits(1000, 12000, 0.0001, 0.002, 5, 0.1, 50, 2, True)
    force, torque = np.array([6.0, 0, 0]), np.zeros(3)
    assert (
        capacity_reason(
            limits,
            rpm=6000,
            velocity_tool=[0.001, 0, 0],
            axial_depth_m=0.001,
            force_tool=force,
            torque_tool=torque,
        )
        == "cutting_force_overload"
    )
    assert force[0] == 6
    assert (
        capacity_reason(
            replace(limits, max_force_n=10),
            rpm=6000,
            velocity_tool=[0.001, 0, 0],
            axial_depth_m=0.001,
            force_tool=force,
            torque_tool=torque,
        )
        is None
    )
