"""G7 可计量合成参考：闭式积分与真实准静态 F/T；不是钢材实测标定。"""

import math
from itertools import product
import mujoco
import numpy as np
from ibero.benches.milling import MillingFixture
from ibero.materials.parameters import StockParameters, strict_parameters
from ibero.processes.tools import EndMillGeometry
from ibero.processes.milling_forces import (
    MillingCoefficients,
    MillingLimits,
    mean_side_wrench,
)


def synthetic_load_report(scene):
    c = strict_parameters(MillingCoefficients, scene.config["process"]["coefficients"])
    limits = strict_parameters(MillingLimits, scene.config["process"]["limits"])
    tool = strict_parameters(EndMillGeometry, scene.config["tool"])
    analytical = []
    for samples, transition in product((64, 128, 256), (0.0, 1e-8, 5e-9)):
        for rpm, feed, depth in (
            (3000, 0.0002, 0.001),
            (6000, 0.0005, 0.001),
            (6000, 0.001, 0.002),
        ):
            fz = feed / (rpm / 60 * limits.teeth)
            n, r = limits.teeth, tool.radius_m
            expected_f = np.array(
                [
                    -n * depth * (c.radial_pa * fz / 4 + c.radial_edge_n_m / math.pi),
                    -n
                    * depth
                    * (c.tangential_pa * fz / 4 + c.tangential_edge_n_m / math.pi),
                    n * depth * (c.axial_pa * fz / math.pi + c.axial_edge_n_m / 2),
                ]
            )
            expected_tz = (
                -r
                * n
                * depth
                * (c.tangential_pa * fz / math.pi + c.tangential_edge_n_m / 2)
            )
            force, torque = mean_side_wrench(
                c,
                radius_m=r,
                rpm=rpm,
                teeth=n,
                velocity_tool=[feed, 0, 0],
                axial_lengths_m=np.full(samples, depth),
                edge_transition_chip_m=transition,
            )
            fe = float(np.linalg.norm(force - expected_f) / np.linalg.norm(expected_f))
            te = abs(float(torque[2] / expected_tz - 1))
            analytical.append(
                {
                    "angular_samples": samples,
                    "edge_transition_chip_m": transition,
                    "rpm": rpm,
                    "feed_m_s": feed,
                    "depth_m": depth,
                    "expected_force_n": expected_f.tolist(),
                    "actual_force_n": force.tolist(),
                    "expected_torque_z_nm": expected_tz,
                    "actual_torque_z_nm": float(torque[2]),
                    "force_relative_error": fe,
                    "torque_relative_error": te,
                    "passed": max(fe, te) <= 0.05,
                }
            )
    transfer = []
    for label, angle, force, torque in (
        ("nonzero_lever", 0, [1, -2, 3], [0.01, 0.02, -0.025]),
        ("rotated_frame", math.pi / 2, [1, 2, -3], [0.03, -0.02, 0.01]),
        ("pure_moment", math.pi / 3, [0, 0, 0], [0.01, -0.02, 0.03]),
    ):
        env = MillingFixture(
            StockParameters(size_m=(0.006, 0.006, 0.002)),
            0.002,
            tool,
            c,
            limits,
            ft_quaternion=(math.cos(angle / 2), 0, 0, math.sin(angle / 2)),
        )
        site = env.model.site("mill_ft_site").id
        force, torque = np.array(force, dtype=float), np.array(torque, dtype=float)
        mujoco.mj_forward(env.model, env.data)
        baseline_f = env.data.sensor("mill_force").data.copy()
        baseline_t = env.data.sensor("mill_torque").data.copy()
        reaction_error = 0.0
        for _ in range(3000):
            point = env.data.site("mill_tip").xpos.copy() + [0.001, -0.002, 0.003]
            env.loads.begin(env.data)
            env.loads.add_wrench(env.data, env.process.tool_body, force, torque, point)
            env.loads.add_wrench(
                env.data, env.process.stock_body, -force, -torque, point
            )
            env.loads.commit(env.data)
            net_f = env.data.xfrc_applied[:, :3].sum(axis=0)
            net_t = (
                env.data.xfrc_applied[:, 3:]
                + np.cross(env.data.xipos, env.data.xfrc_applied[:, :3])
            ).sum(axis=0)
            reaction_error = max(
                reaction_error,
                float(np.linalg.norm(net_f)),
                float(np.linalg.norm(net_t)),
            )
            mujoco.mj_step(env.model, env.data)
            mujoco.mj_forward(env.model, env.data)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        lever = point - env.data.site_xpos[site]
        expected_f, expected_t = (
            -rotation.T @ force,
            -rotation.T @ (torque + np.cross(lever, force)),
        )
        actual_f = env.data.sensor("mill_force").data.copy() - baseline_f
        actual_t = env.data.sensor("mill_torque").data.copy() - baseline_t
        fe, te = (
            float(np.linalg.norm(actual_f - expected_f)),
            float(np.linalg.norm(actual_t - expected_t)),
        )
        transfer.append(
            {
                "case": label,
                "baseline_force_n": baseline_f.tolist(),
                "baseline_torque_nm": baseline_t.tolist(),
                "expected_force_n": expected_f.tolist(),
                "measured_increment_force_n": actual_f.tolist(),
                "expected_torque_nm": expected_t.tolist(),
                "measured_increment_torque_nm": actual_t.tolist(),
                "force_error_n": fe,
                "torque_error_nm": te,
                "world_reaction_balance_error": reaction_error,
                "passed": fe <= 1e-6 and te <= 1e-6 and reaction_error <= 1e-6,
            }
        )
    return {
        "reference_status": "analytical_and_synthetic_not_measured",
        "analytical": analytical,
        "transfer": transfer,
        "passed": all(r["passed"] for r in analytical + transfer),
    }
