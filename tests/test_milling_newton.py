"""有界半光滑求解：保留真实失败子步，并验证只求载荷、不改物理/材料状态。"""

import json
from pathlib import Path
import mujoco
import numpy as np
import pytest
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.processes.prepared_milling_wrench import PreparedMillingWrench
from ibero.processes.implicit_wrench import ImplicitWrenchCoupling
from ibero.review import Trace


def test_one_sided_derivative_not_hidden_by_negligible_central_descent():
    # 独立数学折点反例；不是材料标定曲线。旧逻辑第一点微弱下降就跳过了合法单侧导数。
    solver = object.__new__(ImplicitWrenchCoupling)
    solver.max_iterations = 40
    solver.last_diagnostics = {}
    initial = np.array([0.0, 0.0, 0.0, 600.0])

    def residual(u):
        r = u - [0, 0, 0, 600]
        r[0] = (u[0] if u[0] >= 0 else 1e9 * u[0]) - 1e-6
        return r

    root = solver._iterate(residual, initial, np.array([1, 1, 1, 0.001]), initial)
    np.testing.assert_allclose(root, [1e-6, 0, 0, 600], atol=1e-10)
    assert solver.last_diagnostics["iterations"] < 5


def test_real_seed_one_fold_snapshot_converges_without_material_or_state_changes():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/milling_newton_seed1.json").read_text(
            encoding="utf-8"
        )
    )
    e = PlateMillingEnv(shape="slot")
    assert (e.model.nq, e.model.nv) == (fixture["nq"], fixture["nv"])
    assert int(Trace.spec) == fixture["state_spec"]
    # 这是具名数值回归台架的初态，不是解禁 StockTrace 跨版本回放/续跑。
    event = e.stock.prepare_removal(
        np.array(fixture["removed_cell_ids"]), "numeric-fixture"
    )
    e.binding.commit(event, e.data)
    mujoco.mj_setState(
        e.model, e.data, np.array(fixture["integration_state"]), Trace.spec
    )
    e.process.set_collision_mode(2)
    mujoco.mj_forward(e.model, e.data)
    pose, velocity, _, _ = e.process.kinematics(e.data)
    lengths, centers, fraction, pending = e.process._engagement(pose, velocity)
    assert pending
    prepared = PreparedMillingWrench(
        e.coefficients,
        radius_m=e.tool.radius_m,
        teeth=e.limits.teeth,
        lengths_m=lengths,
        centroids_m=centers,
        face_fraction=fraction,
        edge_transition_chip_m=e.process.edge_transition_chip_m,
        center_cutting=True,
    )

    def law(u):
        value = prepared(pose.rotation.T @ u[:3], u[3] * 30 / np.pi)
        return np.r_[pose.rotation @ value[:3], pose.rotation @ value[3:]]

    before_q, before_v, before_hash = (
        e.data.qpos.copy(),
        e.data.qvel.copy(),
        e.stock.state_hash(),
    )
    wrench, _, error = e.process.coupling.solve(
        e.data, e.data.qfrc_applied, e.data.xfrc_applied, pose.position, law
    )
    assert error < 1e-9 and np.linalg.norm(wrench[:3]) < 15
    assert e.process.coupling.last_diagnostics["iterations"] < 10
    assert e.stock.state_hash() == before_hash
    np.testing.assert_array_equal(e.data.qpos, before_q)
    np.testing.assert_array_equal(e.data.qvel, before_v)


@pytest.mark.parametrize("value", [0, 161, True, 2.5])
def test_iteration_budget_is_bounded_before_allocating_model(value):
    with pytest.raises(ValueError, match="iteration budget"):
        ImplicitWrenchCoupling(
            None, tool_body=0, tip_site=0, spindle_dof=0, max_iterations=value
        )
