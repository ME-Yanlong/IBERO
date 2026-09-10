"""卡扣反事实必须由实际积分、接触和被动恢复产生。"""

import pytest

from ibero.benches.latch import LatchFixture, run_fixture


@pytest.mark.parametrize(
    "case", ["no_press", "partial_press", "offset_press", "early_release"]
)
def test_cannot_release_without_sustained_correct_press(case):
    result = run_fixture(case)
    assert not result["final"]["released"]
    assert result["final"]["invalid_reason"] is None
    assert result["final"]["peak_penetration_m"] < 0.0003


def test_press_pull_releases_through_real_geometry():
    result = run_fixture()
    assert result["final"]["released"]
    assert result["final"]["invalid_reason"] is None
    assert result["final"]["clearance_m"] > 0
    assert any(e["state"] == "clearance_open" for e in result["events"])


def test_overload_is_not_success():
    result = run_fixture("overload")
    assert result["final"]["invalid_reason"]
    assert not result["final"]["released"]


def test_reset_clears_state_and_no_hidden_lock_switch():
    fixture = LatchFixture()
    assert fixture.model.neq == 0
    fixture.step(-0.02, 0, 0.5)
    fixture.reset()
    assert fixture.observer.invalid_reason is None
    assert fixture.data.time == 0
    assert fixture.observer.observe(fixture.data)["latch_state"] == "locked"
    assert fixture.peak_press_force_n == 0
    assert not fixture.data.xfrc_applied.any()


def test_initial_geometry_clear_and_nonintegral_time_rejected():
    fixture = LatchFixture()
    assert fixture.observer.observe(fixture.data)["penetration_m"] == 0
    with pytest.raises(ValueError):
        fixture.step(0, 0, fixture.model.opt.timestep * 1.5)
    assert fixture.data.time == 0


def test_invalid_latches_until_reset():
    fixture = LatchFixture(max_force_n=0.00001)
    result = fixture.step(-0.02, 0, 0.1)
    assert result["invalid_reason"]
    with pytest.raises(RuntimeError):
        fixture.step(0, 0)
    fixture.reset()
    assert fixture.observer.invalid_reason is None


def test_recipe_controls_fixture_parameters():
    from pathlib import Path
    from ibero.core.scene_loader import SceneLoader

    scene = SceneLoader().validate(
        Path(__file__).resolve().parents[1] / "scenes/latch_bench"
    )
    fixture = LatchFixture.from_scene(scene)
    assert fixture.model.opt.timestep == scene.config["physics"]["timestep_s"]
    assert fixture.observer.max_force_n == scene.constraints["safety"]["max_force_n"]


def test_latch_observation_is_rigid_frame_invariant():
    import mujoco
    import numpy as np
    from ibero.core.scene_compiler import fixture_spec
    from ibero.mechanisms.snap_latch import add_latch, LatchObserver
    from ibero.materials.parameters import BeamParameters, LatchParameters

    spec = fixture_spec(
        {
            "id": "rotated",
            "physics": {"timestep_s": 0.0000025, "gravity_m_s2": [0, 0, 0]},
        }
    )
    names = add_latch(
        spec,
        BeamParameters(),
        LatchParameters(),
        origin=[0.48, 0.2, 0.85],
        quaternion=[2**-0.5, 0, 0, 2**-0.5],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    row = LatchObserver(model, names).observe(data)
    assert row["clearance_m"] == pytest.approx(-0.003)
    assert row["deflection_m"] == pytest.approx(0)
    assert row["latch_state"] == "locked"
    np.testing.assert_allclose(data.body("latch_plug").xpos, [0.48, 0.2, 0.85])
