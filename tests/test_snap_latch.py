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
