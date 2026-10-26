"""布线验证入口的真实环境、反例和证据完整性检查。"""

from pathlib import Path

from ibero import verify_routing


def test_b1_b2_scenes_validate_with_distinct_clip_counts():
    single = verify_routing.validate_scene("cable_clip_insert")
    straight = verify_routing.validate_scene("cable_route_straight")
    assert single["passed"] and single["clip_count"] == 1
    assert straight["passed"] and straight["clip_count"] == 3
    assert single["scene_hash"] != straight["scene_hash"]


def test_no_entry_and_wide_jaw_never_count_as_retained():
    checks = verify_routing.counterfactual_checks()
    assert {row["name"] for row in checks} == {
        "counterfactual:no_entry", "counterfactual:wide_jaw",
    }
    assert all(row["passed"] and row["retained_count"] == 0 for row in checks)


def test_real_short_episode_records_trace_and_does_not_claim_success(tmp_path: Path):
    trace_path = tmp_path / "short.npz"
    result = verify_routing.run_episode(
        "cable_clip_insert", seed=0, max_steps=1, trace_path=trace_path
    )
    assert result["scene_id"] == "cable_clip_insert"
    assert result["seed"] == 0 and result["steps"] == 1
    assert not result["passed"]
    assert not result["terminated"] and not result["truncated"]
    assert trace_path.is_file() and len(result["trace_sha256"]) == 64
    assert result["trace_roundtrip_passed"]
    assert result["deterministic_reset_passed"]


def test_tasks_gate_preserves_failed_seeds_and_partial_runs_are_diagnostic(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        verify_routing, "validate_scene",
        lambda name: {"name": f"scene:{name}", "passed": True},
    )

    def fake_episode(name, seed, **_kwargs):
        return {
            "name": f"episode:{name}:{seed}",
            "passed": seed != 9,
            "seed": seed,
        }

    monkeypatch.setattr(verify_routing, "run_episode", fake_episode)
    report = verify_routing.build_report(
        "tasks", output=tmp_path / "full.json"
    )
    assert report["gate_complete"] and report["passed"]
    assert report["episode_successes"] == 36
    assert report["episode_denominator"] == 40
    assert report["required_successes_per_scene"] == {
        "cable_clip_insert": 9,
        "cable_route_straight": 9,
        "cable_route_l": 8,
        "cable_route_rectangle": 8,
    }
    assert sum(not row["passed"] for row in report["checks"]) == 4

    partial = verify_routing.build_report(
        "tasks", output=tmp_path / "partial.json", seeds=(0,)
    )
    assert not partial["gate_complete"]
    assert partial["passed"] is None
    assert partial["episode_denominator"] == 4

    selected = verify_routing.build_report(
        "tasks", output=tmp_path / "selected.json",
        scenes=("cable_clip_insert", "cable_route_straight"),
    )
    assert selected["episode_denominator"] == 20
    assert selected["episode_successes"] == 18
    assert selected["gate_complete"] is False
    assert selected["passed"] is None
