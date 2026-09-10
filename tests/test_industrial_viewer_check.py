"""验收进度只能读计数，不能用进度状态宣称 GUI 或加工已通过。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_viewer_progress_is_read_only_and_not_acceptance():
    path = Path(__file__).resolve().parents[1] / "scripts/check_industrial_viewer.py"
    spec = importlib.util.spec_from_file_location("viewer_checker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states = [object(), object()]
    app = SimpleNamespace(
        env=SimpleNamespace(data=SimpleNamespace(time=2.0)),
        index=1,
        trace=SimpleNamespace(states=states),
        completed_episodes=0,
        running=True,
        finished=False,
        closed=False,
        future=SimpleNamespace(done=lambda: False),
    )
    state = dict(phase="full_first", started=5.0, phase_durations_s={"initial": 0.2})
    row = module.progress_snapshot(app, state, 10.0)
    assert row["wall_seconds"] == 5.0 and row["simulation_seconds"] == 2.0
    assert row["recorded_frames"] == 2 and row["worker_pending"]
    assert "passed" not in row and not row["finished"]
    row["phase_durations_s"]["initial"] = 999
    assert state["phase_durations_s"]["initial"] == 0.2
    assert len(states) == 2 and app.env.data.time == 2.0 and app.running
    app.closed = True
    app.future = SimpleNamespace(done=lambda: True)
    closed = module.progress_snapshot(app, state, 12.0)
    assert closed["closed"] and not closed["worker_pending"] and not closed["running"]
