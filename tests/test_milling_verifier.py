"""验收器失败路径：证据附件失败不能掩盖物理失败或误计通过。"""

import importlib.util
import json
from pathlib import Path


def test_verifier_keeps_original_failure_when_both_artifacts_fail(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "milling_verifier_under_test", root / "scripts/verify_milling_process.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    def physics_failure(*args, **kwargs):
        raise RuntimeError("injected physics failure")

    def artifact_failure(*args, **kwargs):
        raise OSError("injected storage failure")

    monkeypatch.setattr(verifier.MillingFixture, "step", physics_failure)
    monkeypatch.setattr(verifier, "section_image", artifact_failure)
    monkeypatch.setattr(verifier.StockTrace, "save", artifact_failure)
    result = verifier.run_shape(
        ("slot", str(tmp_path), str(root / "scenes/milling_bench"), 0)
    )
    raw = json.loads((tmp_path / "slot/report.json").read_text(encoding="utf-8"))
    assert not result["passed"] and not raw["passed"]
    assert raw["exception"] == "RuntimeError: injected physics failure"
    assert len(raw["artifact_errors"]) == 2
    assert raw["scene_config"]["kind"] == "milling_bench"
    assert raw["constraints"]["task"]["max_seconds"] > 0
    assert raw["rows"] == [] and raw["frozen_source"]
