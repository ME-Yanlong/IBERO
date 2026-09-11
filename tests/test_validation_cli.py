"""统一编排的分组、失败退出码、原证据保护和参考缺项语义。"""

import json
from types import SimpleNamespace
import pytest
from ibero import validation_cli as cli


def options(tmp_path, suite="quick", **changes):
    values = dict(
        suite=suite,
        output=tmp_path / "report.json",
        workers=3,
        group="all",
        scene="plate_milling",
        full=False,
        reference_csv=None,
        reference_metadata=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_reference_absent_is_not_passed(tmp_path):
    assert cli.run_suite(options(tmp_path, "reference")) == 3
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is None and not report["calibrated"]
    assert report["status"] == "not_available"
    with pytest.raises(ValueError, match="overwrite"):
        cli.run_suite(options(tmp_path, "reference"))


def test_headless_suites_cannot_implicitly_launch_gui(tmp_path):
    for suite in cli.SUITES:
        if suite in {"latch", "reference"}:
            continue
        command, _ = cli.suite_command(options(tmp_path, suite), tmp_path / "data")
        is_gui = any("check_industrial_viewer.py" in item for item in command)
        assert is_gui == (suite == "gui")
        assert "--viewer" not in command


@pytest.mark.parametrize("code", [0, 1])
def test_quick_exit_code_and_explicit_threads_preserved(monkeypatch, tmp_path, code):
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    def run(command, **kwargs):
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        assert "OPENBLAS_NUM_THREADS" not in kwargs["env"]
        assert kwargs["cwd"] == cli.ROOT
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.run_suite(options(tmp_path)) == code
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] == (code == 0)
    assert report["scope"] == "selected_suite_only_not_automatic_G9"


def test_missing_child_evidence_fails_even_with_zero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    assert cli.run_suite(options(tmp_path, "stock-geometry")) == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert not report["passed"] and "FileNotFoundError" in report["exception"]


def test_imported_synthetic_curve_is_not_real_calibration(tmp_path):
    csv_path, meta_path = tmp_path / "curve.csv", tmp_path / "meta.json"
    csv_path.write_text(
        "displacement_m,force_n\n0,0\n0.001,1\n0.002,2\n0.003,3\n", encoding="utf-8"
    )
    meta_path.write_text(
        json.dumps(
            dict(
                source="synthetic test",
                kind="synthetic",
                units=dict(displacement="m", force="N"),
            )
        ),
        encoding="utf-8",
    )
    result = cli.reference_report(csv_path, meta_path)
    assert result["passed"] is None and not result["calibrated"]
    assert result["sample_count"] == 4 and result["status"] == "imported_not_calibrated"
