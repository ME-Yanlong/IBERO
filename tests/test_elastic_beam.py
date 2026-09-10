"""端载解析值、卸载回弹及参考数据来源检查。"""

import numpy as np
import pytest
import json
from pathlib import Path
from ibero.benches.beam import beam_static_test
from ibero.materials.references import fit_linear_stiffness, load_force_curve


@pytest.mark.parametrize("segments", [4, 8, 16])
def test_cantilever_stiffness_and_recovery(segments):
    result = beam_static_test(segments=segments)
    assert result["stiffness_relative_error"] <= 0.05
    assert result["residual_fraction"] <= 0.02


def test_calibration_is_validated_on_disjoint_samples():
    x = np.arange(1, 7) * 0.0001
    values = np.c_[x, 200 * x]
    result = fit_linear_stiffness(values, [0, 2, 4], [1, 3, 5])
    assert abs(result["stiffness_n_m"] - 200) < 1e-10
    assert result["validation_relative_l2"] < 1e-10
    with pytest.raises(ValueError):
        fit_linear_stiffness(values, [0, 1, 2], [2, 3, 4])


def test_reference_import_provenance_and_heldout():
    root = Path(__file__).resolve().parents[1] / "validation/references"
    values, metadata = load_force_curve(
        root / "beam_analytical.csv", root / "beam_analytical.json"
    )
    assert metadata["kind"] == "analytical"
    result = fit_linear_stiffness(values, [0, 2, 4], [1, 3, 5])
    assert result["stiffness_n_m"] == pytest.approx(222.2222222222222)
    assert result["validation_relative_l2"] < 1e-12


@pytest.mark.parametrize("bad", ["units", "source", "nonfinite", "columns", "few"])
def test_bad_reference_rejected(tmp_path, bad):
    metadata = {
        "source": "synthetic test",
        "kind": "synthetic",
        "units": {"displacement": "m", "force": "N"},
    }
    csv_text = "displacement_m,force_n\n0.1,1\n0.2,2\n0.3,3\n0.4,4\n"
    if bad == "units":
        metadata["units"]["force"] = "kN"
    elif bad == "source":
        metadata["source"] = ""
    elif bad == "nonfinite":
        csv_text = csv_text.replace("0.3,3", "0.3,nan")
    elif bad == "columns":
        csv_text = csv_text.replace("force_n", "force")
    else:
        csv_text = "displacement_m,force_n\n0.1,1\n"
    csv_path, meta_path = tmp_path / "curve.csv", tmp_path / "meta.json"
    csv_path.write_text(csv_text, encoding="utf-8")
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        load_force_curve(csv_path, meta_path)
