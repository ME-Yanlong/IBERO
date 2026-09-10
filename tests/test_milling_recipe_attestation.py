"""旧证据补核只验证身份，不把旧失败改成成功、不产生新物理轨迹。"""

import importlib.util
import json
from pathlib import Path

import pytest

from ibero.benches.milling import MillingFixture
from ibero.core.stock_trace import StockTrace


def test_recipe_attestation_preserves_failure_and_rejects_wrong_identity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "recipe_attestation_under_test", root / "scripts/attest_milling_recipes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = root / "scenes/milling_bench"
    env = MillingFixture.from_scene(scene)
    env.reset(seed=7)
    trace = StockTrace(env)
    trace.append(env.last_info)
    trace.save(tmp_path / "trace.npz")
    raw = dict(
        passed=False,
        seed=7,
        shape="slot",
        frozen_source=True,
        manifest=env.manifest(),
        invalid_reason="synthetic_failed_case",
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    original = path.read_bytes()
    result = module.attest(path, scene)
    assert result["passed"] and result["physical_steps_executed"] == 0
    assert result["trace_frames_validated"] == 1
    assert path.read_bytes() == original
    assert not json.loads(original)["passed"]
    raw["manifest"]["scene_hash"] = "wrong_recipe"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen recipe identity"):
        module.attest(path, scene)
