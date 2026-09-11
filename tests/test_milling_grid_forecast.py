"""只读网格预估暴露离散误差，不能生成材料事件或替代真实物理。"""

import copy
import importlib.util
from pathlib import Path

import pytest
from ibero.core.scene_loader import SceneLoader


@pytest.mark.parametrize("kind", ["milling_bench", "plate_milling"])
def test_nominal_grid_forecast_reports_depth_quantization_without_events(kind):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/forecast_milling_geometry.py"
    spec = importlib.util.spec_from_file_location("grid_forecast_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = SceneLoader().validate(root / "scenes" / kind)
    config = copy.deepcopy(scene.config)
    row = module.forecast(scene, [0.001, 0.00075, 0.002 / 3, 0.0005])
    assert "passed" not in row and row["physical_steps"] == 0
    assert scene.config == config
    selected = {(r["cell_size_m"], r["shape"]): r for r in row["results"]}
    assert all(r["material_events"] == 0 for r in row["results"])
    # 机器人 0.5 mm 格距还会触发既定刀轴投影上限；几何误差合格不能掩盖这一点。
    fine = selected[0.0005, "through_hole"]
    assert fine["candidate_recipe_valid"] is (kind == "milling_bench")
    if kind == "plate_milling":
        assert "quarter voxel" in fine["candidate_recipe_error"]
    assert selected[0.00075, "slot"]["relative_volume_error"] == pytest.approx(
        0.08683053048
    )
    assert not selected[0.00075, "pocket"]["within_volume_criterion"]
    assert all(
        selected[0.002 / 3, shape]["within_volume_criterion"]
        for shape in ("slot", "pocket", "through_hole")
    )
    # 更细不保证体积误差单调下降：中心采样会出现网格相位/几何对齐效应。
    assert (
        selected[0.0005, "through_hole"]["relative_volume_error"]
        > selected[0.002 / 3, "through_hole"]["relative_volume_error"]
    )
