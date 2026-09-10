"""三网格汇总器合成反例；这些 JSON 不是物理验收证据。"""

import copy
import importlib.util
import json
from pathlib import Path


def test_grid_summary_rejects_modified_control_missing_raw_and_hidden_probe(
    tmp_path, monkeypatch
):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "grid_summary_test", scripts / "summarize_milling_grid.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths, cases = {}, {}

    def save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    for label, cell in (("base", 0.001), ("medium", 0.00075), ("fine", 0.0005)):
        paths[label] = tmp_path / label / "report.json"
        rows = []
        for shape in ("slot", "pocket", "through_hole"):
            row = dict(
                shape=shape,
                seed=0,
                passed=True,
                frozen_source=True,
                manifest=dict(source_hash="synthetic", seed=0),
                scene_config=dict(
                    kind="milling_bench",
                    numerics=dict(cell_size_m=cell),
                    control=dict(feed=1),
                ),
                constraints=dict(unchanged=True),
                controller_finished=True,
                shape_check=dict(passed=True, removed_volume_m3=1e-6),
                probe_check=dict(passed=True),
                replay_check=dict(passed=True),
                rows=[
                    dict(
                        peak_cutting_force_step_n=1,
                        peak_spindle_torque_step_nm=0.1,
                        peak_spindle_power_step_w=3,
                    )
                ],
            )
            path = paths[label].parent / shape / "report.json"
            cases[label, shape] = path, row
            save(path, row)
            rows.append(row)
        save(paths[label], dict(passed=True, results=rows))
    assert module.summarize(paths)["passed"]
    path, row = cases["fine", "slot"]
    original = copy.deepcopy(row)
    row["scene_config"]["control"]["feed"] = 0.5
    save(path, row)
    assert not module.summarize(paths)["passed"]
    save(path, original)
    row = copy.deepcopy(original)
    row["probe_check"]["passed"] = False
    save(path, row)
    assert not module.summarize(paths)["passed"]
    save(path, original)
    # 只移动合成临时文件，不能拿上层 group 的 passed=True 补缺失的原始证据。
    path.rename(path.with_name("missing_raw_fixture.json"))
    result = module.summarize(paths)
    assert not result["passed"] and len(result["results"]) == 9
