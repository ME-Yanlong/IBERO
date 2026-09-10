"""阶段汇总独立重读原始证据；这些合成文件不是物理通过记录。"""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


@pytest.mark.parametrize("tamper", ["raw_probe", "unhalved_step", "changed_feed"])
def test_g7_summary_rejects_hidden_evidence_changes(tmp_path, tamper):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "g7_summary_under_test", root / "scripts/summarize_milling_stage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = yaml.safe_load(
        (root / "scenes/milling_bench/scene_config.yaml").read_text(encoding="utf-8")
    )

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    base = dict(
        passed=True,
        frozen_source=True,
        controller_finished=True,
        manifest={"source_hash": "same"},
        probe_check={"passed": True},
        replay_check={"passed": True},
        shape_check={"passed": True, "removed_volume_m3": 1e-7},
        rows=[
            dict(
                peak_cutting_force_step_n=4.0,
                peak_spindle_torque_step_nm=0.1,
                peak_spindle_power_step_w=50.0,
            )
        ],
        scene_config=cfg,
        constraints={"same": True},
        seed=0,
        sim_seconds=2.0,
        wall_seconds=50.0,
    )
    cases = {}
    for shape in ("slot", "pocket", "through_hole"):
        row = dict(copy.deepcopy(base), shape=shape)
        cases[shape] = row
        save(tmp_path / shape / "report.json", row)
    args = SimpleNamespace(
        shapes=save(tmp_path / "shapes.json", {"results": list(cases.values())}),
        loads=save(
            tmp_path / "loads.json",
            dict(passed=True, frozen_source=True, source_hash="same"),
        ),
        half_step_slot=tmp_path / "half_step.json",
        half_edge_slot=tmp_path / "half_edge.json",
    )
    step = copy.deepcopy(cases["slot"])
    step["scene_config"]["physics"]["timestep_s"] /= 2
    edge = copy.deepcopy(cases["slot"])
    edge["scene_config"]["numerics"]["edge_transition_chip_m"] /= 2
    save(args.half_step_slot, step)
    save(args.half_edge_slot, edge)
    assert module.summarize(args)["passed"]
    if tamper == "raw_probe":
        cases["slot"]["probe_check"]["passed"] = False
        save(tmp_path / "slot/report.json", cases["slot"])
    elif tamper == "unhalved_step":
        step["scene_config"]["physics"]["timestep_s"] *= 2
        save(args.half_step_slot, step)
    else:
        edge["scene_config"]["control"]["feed_m_s"] /= 2
        save(args.half_edge_slot, edge)
    assert not module.summarize(args)["passed"]
