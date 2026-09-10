"""G8 汇总器反例：重复种子、缺 GUI 和缺实际探针不能凑成功率。"""

import importlib.util
import copy
import json
from pathlib import Path

import yaml


def test_g8_summary_keeps_fixed_denominators_and_separate_gui_gate(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_robot_milling.py"
    spec = importlib.util.spec_from_file_location("g8_summary_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load(
        (script.parents[1] / "scenes/plate_milling/scene_config.yaml").read_text(
            encoding="utf-8"
        )
    )

    def save(name, value):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    group = {"passed": True, "results": []}
    raw_cases = []
    for shape in ("slot", "pocket", "through_hole"):
        for seed in range(10):
            raw = {
                "shape": shape,
                "seed": seed,
                "passed": True,
                "frozen_source": True,
                "manifest": {
                    "source_hash": "same",
                    "scene_kind": "plate_milling",
                    "seed": seed,
                },
                "scene_config": config,
                "constraints": {"unchanged": True},
                "task_check": {"result": {"success": True}},
                "shape_check": {"passed": True},
                "probe_check": {"passed": True},
                "replay_check": {"passed": True},
                "controller_finished": True,
            }
            path = save(f"{shape}_{seed}", raw)
            raw_cases.append((path, raw))
            group["results"].append({"report_path": str(path)})
    viewer = dict.fromkeys(
        (
            "passed",
            "full",
            "source_unchanged_during_run",
            "successful_episode_1",
            "successful_episode_2",
            "persistent_finish",
            "initial_wait",
            "pause",
            "single_step",
            "previous_frame",
            "history_does_not_integrate",
            "replay",
            "rerun",
        ),
        True,
    )
    viewer.update(scene="plate_milling", source_hash="same")
    paths = {
        "robot_group": save("group", group),
        "viewer": save("viewer", viewer),
        "g7": save("g7", {"passed": True, "source_hash": "same"}),
        "preflight": save(
            "preflight", {"passed": True, "source_hash": "same", "frozen_source": True}
        ),
        "performance": save(
            "performance",
            {"passed": True, "source_hash": "same", "frozen_source": True, "events": 2},
        ),
    }
    variant_cases = []
    variant_entries = []
    for label, scale in (("coefficients_090", 0.9), ("coefficients_110", 1.1)):
        for seed in range(100, 105):
            raw = copy.deepcopy(raw_cases[-1][1])
            raw["seed"] = raw["manifest"]["seed"] = seed
            coefficients = raw["scene_config"]["process"]["coefficients"]
            for name in coefficients:
                if name not in {"source", "material_grade"}:
                    coefficients[name] *= scale
            raw["manifest"]["coefficients"] = copy.deepcopy(coefficients)
            raw["scene_config"]["materials"]["stock"]["parameter_source"] += (
                f"; coefficient sensitivity x{scale}; not measured material data"
            )
            path = save(f"{label}_{seed}", raw)
            variant_cases.append((path, raw))
            variant_entries.append(
                dict(
                    variant=label,
                    seed=seed,
                    passed=True,
                    frozen_source=True,
                    manifest={"source_hash": "same"},
                    report_path=str(path),
                )
            )
    variant_group = {
        "passed": True,
        "predefined_scales": [0.9, 1.1],
        "results": variant_entries,
    }
    paths["variants"] = save("variants", variant_group)
    assert module.summarize(paths)["passed"]  # 仅合成汇总器测试，不是机器人证据。
    # 摘要全部为 True，但两个实际变体探针失败；不能算 4/5。
    for path, raw in variant_cases[:2]:
        raw["probe_check"]["passed"] = False
        path.write_text(json.dumps(raw), encoding="utf-8")
    assert not module.summarize(paths)["checks"]["coefficients_090"]
    for path, raw in variant_cases[:2]:
        raw["probe_check"]["passed"] = True
        path.write_text(json.dumps(raw), encoding="utf-8")
    path, raw = variant_cases[0]
    saved_config = copy.deepcopy(raw["scene_config"])
    raw["scene_config"]["process"]["limits"]["max_force_n"] = 100.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert not module.summarize(paths)["checks"]["variant_recipe_integrity"]
    raw["scene_config"] = saved_config
    path.write_text(json.dumps(raw), encoding="utf-8")
    original_path = variant_entries[0]["report_path"]
    variant_entries[0]["report_path"] = str(tmp_path / "missing.json")
    save("variants", variant_group)
    assert not module.summarize(paths)["passed"]
    variant_entries[0]["report_path"] = original_path
    save("variants", variant_group)
    assert module.summarize(paths)["passed"]
    viewer["successful_episode_2"] = False
    save("viewer", viewer)
    assert not module.summarize(paths)["passed"]
    viewer["successful_episode_2"] = True
    save("viewer", viewer)
    path, raw = raw_cases[0]
    raw["seed"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert not module.summarize(paths)["checks"]["slot"]
    raw["seed"] = 0
    path.write_text(json.dumps(raw), encoding="utf-8")
    for path, raw in raw_cases[:2]:
        raw["probe_check"]["passed"] = False
        path.write_text(json.dumps(raw), encoding="utf-8")
    result = module.summarize(paths)
    assert not result["passed"] and result["counts"]["slot"]["successes"] == 8
