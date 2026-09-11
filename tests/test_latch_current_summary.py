"""最终 G4 汇总反例；合成 JSON 仅验证裁判，不作为物理成功证据。"""

import importlib.util
import json
from pathlib import Path


def test_hidden_artifact_errors_and_missing_raw_identity_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "latch_current_summary", root / "scripts/summarize_latch_current.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene, source = "latch_release", "a" * 64
    identities = (
        [(scene, s, "normal") for s in range(20)]
        + [
            (scene + "_" + v, s, "normal")
            for v in ("thinner", "shallower")
            for s in range(100, 105)
        ]
        + [
            (scene, 0, c)
            for c in ("no_press", "partial_press", "offset_press", "early_release")
        ]
    )
    rows = []
    for index, (name, seed, case) in enumerate(identities):
        folder = tmp_path / str(index)
        folder.mkdir()
        normal = case == "normal"
        raw = dict(
            scene=name,
            seed=seed,
            case=case,
            passed=True,
            success=normal,
            manifest=dict(source_hash=source),
            final=dict(success=normal, released=normal, invalid_reason=None),
            metric_checks=dict(force=True),
            trace=[dict(grasped=True)],
        )
        (folder / "report.json").write_text(json.dumps(raw), encoding="utf-8")
        (folder / "trace.npz").write_bytes(b"synthetic-presence-only")
        rows.append(
            dict(scene=name, seed=seed, case=case, passed=True, path=str(folder))
        )
    group = tmp_path / "group.json"
    group.write_text(
        json.dumps(
            dict(source_hash=source, checks=dict(frozen_source=True), results=rows)
        ),
        encoding="utf-8",
    )
    assert module.check_group(group, scene, source)["passed"]
    raw_path = tmp_path / "30/report.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["exception"] = "save failed after physical success"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    assert not module.check_group(group, scene, source)["checks"]["counterfactuals"]
    # 默认允许两个真实失败，不代表可以缺少两个原始身份记录。
    raw_path = tmp_path / "0/report.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["manifest"]["source_hash"] = "b" * 64
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    result = module.check_group(group, scene, source)
    assert not result["checks"]["all_raw_identities_present"]
