"""目录发现不执行任务代码；网页打包保留来源并拒绝错误摘要。"""

import hashlib
import json

import pytest

from ibero.task_catalog import discover_tasks
from scripts.build_demo_gallery import build_gallery
from scripts.make_demo_gallery_manifest import make_manifest


def test_catalog_reads_static_instruction_without_importing_scene_code(tmp_path):
    scene = tmp_path / "cable_route_straight"
    scene.mkdir()
    (scene / "scene_config.yaml").write_text(
        "id: cable_route_straight\ntask_family: cable_routing\nschema_version: ibero.routing/v0.2\n",
        encoding="utf-8",
    )
    (scene / "task_spec.py").write_text(
        "from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('bad')\n"
        "class Task:\n    instruction = 'Route the cable through three clips.'\n",
        encoding="utf-8",
    )
    cards = discover_tasks(tmp_path)
    assert len(cards) == 1
    assert cards[0].family == "cable_routing"
    assert cards[0].instruction == "Route the cable through three clips."
    assert not (scene / "task_spec.ran").exists()


def test_catalog_reads_task_spec_constructor_instruction_without_import(tmp_path):
    scene = tmp_path / "cable_clip_insert"
    scene.mkdir()
    (scene / "scene_config.yaml").write_text(
        "id: cable_clip_insert\nsuite: harness\nschema_version: ibero.routing/v0.2\n",
        encoding="utf-8",
    )
    (scene / "task_spec.py").write_text(
        "from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('bad')\n"
        "TASK_SPEC = CableRoutingTask(goal, task_id='cable_clip_insert', "
        "instruction='用双手抓住裸线，压入立式板的线夹。')\n",
        encoding="utf-8",
    )
    cards = discover_tasks(tmp_path)
    assert cards[0].instruction == "用双手抓住裸线，压入立式板的线夹。"
    assert not (scene / "task_spec.ran").exists()


def test_gallery_packages_relative_videos_and_escapes_text(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"small fixture representing an existing encoded video")
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            dict(
                title="Real episode",
                seed=0,
                source_commit="1234567890",
                source_case_passed=True,
                full_decode_verified=True,
                video_sha256=video_hash,
            )
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "gallery.json"
    manifest.write_text(
        json.dumps(
            dict(
                title="IBERO <Tasks>",
                entries=[dict(id="route_01", video="source.mp4", source_manifest="source.json", description="<b>press and retain</b>")],
            )
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "portable"
    index = build_gallery(manifest, destination)
    html = index.read_text(encoding="utf-8")
    rows = json.loads((destination / "index.json").read_text(encoding="utf-8"))["entries"]
    assert (destination / rows[0]["video"]).read_bytes() == video.read_bytes()
    assert (destination / rows[0]["source_manifest"]).read_bytes() == source.read_bytes()
    assert rows[0]["success"] is True and rows[0]["full_decode_verified"] is True
    assert "IBERO &lt;Tasks&gt;" in html
    assert "&lt;b&gt;press and retain&lt;/b&gt;" in html
    assert '<b>press and retain</b>' not in html
    assert "videos/route_01.mp4" in html
    with pytest.raises(FileExistsError):
        build_gallery(manifest, destination)


def test_gallery_rejects_wrong_video_hash_before_creating_output(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"video bytes")
    (tmp_path / "clip.json").write_text(
        json.dumps({"video_sha256": "0" * 64, "full_decode_verified": True}),
        encoding="utf-8",
    )
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps({"entries": [{"id": "clip", "video": "clip.mp4", "source_manifest": "clip.json"}]}),
        encoding="utf-8",
    )
    output = tmp_path / "share"
    with pytest.raises(ValueError, match="hash differs"):
        build_gallery(manifest, output)
    assert not output.exists()


def test_manifest_generator_uses_new_video_sidecars_and_stays_relative(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "cable_clip_insert.mp4").write_bytes(b"encoded bytes")
    (videos / "cable_clip_insert.json").write_text(
        json.dumps({"title": "单夹任务", "video_sha256": hashlib.sha256(b"encoded bytes").hexdigest()}),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.json"
    make_manifest(videos, output, title="布线演示")
    data = json.loads(output.read_text(encoding="utf-8"))
    entry = data["entries"][0]
    assert entry["title"] == "单夹任务"
    assert entry["video"] == "videos/cable_clip_insert.mp4"
    assert entry["source_manifest"] == "videos/cable_clip_insert.json"
    assert build_gallery(output, tmp_path / "share").is_file()
