"""核对演讲稿包的结构、原文件保护、视频内嵌以及本机 Office 审阅结果。"""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/presentation_v4_additive"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main", "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def main():
    manifest = json.loads((OUT / "build_manifest.json").read_text(encoding="utf-8"))
    source = ROOT / "IBERO_工业级双臂具身智能操作基准平台演讲PPT_v3.pptx"
    target = ROOT / "IBERO_工业级双臂具身智能操作基准平台演讲PPT_v4.pptx"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_v3_sha256"]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == manifest["output_sha256"]
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        assert len(presentation.find("p:sldIdLst", NS)) == 29
        videos = [n for n in archive.namelist() if n.startswith("ppt/media/") and n.endswith(".mp4")]
        assert len(videos) == 7
        assert {hashlib.sha256(archive.read(n)).hexdigest() for n in videos} == {v["sha256"] for v in manifest["additions"] if v["kind"] == "video"}
        all_text = []
        external = []
        for name in archive.namelist():
            if name.endswith(".rels"):
                for rel in ET.fromstring(archive.read(name)):
                    if rel.attrib.get("TargetMode") == "External":
                        external.append({"part": name, **rel.attrib})
            if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                all_text.extend(t.text or "" for t in ET.fromstring(archive.read(name)).findall(".//a:t", NS))
        assert not external, external
        # 原占位页也属于用户要求保留的内容，不能再次删改。
        assert "文字或图发给我统一生成PPT" in "".join(all_text)
        exceptions = {"ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "[Content_Types].xml"}
        with ZipFile(source) as original:
            preserved = [n for n in original.namelist() if n not in exceptions]
            assert all(original.read(n) == archive.read(n) for n in preserved)
            old_ids = list(ET.fromstring(original.read("ppt/presentation.xml")).find("p:sldIdLst", NS))
            new_ids = list(presentation.find("p:sldIdLst", NS))
            assert [dict(new_ids[p-1].attrib) for p in manifest["original_slide_positions"]] == [dict(el.attrib) for el in old_ids]
    issues = json.loads((OUT / "render/layout_issues.json").read_text(encoding="utf-8-sig"))
    # 只要求新增页没有越界；不借检查之名修改用户原页的历史排版。
    assert all(row["Slide"] in manifest["original_slide_positions"] for row in issues), issues
    playback = json.loads((OUT / "native_playback_check.json").read_text(encoding="utf-8-sig"))
    assert len(playback) == 7 and all(row["Passed"] and row["PositionMs"] > 250 and row["ClickAction"] == 12 for row in playback)
    assert {row["Slide"] for row in playback} == set(manifest["video_positions"])
    assert len(list((OUT / "render").glob("slide_*.png"))) == 29
    from PIL import Image, ImageChops
    matched = 0
    for old_index, new_index in enumerate(manifest["original_slide_positions"], 1):
        before = Image.open(ROOT / f"artifacts/presentation_v4/v3_render/slide_{old_index:02d}.png").convert("RGB")
        after = Image.open(OUT / f"render/slide_{new_index:02d}.png").convert("RGB")
        assert ImageChops.difference(before, after).getbbox() is None, (old_index, new_index)
        matched += 1
    result = {"passed": True, "slides": 29, "embedded_videos": 7, "original_v3_unchanged": True,
              "original_parts_byte_identical": len(preserved), "original_slides_pixel_identical": matched,
              "media_bytes_verified": True, "external_relationships": 0, "original_placeholders_preserved": True,
              "powerpoint_rendered_slides": 29, "new_slide_text_bound_issues": 0,
              "preserved_original_layout_warnings": len(issues), "native_video_playback_passed": 7,
              "presentation_sha256": manifest["output_sha256"]}
    (OUT / "final_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
