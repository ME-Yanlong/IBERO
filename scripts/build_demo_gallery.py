"""从显式视频清单制作可离线分享的 IBERO 演示目录。"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


SAFE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*\Z")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_file(manifest: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty path")
    path = Path(value)
    path = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _prepare(manifest: Path) -> tuple[str, list[dict]]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise ValueError("Gallery manifest must contain an entries list")
    title = raw.get("title", "IBERO · 任务演示")
    if not isinstance(title, str) or not title:
        raise ValueError("Gallery title must be nonempty text")
    if not raw["entries"]:
        raise ValueError("Gallery requires at least one video entry")
    seen = set()
    entries = []
    for item in raw["entries"]:
        if not isinstance(item, dict):
            raise ValueError("Every gallery entry must be a mapping")
        scene_id = item.get("id")
        if not isinstance(scene_id, str) or not SAFE_ID.fullmatch(scene_id):
            raise ValueError(f"Invalid entry id: {scene_id!r}")
        if scene_id in seen:
            raise ValueError(f"Duplicate entry id: {scene_id}")
        seen.add(scene_id)
        video = _source_file(manifest, item.get("video"), f"{scene_id}.video")
        if video.suffix.lower() != ".mp4":
            raise ValueError(f"{scene_id}.video must be an MP4 file")
        source = (
            _source_file(manifest, item["source_manifest"], f"{scene_id}.source_manifest")
            if item.get("source_manifest") is not None
            else None
        )
        report = (
            _source_file(manifest, item["report"], f"{scene_id}.report")
            if item.get("report") is not None
            else None
        )
        for label, path in (("source_manifest", source), ("report", report)):
            if path is not None and path.suffix.lower() != ".json":
                raise ValueError(f"{scene_id}.{label} must be JSON")
        source_data = json.loads(source.read_text(encoding="utf-8")) if source else {}
        if not isinstance(source_data, dict):
            raise ValueError(f"{scene_id}.source_manifest must be a JSON object")
        video_hash = sha256(video)
        declared_hash = source_data.get("video_sha256")
        if declared_hash is not None and declared_hash != video_hash:
            raise ValueError(f"{scene_id} video hash differs from its source manifest")
        description = item.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"{scene_id}.description must be text")
        entry_title = item.get("title", source_data.get("title", scene_id))
        if not isinstance(entry_title, str) or not entry_title:
            raise ValueError(f"{scene_id}.title must be nonempty text")
        entries.append(
            dict(
                id=scene_id,
                title=entry_title,
                description=description,
                video_source=video,
                source_manifest_source=source,
                report_source=report,
                video_sha256=video_hash,
                source_manifest_sha256=sha256(source) if source else None,
                report_sha256=sha256(report) if report else None,
                # 完整解码标志来自导出报告；只有同时匹配视频摘要才称已核验。
                full_decode_verified=bool(source_data.get("full_decode_verified") and declared_hash),
                success=source_data.get("source_case_passed") if source else None,
                seed=source_data.get("seed") if source else None,
                source_commit=source_data.get("source_commit") if source else None,
                simulation_end_s=source_data.get("simulation_end_s") if source else None,
            )
        )
    return title, entries


def _page(title: str, rows: list[dict]) -> str:
    cards = []
    for row in rows:
        label = (
            "来源视频已核验" if row["full_decode_verified"] else "来源视频尚未完成解码核验"
        )
        status = (
            "本回合成功" if row["success"] is True else "本回合未成功" if row["success"] is False else "本回合结果未知"
        )
        video = html.escape(row["video"], quote=True)
        evidence = (
            f'<a href="{html.escape(row["source_manifest"], quote=True)}">来源记录</a>'
            if row["source_manifest"]
            else "无来源记录"
        )
        report = (
            f' · <a href="{html.escape(row["report"], quote=True)}">任务报告</a>'
            if row["report"]
            else ""
        )
        details = " · ".join(
            part
            for part in (
                label,
                status,
                f"seed {row['seed']}" if row["seed"] is not None else "",
                f"源码 {str(row['source_commit'])[:7]}" if row["source_commit"] else "",
            )
            if part
        )
        cards.append(
            f'<article><h2>{html.escape(row["title"])}</h2>'
            f'<p>{html.escape(row["description"])}</p>'
            f'<video controls preload="none" playsinline src="{video}"></video>'
            f'<p>{html.escape(details)}</p>'
            f'<p><a href="{video}">打开 MP4</a> · {evidence}{report}</p>'
            '<label>播放速度 <select onchange="this.closest(\'article\').querySelector(\'video\').playbackRate=Number(this.value)">'
            '<option value="0.5">0.5×</option><option value="1" selected>1×</option>'
            '<option value="2">2×</option><option value="4">4×</option></select></label></article>'
        )
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>'
        "body{background:#10161f;color:#e6edf5;font:16px/1.6 system-ui,'Microsoft YaHei';"
        "max-width:1400px;margin:auto;padding:24px}a{color:#89bdff}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:20px}"
        "article{background:#1b2635;border:1px solid #3b4c5e;border-radius:10px;padding:18px}"
        "article h2{margin:0}article p{color:#c1cede}video{width:100%;background:#000}"
        "select{margin-left:6px;padding:4px}</style>"
        f"<h1>{html.escape(title)}</h1>"
        "<p>按仿真时间制作的离线录像；播放器速度不表示仿真求解速度。"
        "单回合视频不代表整套评测通过，材料参数是否实测标定请查看场景说明。</p>"
        "<main>" + "\n".join(cards) + "</main></html>"
    )


def build_gallery(manifest: str | Path, output_dir: str | Path) -> Path:
    """先验证全部输入，再复制到新目录；拒绝覆盖已有目录。"""

    source_manifest = Path(manifest).resolve()
    title, entries = _prepare(source_manifest)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite gallery directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ibero_gallery_", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "videos").mkdir()
        (stage / "evidence").mkdir()
        rows = []
        for entry in entries:
            scene_id = entry["id"]
            video_target = stage / "videos" / f"{scene_id}.mp4"
            shutil.copy2(entry["video_source"], video_target)
            if sha256(video_target) != entry["video_sha256"]:
                raise OSError(f"Video copy changed content: {scene_id}")
            row = {
                key: value
                for key, value in entry.items()
                if key not in {"video_source", "source_manifest_source", "report_source"}
            }
            row["video"] = f"videos/{scene_id}.mp4"
            row["source_manifest"] = None
            row["report"] = None
            if entry["source_manifest_source"]:
                path = stage / "evidence" / f"{scene_id}.json"
                shutil.copy2(entry["source_manifest_source"], path)
                row["source_manifest"] = f"evidence/{scene_id}.json"
            if entry["report_source"]:
                path = stage / "evidence" / f"{scene_id}.report.json"
                shutil.copy2(entry["report_source"], path)
                row["report"] = f"evidence/{scene_id}.report.json"
            rows.append(row)
        (stage / "index.json").write_text(
            json.dumps({"title": title, "entries": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (stage / "index.html").write_text(_page(title, rows), encoding="utf-8")
        (stage / "README.md").write_text(
            "# IBERO 演示目录\n\n双击 `index.html` 浏览；复制整个目录即可分享，"
            "所有视频与来源记录均使用相对路径。\n\n"
            "`index.json` 保存复制后视频的 SHA-256、回合状态与来源摘要。"
            "页面中的“已核验”只表示来源导出记录声明全片解码通过，"
            "且其视频哈希与输入文件相符；不代表当前源码完整回归或物理实测标定。\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite gallery directory: {output}")
        os.rename(stage, output)
    return output / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="JSON with title and entries")
    parser.add_argument("--output-dir", required=True, type=Path, help="new portable directory")
    args = parser.parse_args()
    try:
        result = build_gallery(args.manifest, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Gallery error: {exc}\n")
    print(f"Gallery ready: {result}")


if __name__ == "__main__":
    main()
