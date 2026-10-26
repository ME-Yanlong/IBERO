"""根据 MP4 与同名来源 JSON 自动生成离线演示目录的输入清单。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _reference(path: Path, base: Path) -> str:
    """优先写相对路径；不同 Windows 盘符时退回绝对路径。"""

    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return str(path)


def make_manifest(videos_dir: str | Path, output: str | Path, *, title: str) -> Path:
    """只搜一层 MP4；真实编码/哈希验证交给 build_demo_gallery。"""

    videos = Path(videos_dir).resolve()
    output = Path(output).resolve()
    if not videos.is_dir():
        raise NotADirectoryError(f"Video directory does not exist: {videos}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    files = sorted(videos.glob("*.mp4"))
    if not files:
        raise ValueError(f"No MP4 videos in {videos}")
    entries = []
    for file in files:
        source = file.with_suffix(".json")
        report = file.with_suffix(".report.json")
        metadata = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Video source manifest must be an object: {source}")
        item = {
            "id": file.stem,
            "title": metadata.get("title", file.stem),
            "video": _reference(file, output.parent),
        }
        if source.is_file():
            item["source_manifest"] = _reference(source, output.parent)
        if report.is_file():
            item["report"] = _reference(report, output.parent)
        entries.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"title": title, "entries": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="IBERO · 任务演示")
    args = parser.parse_args()
    try:
        result = make_manifest(args.videos_dir, args.output, title=args.title)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Manifest error: {exc}\n")
    print(f"Gallery manifest ready: {result}")


if __name__ == "__main__":
    main()
