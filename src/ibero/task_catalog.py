"""列出已配置任务与台架；读取菜谱，不执行贡献者 Python 代码。"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import yaml


DEFAULT_SCENES_ROOT = Path(__file__).resolve().parents[2] / "scenes"


@dataclass(frozen=True)
class TaskCard:
    """任务目录的静态索引，成功条件仍以实际 task_spec.py 为准。"""

    scene_id: str
    family: str
    role: str
    instruction: str
    scene_path: str
    schema_version: str
    scene_source_sha256: str


def _static_instruction(path: Path) -> str:
    """读取类属性或 TASK_SPEC 构造参数里的常量，不导入场景代码。"""

    if not path.is_file():
        return ""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        # 新布线菜谱直接构造共享 TaskSpec，instruction 位于具名参数。
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_SPEC"
            for target in node.targets
        ) and isinstance(node.value, ast.Call):
            for keyword in node.value.keywords:
                if keyword.arg == "instruction":
                    try:
                        value = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                        break
                    if isinstance(value, str):
                        return value.strip()
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "instruction"
                for target in item.targets
            ):
                continue
            try:
                value = ast.literal_eval(item.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
            if isinstance(value, str):
                return value.strip()
    return ""


def _source_sha256(scene: Path) -> str:
    """目录索引用的源文件摘要，不声称代替仿真模型/资产 identity。"""

    digest = hashlib.sha256()
    for name in ("scene_config.yaml", "constraints.yaml", "task_spec.py"):
        path = scene / name
        digest.update(name.encode("ascii"))
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def discover_tasks(scenes_root: str | Path = DEFAULT_SCENES_ROOT) -> tuple[TaskCard, ...]:
    """扫描场景目录；已注册/可运行性由 scene checker 单独验证。"""

    root = Path(scenes_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Scene directory does not exist: {root}")
    cards = []
    for scene in sorted(root.iterdir(), key=lambda p: p.name):
        config_path = scene / "scene_config.yaml"
        if not scene.is_dir() or not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Scene configuration must be a mapping: {config_path}")
        scene_id = config.get("id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"Scene id must be a nonempty string: {config_path}")
        if scene_id != scene.name:
            raise ValueError(f"Scene id {scene_id!r} does not match directory {scene.name!r}")
        family = config.get("task_family") or config.get("suite") or config.get("kind")
        if not isinstance(family, str) or not family:
            family = "unspecified"
        role = (
            "bench"
            if scene_id.endswith("_bench")
            else "variant"
            if scene_id.endswith(("_thinner", "_shallower", "_offset"))
            else "task"
        )
        schema = config.get("schema_version", "")
        cards.append(
            TaskCard(
                scene_id=scene_id,
                family=family,
                role=role,
                instruction=_static_instruction(scene / "task_spec.py"),
                # 相对 --scenes-dir，目录清单在另一台机器上仍然可读。
                scene_path=scene.name,
                schema_version=str(schema),
                scene_source_sha256=_source_sha256(scene),
            )
        )
    return tuple(cards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES_ROOT)
    parser.add_argument("--json", action="store_true", help="write machine-readable cards")
    args = parser.parse_args()
    try:
        cards = discover_tasks(args.scenes_dir)
    except (OSError, ValueError, yaml.YAMLError, SyntaxError) as exc:
        parser.exit(2, f"Task catalog error: {exc}\n")
    if args.json:
        print(json.dumps([asdict(card) for card in cards], ensure_ascii=False, indent=2))
        return
    for card in cards:
        print(f"{card.scene_id:30} {card.role:7} {card.family:22} {card.instruction}")
    print(f"{len(cards)} scene recipes. Use ibero.check_scene for validation.")


if __name__ == "__main__":
    main()
