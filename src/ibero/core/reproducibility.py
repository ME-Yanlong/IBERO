"""除场景和资产外，同时标识生成模型、力学和控制源码。"""

import hashlib
from pathlib import Path


def simulation_source_hash() -> str:
    # 每次从文件内容重新计算；开发过程中同一进程也可能修改源码。
    # 缓存会让回放/验收错误接受旧模型生成的新文件，破坏证据身份。
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
