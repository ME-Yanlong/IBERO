"""除场景和资产外，同时标识生成模型、力学和控制源码。"""

from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=1)
def simulation_source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
