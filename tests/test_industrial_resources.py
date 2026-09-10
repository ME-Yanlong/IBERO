"""启动资源估算独立于物理门槛，物理内存空闲不能掩盖提交额度耗尽。"""

import importlib.util
from pathlib import Path

import pytest


def test_worker_budget_uses_commit_headroom_and_does_not_claim_portable_check():
    path = Path(__file__).resolve().parents[1] / "scripts/industrial_resources.py"
    spec = importlib.util.spec_from_file_location("industrial_resources_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    low = dict(
        available_commit_bytes=module.GIB, available_physical_bytes=8 * module.GIB
    )
    assert not module.assess_budget(1, low)["passed"]
    healthy = dict(available_commit_bytes=8 * module.GIB)
    assert module.assess_budget(3, healthy)["passed"]
    assert not module.assess_budget(8, healthy)["passed"]
    unknown = module.assess_budget(1, {"available_commit_bytes": None})
    assert not unknown["checked"]  # 未知平台没有冒称通过内存检查。
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            module.assess_budget(invalid, healthy)
