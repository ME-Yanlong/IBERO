"""验收启动前的保守资源检查；只控制进程并发，不改变任何物理参数。"""

import ctypes
import json
from pathlib import Path
import sys

GIB = 1024**3


def memory_snapshot():
    if sys.platform != "win32":
        return {
            "platform": sys.platform,
            "available_commit_bytes": None,
            "note": "Windows commit-budget guard unavailable; not a capacity certificate",
        }

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load_percent", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    query.argtypes, query.restype = [ctypes.POINTER(MemoryStatus)], ctypes.c_int
    if not query(ctypes.byref(state)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "platform": sys.platform,
        "available_commit_bytes": int(state.available_page_file),
        "available_physical_bytes": int(state.available_physical),
        "physical_load_percent": int(state.load_percent),
    }


def assess_budget(workers, snapshot):
    if type(workers) is not int or workers < 1:
        raise ValueError("Positive integer worker count required")
    # 本机实测每个已导入依赖的工作进程约 1.2 GiB private commit。
    # 取 1.5 GiB/进程＋2 GiB 余量，属于启动估算，不是峰值内存或无 OOM 保证。
    required = int((2.0 + 1.5 * workers) * GIB)
    available = snapshot.get("available_commit_bytes")
    return dict(
        scope="startup_resource_estimate_not_physics_gate",
        workers=workers,
        snapshot=snapshot,
        required_commit_bytes=required,
        estimate_per_worker_gib=1.5,
        reserve_gib=2.0,
        checked=available is not None,
        passed=available is None or available >= required,
    )


def require_worker_budget(output, workers):
    report = assess_budget(workers, memory_snapshot())
    path = Path(output) / "resource_preflight.json"
    if path.exists():
        raise FileExistsError("Refusing to replace existing resource evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if not report["passed"]:
        print(
            "Insufficient Windows commit headroom; reduce --workers or wait for "
            "existing validation jobs to finish. No physics cases started.",
            flush=True,
        )
        raise SystemExit(2)
    return report
