"""收集已全部完成的正常/变体组，旧错误动作仍可继续运行；不修改原进度或报告。"""

import argparse
import hashlib
import json
from pathlib import Path


def source_hash(root):
    root = root / "src/ibero"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--worktree", type=Path, required=True)
    p.add_argument(
        "--scene", choices=["latch_release", "harness_unplug"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to replace existing evidence")
    original = json.loads(args.progress.read_text(encoding="utf-8"))
    before = source_hash(args.worktree)
    expected = {(args.scene, s, "normal") for s in range(20)} | {
        (args.scene + "_" + v, s, "normal")
        for v in ("thinner", "shallower")
        for s in range(100, 105)
    }
    rows = [r for r in original["results"] if r["case"] == "normal"]
    actual = [(r["scene"], r["seed"], r["case"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            "Normal and variant group is not complete or has duplicate trials"
        )
    identities = []
    for row in rows:
        raw = json.loads(
            (Path(row["path"]) / "report.json").read_text(encoding="utf-8")
        )
        identities.append(raw["manifest"]["source_hash"] == before)
    frozen = before == source_hash(args.worktree) == original["source_hash"] and all(
        identities
    )
    report = {
        "scope": "completed_normal_variants_only",
        "source_hash": before,
        "gate_version": original["gate_version"],
        "checks": {"frozen_source": frozen},
        "results": rows,
        "parent_progress": str(args.progress.resolve()),
        "explicitly_excluded": "All original counterfactual trials remain in their parent report; corrected counterfactuals are validated separately.",
        "passed": frozen and all(r["passed"] for r in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        args.output,
        "frozen",
        frozen,
        "normal/variant passes",
        sum(r["passed"] for r in rows),
        "/",
        len(rows),
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
