"""S9 单源码 G4 汇总：完整 34 用例各自重读，旧多版本里程碑不冒充当前回归。"""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_group(path, scene, source):
    group = read(path)
    expected = (
        {(scene, s, "normal") for s in range(20)}
        | {
            (scene + "_" + variant, s, "normal")
            for variant in ("thinner", "shallower")
            for s in range(100, 105)
        }
        | {
            (scene, 0, case)
            for case in ("no_press", "partial_press", "offset_press", "early_release")
        }
    )
    rows, identities = [], []
    for summary in group.get("results", []):
        identity = (summary.get("scene"), summary.get("seed"), summary.get("case"))
        identities.append(identity)
        raw_path = Path(summary.get("path", "")) / "report.json"
        row = dict(
            identity=identity,
            passed=False,
            source_identity_valid=False,
            report=str(raw_path),
        )
        if raw_path.is_file():
            raw = read(raw_path)
            final = raw.get("final", {})
            normal = identity[2] == "normal"
            row["source_identity_valid"] = (
                raw.get("manifest", {}).get("source_hash") == source
                and tuple(raw.get(k) for k in ("scene", "seed", "case")) == identity
                and type(raw.get("seed")) is int
            )
            valid = (
                raw.get("passed") is True
                and not raw.get("exception")
                and raw.get("manifest", {}).get("source_hash") == source
                and tuple(raw.get(k) for k in ("scene", "seed", "case")) == identity
                and type(raw.get("seed")) is int
                and (raw_path.parent / "trace.npz").is_file()
            )
            if normal:
                valid &= (
                    raw.get("success") is True
                    and final.get("success") is True
                    and not final.get("invalid_reason")
                    and bool(raw.get("metric_checks"))
                    and all(v is True for v in raw.get("metric_checks", {}).values())
                )
            else:
                valid &= (
                    raw.get("success") is False
                    and final.get("success") is False
                    and any(r.get("grasped") is True for r in raw.get("trace", []))
                    and (
                        final.get("released") is False
                        or bool(final.get("invalid_reason"))
                    )
                )
            row.update(
                passed=bool(valid),
                exception=raw.get("exception"),
                invalid_reason=final.get("invalid_reason"),
                report_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            )
        rows.append(row)
    checks = dict(
        exact_denominator=len(identities) == 34 and set(identities) == expected,
        all_raw_identities_present=all(r["source_identity_valid"] for r in rows),
        frozen_source=group.get("source_hash") == source
        and group.get("checks", {}).get("frozen_source") is True,
    )
    for variant, required in (
        (scene, 18),
        (scene + "_thinner", 4),
        (scene + "_shallower", 4),
    ):
        selected = [
            r
            for r in rows
            if r["identity"][0] == variant and r["identity"][2] == "normal"
        ]
        checks[variant] = sum(r["passed"] for r in selected) >= required
    faults = [r for r in rows if r["identity"][2] != "normal"]
    checks["counterfactuals"] = len(faults) == 4 and all(r["passed"] for r in faults)
    return dict(passed=all(checks.values()), checks=checks, results=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("latch", "harness", "latch-viewer", "harness-viewer", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to replace evidence")
    source = read(args.latch).get("source_hash")
    results = {
        "latch_release": check_group(args.latch, "latch_release", source),
        "harness_unplug": check_group(args.harness, "harness_unplug", source),
    }
    checks = {name: row["passed"] for name, row in results.items()}
    checks["source_identity"] = isinstance(source, str) and len(source) == 64
    for name in ("latch_viewer", "harness_viewer"):
        row = read(getattr(args, name))
        checks[name] = (
            row.get("passed") is True
            and row.get("full") is True
            and row.get("scene")
            == ("latch_release" if name == "latch_viewer" else "harness_unplug")
            and row.get("source_hash") == source
            and all(
                row.get(k) is True
                for k in (
                    "successful_episode_1",
                    "successful_episode_2",
                    "persistent_finish",
                    "source_unchanged_during_run",
                )
            )
        )
    result = dict(
        scope="G4_single_source_full_matrices_and_two_full_viewers",
        source_hash=source,
        results=results,
        checks=checks,
        passed=all(checks.values()),
        inputs={k: str(v.resolve()) for k, v in vars(args).items() if k != "output"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output, result["passed"], checks, flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
