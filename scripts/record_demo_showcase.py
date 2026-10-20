"""逐个制作七段 Core-0.1 演示录像及本地视频目录（使用原始源码快照）。"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/demo_videos_20260928")
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.ffmpeg = args.ffmpeg.resolve()
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_source = ROOT / "artifacts/validation_worktrees/industrial_final_s9_20260911"
    milling_source = ROOT / "artifacts/validation_worktrees/milling_regularized_s8_20260911"
    final_evidence = ROOT / "artifacts/industrial_core01/final_s9/3585854"
    milling_evidence = ROOT / "artifacts/industrial_core01/stock/milling_process/a557bb0_unseen_seed200_three_shapes"
    cases = [
        ("01_cable_handover", "cable_handover", None, final_source, args.output_dir / "traces/cable_handover.npz", None),
        ("02_cable_stretch", "cable_stretch", None, final_source, args.output_dir / "traces/cable_stretch.npz", None),
        ("03_latch_release", "latch_release", None, final_source, final_evidence / "latch-robot_evidence/result/latch_release_0_normal/trace.npz", True),
        ("04_harness_unplug", "harness_unplug", None, final_source, final_evidence / "harness-robot_evidence/result/harness_unplug_0_normal/trace.npz", True),
        *[(f"0{i}_milling_{shape}", "plate_milling", shape, milling_source, milling_evidence / shape / "trace.npz", True)
          for i, shape in enumerate(("through_hole", "slot", "pocket"), 5)],
    ]
    # 整组预检：找不到历史源码/工业轨迹就停止，不以不匹配的新模型强行回放。
    for _, env_id, _, source, trace, report in cases:
        if not (source / "src/ibero").is_dir():
            raise FileNotFoundError(source)
        if env_id not in ("cable_handover", "cable_stretch") and not trace.is_file():
            raise FileNotFoundError(trace)
        if report and not trace.with_name("report.json").is_file():
            raise FileNotFoundError(trace.with_name("report.json"))
    results = []
    for name, env_id, shape, source, trace, has_report in cases:
        child_env = dict(os.environ, PYTHONPATH=str(source / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if not trace.exists():
            # 仅旧线束场景缺失轨迹时录制；工业场景必须复用已完成证据，避免重启长回归。
            subprocess.run([sys.executable, "-X", "utf8", "-m", "ibero.demo", "--env", env_id,
                            "--seed", "0", "--save-trace", str(trace)], cwd=source, env=child_env, check=True)
        output = args.output_dir / f"{name}.mp4"
        report = trace.with_name("report.json") if has_report else None
        if output.exists():
            # 可重复执行，但只有完整解码检查和源文件摘要均匹配的产物才能跳过。
            meta = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            if (not meta["full_decode_verified"] or meta["video_sha256"] != sha256(output)
                    or meta["trace_sha256"] != sha256(trace)
                    or (report and meta["source_report_sha256"] != sha256(report))):
                raise ValueError(f"Existing artifact failed integrity check: {output}")
            print(f"Verified existing: {name}", flush=True)
        else:
            command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/export_demo_video.py"),
                       "--env", env_id, "--trace", str(trace), "--output", str(output),
                       "--ffmpeg", str(args.ffmpeg)]
            if shape:
                command += ["--shape", shape]
            if report:
                command += ["--report", str(report)]
            print(f"EXPORT: {name}", flush=True)
            subprocess.run(command, cwd=source, env=child_env, check=True)
            meta = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        results.append(dict(file=output.name, manifest=output.with_suffix(".json").name,
                            title=meta["title"], duration_s=meta["video_duration_s"],
                            source_commit=meta["source_commit"], seed=meta["seed"],
                            passed=meta["source_case_passed"], sha256=meta["video_sha256"]))
    args.output_dir.joinpath("index.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for row in results:
        name, title = row["file"], html.escape(row["title"])
        cards.append(f'''<article><h2>{title}</h2>
<video controls preload="none" playsinline poster="{Path(name).with_suffix('.poster.png').name}"><source src="{name}" type="video/mp4"></video>
<p>{row['duration_s']:.2f} 秒 · 仿真时间 1× · seed {row['seed']} · 源码 {row['source_commit'][:7]} · 本回合 {'通过' if row['passed'] else '未确认'}</p>
<a href="{name}">直接打开 MP4</a> · <a href="{row['manifest']}">来源与校验记录</a>
<label>播放速度 <select onchange="this.closest('article').querySelector('video').playbackRate=Number(this.value)">
<option value="0.5">0.5× 慢看</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label></article>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IBERO Core-0.1 仿真演示录像</title><style>
body{background:#10161f;color:#dde7f0;font:16px/1.65 system-ui,'Microsoft YaHei';max-width:1400px;margin:auto;padding:28px}
h1{margin-bottom:8px}a{color:#81bbff}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:24px}
article{background:#192332;border:1px solid #344458;border-radius:10px;padding:18px}h2{margin:0 0 12px;font-size:22px}
video{width:100%;background:#000}label{display:inline-block;margin-left:15px}select{padding:4px}p{color:#b5c4d4}
</style><h1>IBERO Core-0.1 · 交互仿真录像</h1>
<p>完整动作，多视角 + 接触/材料信息。按仿真时钟 1× 播放，与计算耗时无关；可选 0.5× / 2× / 4×。每段开头停留 1.52 秒、末尾停留 2 秒。</p>
<p>这是经过身份校验的真实仿真轨迹离线渲染，不是实时性能展示，也不是实机录像。线束两项为此次补录，其余为已完成的历史成功回合；不代表最终 S9 全量验收完成。铣削飞屑仅为可视化，材料/加工参数尚未实测标定。</p><main>'''
    args.output_dir.joinpath("index.html").write_text(page + "\n".join(cards) + "</main></html>", encoding="utf-8")
    readme = "# Core-0.1 演示录像\n\n双击 `index.html` 浏览全部视频，或者直接用播放器打开 MP4。无需启动 Python 或仿真器。\n\n"
    readme += "所有视频：960×752、25 fps、H.264、仿真时间 1×、不插值物理状态；开头 1.52 秒/末尾 2 秒静帧。网页可切换 0.5×、2×、4×。\n\n"
    readme += "| 演示 | 视频 | 时长 | 源码 |\n|---|---|---:|---|\n"
    for row in results:
        readme += f"| {row['title']} | [{row['file']}]({row['file']}) | {row['duration_s']:.2f} s | {row['source_commit'][:7]} |\n"
    readme += "\n同名 JSON 保存轨迹/源码/场景/资产身份、seed、时间映射、成功依据、视频摘要和全片解码检查；poster 为末帧，preview 为首中末三帧。铣削每个渲染状态均恢复材料事件并核对材料哈希。\n\n"
    readme += "注意：这是单回合能力展示，不是当前版本全量回归通过证明，也不解决仿真计算慢的问题。卡扣/切削为当前近似模型与示例参数；飞屑仅可视化。源轨迹路径见同名 JSON；历史 worktree 必须保留才能按原始身份复现。\n"
    args.output_dir.joinpath("README.md").write_text(readme, encoding="utf-8")
    print(f"ALL COMPLETE: {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
