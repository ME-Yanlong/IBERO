"""把经过身份校验的真实仿真轨迹导出为恒定帧率视频；不重算/插值物理。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import ibero
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ibero.review import Trace


TITLES = {
    "cable_handover": "双臂线束端头交接",
    "cable_stretch": "双臂线束张力控制",
    "latch_release": "按压弹性卡扣并拔出",
    "harness_unplug": "带线束连接器：解锁、拔出、放置",
    "through_hole": "机器人铣削：通孔",
    "slot": "机器人铣削：直槽",
    "pocket": "机器人铣削：浅腔",
}


def frame_indices(times, fps, speed):
    """按仿真时钟取最近的已记录状态；不生成虚构中间姿态。"""
    times = np.asarray(times, dtype=float)
    if (not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) < 0) or fps <= 0
            or not math.isfinite(speed) or speed <= 0):
        raise ValueError("Invalid timestamps, fps or speed")
    count = math.ceil((times[-1] - times[0]) / speed * fps) + 1
    requested = np.minimum(times[0] + np.arange(count) * speed / fps, times[-1])
    right = np.minimum(np.searchsorted(times, requested), len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(times[right] - requested) < abs(times[left] - requested), right, left)
    indices[-1] = len(times) - 1
    return indices


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def case_passed(last_info, source_report=None):
    """新工业场景与早期 Gym 场景的成功字段不同，缺失时不猜测成功。"""
    if source_report is not None:
        return source_report.get("passed")
    return last_info.get("success", last_info.get("is_success"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=("cable_handover", "cable_stretch", "latch_release", "harness_unplug", "plate_milling"))
    parser.add_argument("--shape", choices=("through_hole", "slot", "pocket"), default="through_hole")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--preview-only", action="store_true", help="仅检查首、中、末三帧，不编码视频")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(ibero.__file__).resolve().parents[2]
    scene = root / "scenes" / args.env
    if args.env == "plate_milling":
        from ibero.envs.plate_milling import PlateMillingEnv
        from ibero.core.stock_trace import StockTrace
        from ibero.milling_review import MillingViews
        env = PlateMillingEnv(scene, shape=args.shape)
        trace = StockTrace(env).load(args.trace)
        views = MillingViews(env, visual_chips=True, render_quality="fast")
    elif args.env in ("latch_release", "harness_unplug"):
        from ibero.envs.latch_release import LatchReleaseEnv
        from ibero.envs.harness_unplug import HarnessUnplugEnv
        from ibero.industrial_review import IndustrialViews
        cls = LatchReleaseEnv if args.env == "latch_release" else HarnessUnplugEnv
        env = cls(scene)
        trace = Trace(env).load(args.trace)
        views = IndustrialViews(env.model, env.resolved_config["initialization"]["origin_m"])
    else:
        from ibero.multiview import CableHandoverMultiView

        class VideoCableViews(CableHandoverMultiView):
            # 仅改变显示分辨率，物理/场景配置不变。
            tile_width, tile_height = 480, 320

        env = ibero.CableHandoverEnv(scene_path=scene)
        trace = Trace(env).load(args.trace)
        views = VideoCableViews(env)
    encoder = None
    try:
        # INTEGRATION 首项为仿真时间；用引擎恢复结果交叉检查，避免错当墙钟。
        times = np.array([s[0] for s in trace.states])
        check_data = mujoco.MjData(env.model)
        for i in (0, len(times) - 1):
            mujoco.mj_setState(env.model, check_data, trace.states[i], Trace.spec)
            if abs(check_data.time - times[i]) > 1e-10:
                raise ValueError("Unexpected state timestamp layout")
        indices = frame_indices(times, args.fps, args.speed)
        with np.load(args.trace, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
        identity = meta.get("identity", meta)
        source_report = json.loads(args.report.read_text(encoding="utf-8")) if args.report else None
        passed = case_passed(trace.infos[-1], source_report)
        title = TITLES[args.shape if args.env == "plate_milling" else args.env]
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
        large = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 25)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        print(f"{title}: {len(times)} states, {times[-1]-times[0]:.3f} sim s, source {commit[:7]}, passed={passed}", flush=True)

        def render(index):
            # 没有 mj_step/env.step；铣削逐帧恢复真实材料事件并验证材料哈希。
            if args.env == "plate_milling":
                views.set_frame(trace, index)
                base = views.render(trace.infos[:index + 1])
            elif args.env in ("latch_release", "harness_unplug"):
                views.set_state(trace.states[index])
                base = views.render(trace.infos[:index + 1])
            else:
                trace.restore(index)
                base = Image.fromarray(views.render())
                draw = ImageDraw.Draw(base)
                for n, label in enumerate(views.labels):
                    x, y = n % 2 * 480, n // 2 * 320
                    draw.rectangle((x, y, x + 480, y + 28), fill=(20, 25, 32))
                    draw.text((x + 10, y + 2), label, font=font, fill="white")
            canvas = Image.new("RGB", (960, 752), (16, 22, 31))
            canvas.paste(base, (0, 72))
            draw = ImageDraw.Draw(canvas)
            draw.text((14, 5), title, font=large, fill="white")
            info = trace.infos[index]
            phase = info.get("controller_phase") or info.get("latch_state") or info.get("mode", "")
            status = f"t = {times[index]:.2f} / {times[-1]:.2f} s  |  {phase}"
            if index == len(times) - 1:
                status += "  |  本回合报告：" + ("通过" if passed else "未通过/未确认")
            draw.text((14, 42), status, font=font, fill=(126, 224, 177))
            if args.env.startswith("cable"):
                tension = info.get("cable_tension_n", info.get("audit", {}).get("cable_tension_n"))
                if tension is not None:
                    draw.text((660, 8), f"线束张力 {tension:.2f} N", font=font, fill="white")
            note = f"仿真时间 {args.speed:g}× · 真实轨迹离线回放（非实时录像） · 源码 {commit[:7]}"
            if args.env == "plate_milling":
                note += " · 飞屑仅可视化"
            draw.text((14, 719), note, font=font, fill=(180, 195, 210))
            return canvas

        previews = [render(i) for i in (0, len(times) // 2, len(times) - 1)]
        previews[-1].save(args.output.with_suffix(".poster.png"))
        sheet = Image.new("RGB", (960, 752 * 3))
        for row, picture in enumerate(previews):
            sheet.paste(picture, (0, row * 752))
        sheet.save(args.output.with_suffix(".preview.png"))
        if args.preview_only:
            print("Preview complete (no video encoded)", flush=True)
            return
        # H.264 + yuv420p + faststart：Windows 播放器、浏览器通用；禁止覆盖。
        command = [str(args.ffmpeg.resolve()), "-hide_banner", "-loglevel", "warning", "-n",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x752", "-r", str(args.fps),
                   "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        begin_hold, end_hold = round(1.5 * args.fps), round(2.0 * args.fps)
        with args.output.with_suffix(".encoder.log").open("w", encoding="utf-8") as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, creationflags=creationflags)
            for _ in range(begin_hold):
                encoder.stdin.write(previews[0].tobytes())
            previous, pixels = None, None
            for n, index in enumerate(indices):
                if index != previous:
                    pixels = render(int(index)).tobytes()
                    previous = index
                encoder.stdin.write(pixels)
                if n % max(1, len(indices) // 10) == 0:
                    print(f"{title}: {n}/{len(indices)} frames ({time.perf_counter()-started:.1f}s elapsed)", flush=True)
            for _ in range(end_hold):
                encoder.stdin.write(pixels)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Encoder failed; inspect .encoder.log")
        # 全片解码检查，不仅验证 MP4 文件存在；progress 同时核对输出帧数。
        verified = subprocess.run([str(args.ffmpeg.resolve()), "-v", "error", "-i", str(args.output),
                                   "-progress", "pipe:1", "-f", "null", "-"],
                                  capture_output=True, text=True, creationflags=creationflags, check=True)
        decoded_frames = [int(line.split("=", 1)[1]) for line in verified.stdout.splitlines() if line.startswith("frame=")][-1]
        expected_frames = len(indices) + begin_hold + end_hold
        if verified.stderr.strip() or decoded_frames != expected_frames:
            raise RuntimeError(f"Decode verification failed: frames={decoded_frames}/{expected_frames}, {verified.stderr}")
        manifest = {
            "title": title, "environment": args.env, "shape": args.shape if args.env == "plate_milling" else None,
            "source_root": str(root), "source_commit": commit,
            "source_hash": identity.get("source_hash"), "asset_hash": identity.get("asset_hash"),
            "scene_hash": identity.get("scene_hash"), "mujoco_version": mujoco.__version__, "seed": meta.get("seed"),
            "trace": str(args.trace.resolve()), "trace_sha256": sha256(args.trace),
            "source_report": str(args.report.resolve()) if args.report else None,
            "source_report_sha256": sha256(args.report) if args.report else None,
            "source_case_passed": passed, "source_last_info": trace.infos[-1],
            "simulation_start_s": times[0], "simulation_end_s": times[-1], "source_states": len(times),
            "fps": args.fps, "speed_sim_seconds_per_video_second": args.speed,
            "initial_hold_frames": begin_hold, "final_hold_frames": end_hold,
            "video_frames": expected_frames, "video_duration_s": expected_frames / args.fps,
            "state_sampling": "nearest recorded state; no interpolation", "physics_steps_during_render": 0,
            "material_hash_checked_per_rendered_frame": args.env == "plate_milling",
            "visual_chips_only": args.env == "plate_milling", "full_decode_verified": True,
            "video_sha256": sha256(args.output), "encoding_command": command,
            "export_wall_seconds": time.perf_counter() - started,
            "scope": "单条历史轨迹演示；不替代当前版本完整回归或最终 S9 验收。材料参数尚非实测标定。",
        }
        args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"COMPLETE: {args.output} | {expected_frames/args.fps:.2f}s | decoded {decoded_frames} frames", flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        views.close()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
