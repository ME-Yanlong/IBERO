"""加工四视图：显示拥有独立模型/材料，视觉切屑永远不进入物理或任务。"""

import copy
from bisect import bisect_left
from pathlib import Path
import time
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ibero.industrial_review import IndustrialReviewer
from ibero.core.stock_trace import StockTrace
from ibero.review import Trace
from ibero.materials.stock import VoxelStock
from ibero.materials.stock_collision import StockCollisionBinding
from ibero.control.milling_bench import MillingBenchScript


class MillingViews:
    def __init__(self, env, *, visual_chips=True):
        self.model = copy.deepcopy(env.model)
        self.data = mujoco.MjData(self.model)
        self.stock = VoxelStock(
            env.stock.params,
            env.stock.cell_size_m,
            origin=env.stock.origin,
            rotation=env.stock.rotation,
            max_cells=len(env.stock.centers),
        )
        # 即使创建窗口时已切过料，显示侧的基线也必须取原始碰撞/外观，而不是当前孔洞。
        ids = env.binding.geom_ids
        self.model.geom_contype[ids] = env.binding._initial_type
        self.model.geom_conaffinity[ids] = env.binding._initial_affinity
        self.model.geom_rgba[ids] = env.binding._initial_rgba
        self.binding = StockCollisionBinding(self.model, self.stock)
        self.blade = env.process.blade_geom
        self.blade_body = env.process.blade_body
        self.blade_body_geoms = np.flatnonzero(
            self.model.geom_bodyid == self.blade_body
        )
        self.renderer = mujoco.Renderer(
            self.model, 320, 480, max_geom=self.model.ngeom + 1000
        )
        self.cameras = []
        for lookat, distance, azimuth, elevation in (
            ([0.25, 0, 0.85], 1.35, 135, -20),
            (env.stock.origin, 0.085, 135, -25),
            (env.stock.origin, 0.085, 0, -89),
        ):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = lookat
            camera.distance, camera.azimuth, camera.elevation = (
                distance,
                azimuth,
                elevation,
            )
            self.cameras.append(camera)
        path = Path("C:/Windows/Fonts/msyh.ttc")
        self.font = (
            ImageFont.truetype(str(path), 15)
            if path.exists()
            else ImageFont.load_default()
        )
        self.visual_chips = visual_chips
        self.chip_positions = []
        self.frame_index = 0

    def set_frame(self, trace, index):
        count = trace.event_counts[index]
        events = trace.events[:count]
        self.binding.restore_events(events, self.data)
        mujoco.mj_setState(self.model, self.data, trace.states[index], Trace.spec)
        mode = trace.collision_modes[index]["milling_blade_contype"]
        self.model.geom_contype[self.blade] = mode
        self.model.geom_conaffinity[self.blade] = 1
        self.model.body_contype[self.blade_body] = np.bitwise_or.reduce(
            self.model.geom_contype[self.blade_body_geoms]
        )
        self.model.body_conaffinity[self.blade_body] = np.bitwise_or.reduce(
            self.model.geom_conaffinity[self.blade_body_geoms]
        )
        mujoco.mj_forward(self.model, self.data)
        self.binding.ensure_consistent()
        if self.stock.state_hash() != trace.material_hashes[index]:
            raise ValueError("Display material does not match recorded frame")
        self.frame_index = index
        # 只为最近一小段确有删除事件的帧绘制示意颗粒；无独立刚体、无额外受力。
        earlier = trace.event_counts[max(0, index - 15)]
        recent = events[earlier:]
        self.chip_positions = []
        for event in recent[-3:]:
            born = bisect_left(trace.event_counts, event.after_version)
            age = max(
                0.0,
                trace.infos[index].get("time_s", 0.0)
                - trace.infos[born].get("time_s", 0.0),
            )
            for cell_id in event.removed_ids[:12]:
                point = self.stock.local_to_world(self.stock.centers[cell_id]).copy()
                phase = (cell_id * 0.61803398875) % 1 * 2 * np.pi
                # 确定性的轻量弹道示意，回放同帧同外观；速度/重力不是切屑物性标定。
                radius = 0.007 + 0.02 * age
                point += [
                    radius * np.cos(phase),
                    radius * np.sin(phase),
                    0.009 + 0.04 * age - 0.15 * age**2,
                ]
                self.chip_positions.append(point)

    def _chips(self):
        if not self.visual_chips:
            return
        scene = self.renderer.scene
        for point in self.chip_positions:
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.0003] * 3),
                np.asarray(point),
                np.eye(3).ravel(),
                np.array([0.85, 0.67, 0.25, 1], dtype=np.float32),
            )
            scene.ngeom += 1

    def render(self, infos):
        canvas = Image.new("RGB", (960, 640), (20, 25, 32))
        titles = [
            "G1 工作站 / 右臂安全停放",
            "实际加工局部 / 原生实体孔槽",
            "工件俯视 / 金色颗粒仅为视觉示意",
        ]
        for i, camera in enumerate(self.cameras):
            self.renderer.update_scene(self.data, camera)
            self._chips()
            tile = Image.fromarray(self.renderer.render().copy())
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 480, 25), fill=(20, 25, 32))
            draw.text((8, 2), titles[i], font=self.font, fill="white")
            canvas.paste(tile, ((i % 2) * 480, (i // 2) * 320))
        draw = ImageDraw.Draw(canvas)
        last = infos[-1] if infos else {}
        force = float(np.linalg.norm(last.get("force_world_n", [0, 0, 0])))
        peaks = np.asarray(
            [r.get("peak_cutting_force_step_n", 0.0) for r in infos], dtype=float
        )
        peak = float(peaks.max()) if len(peaks) else 0.0
        lines = [
            f"材料/过程级原型，未实测标定 | {last.get('controller_phase', '等待')}",
            f"切削需求 {force:.2f} / 峰值 {peak:.2f} N | {last.get('actual_rpm', 0):.0f} rpm",
            f"去除 {self.stock.initial_volume_m3 - self.stock.volume_m3:.3e} m³ | 材料版本 {self.stock.version}",
            f"失败：{last.get('failure_reason') or last.get('invalid_reason') or '无'}",
        ]
        for i, line in enumerate(lines):
            draw.text((490, 326 + i * 22), line, font=self.font, fill="white")
        # 占据驱动 XZ 剖面；同一横纵比例，剖面图不能把 4 mm 薄板拉成方块。
        occupied = self.stock.occupied.reshape(self.stock.shape)
        middle = occupied[:, self.stock.shape[1] // 2, :]
        scale = min(410 / self.stock.params.size_m[0], 55 / self.stock.params.size_m[2])
        width, height = (
            self.stock.params.size_m[0] * scale,
            self.stock.params.size_m[2] * scale,
        )
        x0, z0 = 720 - width / 2, 475
        centers = self.stock.centers.reshape((*self.stock.shape, 3))[
            :, self.stock.shape[1] // 2, :, :
        ]
        half = self.stock.half_sizes.reshape((*self.stock.shape, 3))[
            :, self.stock.shape[1] // 2, :, :
        ]
        for i, k in np.ndindex(middle.shape):
            if middle[i, k]:
                c, h = centers[i, k], half[i, k]
                left = x0 + (c[0] - h[0] + self.stock.params.size_m[0] / 2) * scale
                top = z0 - (c[2] + h[2] + self.stock.params.size_m[2] / 2) * scale
                draw.rectangle(
                    (left, top, left + 2 * h[0] * scale, top + 2 * h[2] * scale),
                    fill=(140, 160, 180),
                )
        draw.rectangle((x0, z0 - height, x0 + width, z0), outline=(90, 105, 120))
        draw.text(
            (490, 478),
            "XZ 剖面；红：载荷需求峰值，橙：15 N 限额",
            font=self.font,
            fill=(190, 205, 220),
        )
        # 全回合历史按桶取峰值，不能在退刀后只画最后几秒零载荷，也不能漏掉窄脉冲。
        edges = np.linspace(0, len(peaks), min(400, len(peaks)) + 1, dtype=int)
        values = [
            float(peaks[a:b].max()) for a, b in zip(edges[:-1], edges[1:]) if b > a
        ]
        scale = max(15.0, peak)
        draw.line((510, 508, 510, 620, 935, 620), fill=(150, 165, 185))
        limit_y = 620 - 15 / scale * 105
        draw.line((510, limit_y, 935, limit_y), fill=(255, 185, 65))
        draw.text(
            (760, 503),
            f"0–{scale:.1f} N / 全回合",
            font=self.font,
            fill=(190, 205, 220),
        )
        points = [
            (
                510 + i * 425 / max(1, len(values) - 1),
                620 - value / scale * 105,
            )
            for i, value in enumerate(values)
        ]
        if len(points) > 1:
            draw.line(points, fill=(255, 110, 90), width=2)
        return canvas

    def close(self):
        self.renderer.close()


class MillingReviewer(IndustrialReviewer):
    """复用持久窗口事件循环；材料回看只触碰显示副本，不恢复 live 环境。"""

    def __init__(self, env, *, shape="through_hole", visual_chips=True, **kwargs):
        if shape != env.target.shape:
            raise ValueError("Viewer shape must match environment target identity")
        self.target = env.target
        self.visual_chips = visual_chips
        env.control_dt = 1 / env.scene.config["physics"]["control_hz"]
        env.max_episode_steps = round(
            env.scene.constraints["task"]["max_seconds"] / env.control_dt
        )
        super().__init__(env, **kwargs)
        self.root.title("IBERO G1 加工审阅 / Enter 重跑 / C 视觉切屑")
        self.root.bind("<c>", lambda event: self.toggle_chips())
        self.root.bind("<C>", lambda event: self.toggle_chips())
        self.tk.Button(
            self.root, text="切屑显示 C（仅视觉）", command=self.toggle_chips
        ).pack()

    def _make_policy(self):
        return MillingBenchScript(self.env, self.target)

    def _make_trace(self):
        return StockTrace(self.env)

    def _bind_model(self):
        if hasattr(self, "views"):
            self.views.close()
        self.views = MillingViews(self.env, visual_chips=self.visual_chips)

    def _set_view_frame(self):
        self.views.set_frame(self.trace, self.index)

    def _step(self):
        start = time.monotonic()
        target, rpm = self.policy.command(self.env)
        steps = round(self.env.control_dt / self.env.model.opt.timestep)
        info = self.env.step(target, rpm, substeps=steps)
        self.physics_wall_seconds += time.monotonic() - start
        info["controller_phase"] = self.policy.phase
        finish = (
            self.env._done
            or self.policy.finished
            or len(self.trace.states) >= self.steps
        )
        if finish:
            check = self.env.evaluate_task(self.target)
            info["task_check"] = check
            info["success"] = bool(check["result"]["success"] and self.policy.finished)
            if not info["success"]:
                info["failure_reason"] = info["invalid_reason"] or (
                    "shape_or_finish_check_failed"
                    if self.policy.finished
                    else "demo_step_limit"
                )
            self.env._done = True
        else:
            info["success"] = False
        self.trace.append(info)
        return "finish" if finish else "step"

    def toggle_chips(self):
        self.visual_chips = not self.visual_chips
        self.views.visual_chips = self.visual_chips
        self.draw()
