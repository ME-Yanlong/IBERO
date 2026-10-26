"""立式布线的三视角与状态面板，复用现有 Enter/暂停/回放审阅流程。"""

from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from ibero.control.cable_routing import routing_policy_for
from ibero.industrial_review import IndustrialReviewer
from ibero.review import Trace


class RoutingViews:
    def __init__(self, model, clip_origin):
        self.model = model
        self.data = mujoco.MjData(model)
        self.cameras = []
        for lookat, distance, azimuth, elevation in (
            ([0.43, 0, 0.84], 0.95, 60, -22),
            (clip_origin, 0.24, 90, -10),
            (clip_origin, 0.38, 300, -17),
        ):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = lookat
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
            self.cameras.append(camera)
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        self.font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
        self.renderer = mujoco.Renderer(model, height=320, width=480)

    def set_state(self, state):
        mujoco.mj_setState(self.model, self.data, state, Trace.spec)
        mujoco.mj_forward(self.model, self.data)

    def render(self, infos):
        canvas = Image.new("RGB", (960, 640), (20, 25, 32))
        labels = ("立式布线全景", "夹口正面：压入与保持", "夹口侧面：线与唇边")
        for i, (camera, label) in enumerate(zip(self.cameras, labels)):
            self.renderer.update_scene(self.data, camera=camera)
            tile = Image.fromarray(self.renderer.render().copy())
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 480, 28), fill=(20, 25, 32))
            draw.text((10, 2), label, fill="white", font=self.font)
            canvas.paste(tile, ((i % 2) * 480, (i // 2) * 320))
        info = infos[-1] if infos else {}
        metrics = info.get("metrics", {})
        audit = info.get("audit", {})
        lines = (
            "IBERO 布线任务 · 示例材料参数",
            f"场景 {info.get('scene_id', '?')}  阶段 {info.get('stage', '?')}",
            f"已固定 {int(metrics.get('retained_count', 0))}/{int(metrics.get('clip_count', 0))}  双手离开 {metrics.get('hands_clear', False)}",
            f"夹点速度 {metrics.get('cable_speed_m_s', 0):.3f} m/s  保持 {metrics.get('hold_seconds', 0):.2f} s",
            f"峰值线张力 {audit.get('peak_cable_tension_n', 0):.2f} N",
            f"结果 {'完成' if info.get('is_success') else info.get('failure_reason') or '进行中'}",
        )
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(lines):
            draw.text((492, 332 + i * 34), line, fill="white", font=self.font)
        return canvas

    def close(self):
        self.renderer.close()


class RoutingReviewer(IndustrialReviewer):
    """复用稳定的桌面生命周期和只读物理回放。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root.title("IBERO 双臂布线任务 / Enter 运行一次")

    def _bind_model(self):
        if self.views is not None:
            self.views.close()
        origin = self.env.scene.config["workcell"]["clips"][0]["origin_m"]
        self.views = RoutingViews(self.env.model, origin)

    def _make_policy(self):
        config = self.env.scene.config
        return routing_policy_for(config["id"], config["kind"])
