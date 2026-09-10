"""工业任务四视图审阅：物理线程与显示用 MjData 分离，历史帧不恢复积分。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ibero.review import Trace
from ibero.control.latch import LatchReleaseScript


class IndustrialViews:
    def __init__(self, model, origin):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, 320, 480)
        self.cameras = []
        for lookat, distance, azimuth, elevation in (
            ([0.2, 0, 0.82], 1.35, 135, -20),
            (origin, 0.25, 180, -20),
            (origin, 0.25, 0, -89),
        ):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = lookat
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation
            self.cameras.append(cam)
        path = Path("C:/Windows/Fonts/msyh.ttc")
        self.font = (
            ImageFont.truetype(str(path), 17)
            if path.exists()
            else ImageFont.load_default()
        )

    def set_state(self, state):
        mujoco.mj_setState(self.model, self.data, state, Trace.spec)
        mujoco.mj_forward(self.model, self.data)

    def render(self, infos):
        canvas = Image.new("RGB", (960, 640), (20, 25, 32))
        titles = ["工作站全景", "扣齿与肩部局部", "俯视 / 按压位置"]
        for i, cam in enumerate(self.cameras):
            self.renderer.update_scene(self.data, camera=cam)
            tile = Image.fromarray(self.renderer.render().copy())
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 480, 27), fill=(20, 25, 32))
            draw.text((10, 2), titles[i], font=self.font, fill="white")
            canvas.paste(tile, ((i % 2) * 480, (i // 2) * 320))
        draw = ImageDraw.Draw(canvas)
        x0, y0 = 480, 320
        last = infos[-1] if infos else {}
        lines = [
            "材料与接触（示例参数，非实测标定）",
            f"状态 {last.get('latch_state', 'locked')} / 夹持 {last.get('grasped', False)}",
            f"净空 {1000 * last.get('clearance_m', 0):.3f} mm  退出 {1000 * last.get('withdrawal_m', 0):.2f} mm",
            f"舌片峰值接触 {last.get('peak_contact_force_n', 0):.3f} N",
            f"失败 {last.get('failure_reason') or last.get('invalid_reason') or '无'}",
        ]
        for i, line in enumerate(lines):
            draw.text((x0 + 10, y0 + 8 + i * 25), line, font=self.font, fill="white")
        rows = infos[-500:]
        bottom = y0 + 295
        left = x0 + 30
        width = 425
        height = 120
        draw.line(
            (left, bottom - height, left, bottom, left + width, bottom),
            fill=(160, 170, 185),
            width=1,
        )
        draw.text(
            (left, bottom - height - 22),
            "红：接触力 0–5 N；蓝：净空 −4–6 mm",
            font=self.font,
            fill=(190, 200, 215),
        )
        for key, color, minimum, span in (
            ("contact_force_n", (255, 100, 90), 0, 5),
            ("clearance_m", (90, 180, 255), -0.004, 0.010),
        ):
            points = [
                (
                    left + i * width / max(1, len(rows) - 1),
                    bottom - np.clip((r.get(key, 0) - minimum) / span, 0, 1) * height,
                )
                for i, r in enumerate(rows)
            ]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
        return canvas

    def close(self):
        self.renderer.close()


class IndustrialReviewer:
    """Enter 新回合、P 暂停、N 单步、B 后退、R 回放；结束不退出。"""

    def __init__(
        self, env, *, seed=0, steps=None, playback_rate=1, save_trace=None, replay=None
    ):
        import tkinter as tk
        from PIL import ImageTk

        self.tk = tk
        self.ImageTk = ImageTk
        self.env = env
        self.seed = seed
        self.steps = steps or env.max_episode_steps
        self.rate = playback_rate
        self.save_path = save_trace
        self.root = tk.Tk()
        self.root.title("IBERO 工业材料审阅 / Enter 运行一次")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status = tk.StringVar(
            value="初始等待：Enter 开始；拖动前三视图旋转，滚轮缩放"
        )
        tk.Label(
            self.root, textvariable=self.status, font=("Microsoft YaHei", 11)
        ).pack()
        buttons = tk.Frame(self.root)
        buttons.pack()
        for title, key in [
            ("运行/重跑 Enter", "Return"),
            ("暂停 P", "p"),
            ("单步 N", "n"),
            ("上一帧 B", "b"),
            ("回放 R", "r"),
        ]:
            tk.Button(buttons, text=title, command=lambda key=key: self.key(key)).pack(
                side="left"
            )
        for key in ("Return", "p", "P", "n", "N", "b", "B", "r", "R"):
            self.root.bind("<" + key + ">", lambda event: self.key(event.keysym))
        self.label.bind("<ButtonPress-1>", self._drag_start)
        self.label.bind("<B1-Motion>", self._drag)
        self.label.bind("<MouseWheel>", self._zoom)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.pending_run = False
        self.resetting = False
        self.finished = False
        self.running = False
        self.history = False
        self.replaying = False
        self.closed = False
        self.single = False
        self.completed_episodes = 0
        self.started = time.monotonic()
        self.last_replay_tick = 0
        env.reset(seed=seed)
        self.policy = LatchReleaseScript()
        self.trace = Trace(env)
        self.trace.append(env.last_info)
        if replay:
            self.trace.load(replay)
            self.history = True
        self._bind_model()
        self.index = 0
        self.draw()
        self.root.after(20, self.tick)

    def _bind_model(self):
        if hasattr(self, "views"):
            self.views.close()
        self.views = IndustrialViews(
            self.env.model, self.env.resolved_config["initialization"]["origin_m"]
        )

    def _reset(self):
        self.env.reset(seed=self.seed)
        self.policy = LatchReleaseScript()
        self.trace = Trace(self.env)
        self.trace.append(self.env.last_info)
        return "reset"

    def _step(self):
        _, _, term, trunc, info = self.env.step(self.policy.action(self.env))
        info["controller_phase"] = self.policy.phase
        self.trace.append(info)
        return (
            "finish"
            if term or trunc or len(self.trace.states) - 1 >= self.steps
            else "step"
        )

    def key(self, key):
        key = key.lower()
        if key == "return":
            self.pending_run = True
            self.running = False
            self.replaying = False
        elif key == "p":
            if self.history:
                self.replaying = not self.replaying
            elif not self.env._done and not self.finished:
                self.running = not self.running
        elif key == "n":
            self.running = False
            self.replaying = False
            if self.history:
                self.index = min(len(self.trace.states) - 1, self.index + 1)
                self.draw()
            elif not self.env._done and not self.finished:
                self.single = True
        elif key == "b":
            self.running = False
            self.replaying = False
            self.history = True
            self.index = max(0, self.index - 1)
            if self.future is None:
                self.draw()
        elif key == "r":
            self.running = False
            self.history = True
            self.replaying = True
            self.index = 0
            if self.future is None:
                self.draw()

    def tick(self):
        if self.closed:
            return
        try:
            if self.future is not None and self.future.done():
                result = self.future.result()
                self.future = None
                if result == "reset":
                    self.resetting = False
                    self.finished = False
                    self._bind_model()
                    self.index = 0
                    self.history = False
                    self.running = True
                    self.started = time.monotonic()
                elif result == "finish":
                    self.finished = True
                    self.running = False
                    self.completed_episodes += 1
                    if self.save_path:
                        self.trace.save(self.save_path)
                if not self.history:
                    self.index = len(self.trace.states) - 1
                self.draw()
            if self.future is None:
                if self.pending_run:
                    self.pending_run = False
                    self.resetting = True
                    self.future = self.pool.submit(self._reset)
                elif (
                    (self.running or self.single)
                    and not self.history
                    and not self.env._done
                    and not self.finished
                ):
                    self.single = False
                    self.future = self.pool.submit(self._step)
                elif (
                    self.replaying
                    and time.monotonic() - self.last_replay_tick
                    >= self.env.control_dt / self.rate
                ):
                    self.last_replay_tick = time.monotonic()
                    self.index = min(self.index + 1, len(self.trace.states) - 1)
                    self.draw()
                    if self.index == len(self.trace.states) - 1:
                        self.replaying = False
        except Exception as error:
            self.running = False
            self.status.set("停止：" + str(error))
            self.future = None
        self.root.after(20, self.tick)

    def draw(self):
        if self.resetting:
            return
        self.views.set_state(self.trace.states[self.index])
        self.last_image = self.views.render(self.trace.infos[: self.index + 1])
        self.photo = self.ImageTk.PhotoImage(self.last_image)
        self.label.configure(image=self.photo)
        info = self.trace.infos[self.index]
        state = (
            "只读回放（Enter 新建回合）"
            if self.history
            else "运行中"
            if self.running
            else "等待 / 暂停 / 已结束"
        )
        factor = float(self.env.data.time) / max(time.monotonic() - self.started, 1e-9)
        self.status.set(
            f"{state} | t={info.get('time_s', 0):.3f} s | 帧 {self.index}/{len(self.trace.states) - 1} | 实际实时因子 {factor:.3f} | 成功 {info.get('success', False)}"
        )

    def _drag_start(self, event):
        self.drag = (event.x, event.y, (event.y // 320) * 2 + event.x // 480)

    def _drag(self, event):
        if not hasattr(self, "drag"):
            return
        x, y, index = self.drag
        if index < 3:
            cam = self.views.cameras[index]
            cam.azimuth += (event.x - x) * 0.4
            cam.elevation = np.clip(cam.elevation + (event.y - y) * 0.3, -89, -2)
            self.drag = (event.x, event.y, index)
            self.draw()

    def _zoom(self, event):
        index = (event.y // 320) * 2 + event.x // 480
        if index < 3:
            self.views.cameras[index].distance *= 0.9 if event.delta > 0 else 1.1
            self.draw()

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.views.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.trace.infos[-1]
