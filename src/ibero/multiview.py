"""四视角人工审阅面板：不改变任务物理，只读取已编译场景的状态。"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

if TYPE_CHECKING:
    from ibero.envs.cable_handover import CableHandoverEnv


@dataclass(frozen=True)
class _CameraPreset:
    """一个自由相机相对于本轮 task state 的审阅构图参数。"""

    label: str
    distance: float
    azimuth: float
    elevation: float


class CableHandoverMultiView:
    """把总览、两侧工具和端头/目标区渲染成一个 2×2 RGB 图像。"""

    # 分辨率刻意限制为 320×240/格：四路离屏渲染在普通开发机上仍可实时
    # 刷新，同时足够看清 2F-85 指垫与刚性端头的接触关系。
    tile_width = 320
    tile_height = 240
    _presets = (
        _CameraPreset("总览", 0.72, -135.0, -15.0),
        _CameraPreset("左末端", 0.28, -108.0, -12.0),
        _CameraPreset("右末端", 0.28, 108.0, -12.0),
        _CameraPreset("端头与目标区", 0.35, 180.0, -70.0),
    )

    def __init__(self, env: "CableHandoverEnv") -> None:
        self._env = env
        self._renderer = mujoco.Renderer(
            env.model, height=self.tile_height, width=self.tile_width
        )
        self._cameras = [self._new_camera() for _ in self._presets]
        self.options = mujoco.MjvOption()
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.options.frame = mujoco.mjtFrame.mjFRAME_SITE
        env.model.vis.scale.framelength = 0.035
        env.model.vis.scale.framewidth = 0.002
        env.model.vis.map.force = 0.0003
        env.model.vis.scale.forcewidth = 0.012
        env.model.vis.scale.contactwidth = 0.025
        env.model.vis.scale.contactheight = 0.01

    @staticmethod
    def _new_camera() -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        return camera

    def _update_cameras(self) -> None:
        handles = self._env.handles
        data = self._env.data
        # 近景相机以实际 site/body 为 lookat；所以每次接触、放置或随机化后
        # 都会自动跟随，而不是依赖某一个 seed 下恰好可见的硬编码世界坐标。
        lookats = (
            (
                data.site_xpos[handles.left_pinch_site_id]
                + data.site_xpos[handles.right_pinch_site_id]
            )
            / 2,
            data.site_xpos[handles.left_pinch_site_id],
            data.site_xpos[handles.right_pinch_site_id],
            data.xpos[handles.cable.terminal_body_id],
        )
        for camera, preset, lookat in zip(
            self._cameras, self._presets, lookats, strict=True
        ):
            camera.lookat = np.asarray(lookat, dtype=np.float64)
            camera.distance = preset.distance
            camera.azimuth = preset.azimuth
            camera.elevation = preset.elevation

    def render(self) -> np.ndarray:
        """Return a stable 640×480 2×2 review image for the current sim state."""

        self._update_cameras()
        images: list[np.ndarray] = []
        for camera in self._cameras:
            self._renderer.update_scene(
                self._env.data, camera=camera, scene_option=self.options
            )
            images.append(self._renderer.render().copy())
        return np.concatenate(
            (
                np.concatenate((images[0], images[1]), axis=1),
                np.concatenate((images[2], images[3]), axis=1),
            ),
            axis=0,
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(preset.label for preset in self._presets)

    def close(self) -> None:
        self._renderer.close()


class MultiViewDashboard:
    """可选 Tk 审阅窗口；Enter 和按钮均只请求“执行下一轮”。"""

    def __init__(
        self, env: "CableHandoverEnv", run_requested: threading.Event, controls=None
    ) -> None:
        # tkinter/Pillow 均只服务 demo 的人工审阅，不能成为 Core 物理和 CI
        # 的强依赖。调用方会在导入/建窗失败时自动降级到 MuJoCo viewer。
        import tkinter as tk
        from PIL import Image, ImageTk

        self._tk = tk
        self._image_module = Image
        self._image_tk = ImageTk
        self._run_requested = run_requested
        self._open = True
        self._last_refresh_s = -np.inf
        self._last_refresh_wall = -np.inf
        self._root = tk.Tk()
        try:
            self._views = CableHandoverMultiView(env)
        except Exception:
            self._root.destroy()
            raise
        self._root.title("IBERO Core-0.1 | 四视角审阅")
        self._root.bind("<Return>", self._request_run)
        if controls:
            for key, code in [("p", 80), ("n", 78), ("r", 82), ("b", 66)]:
                self._root.bind(key, lambda event, code=code: controls.key(code))
        self._root.protocol("WM_DELETE_WINDOW", self._close_window)

        self._image_label = tk.Label(self._root)
        self._image_label.pack(padx=8, pady=(8, 2))
        tk.Label(
            self._root,
            text="左上：总览   右上：左末端   左下：右末端   右下：端头 / 目标区",
        ).pack(padx=8, pady=2)
        self._status_label = tk.Label(self._root, justify="left", anchor="w")
        self._status_label.pack(fill="x", padx=8, pady=4)
        self._curve = tk.Canvas(
            self._root, width=640, height=120, bg="#17212a", highlightthickness=0
        )
        self._curve.pack(padx=8, pady=4)
        self._history = deque(maxlen=500)
        self._env = env
        tk.Button(self._root, text="运行一轮（Enter）", command=self._request_run).pack(
            padx=8, pady=(0, 8)
        )
        if controls:
            buttons = tk.Frame(self._root)
            buttons.pack()
            for label, code in [
                ("暂停 / 继续 P", 80),
                ("单步 N", 78),
                ("回放 R", 82),
                ("上一帧 B", 66),
            ]:
                tk.Button(
                    buttons, text=label, command=lambda code=code: controls.key(code)
                ).pack(side="left")
        self._photo: Any | None = None

    def _request_run(self, _event: Any | None = None) -> None:
        self._run_requested.set()

    def _close_window(self) -> None:
        self._open = False
        self._root.destroy()

    def present(
        self, info: dict[str, Any], status: str, *, force: bool = False
    ) -> bool:
        """处理窗口事件并按最多 10 Hz 更新图像和审计字段。"""

        if not self._open:
            return False
        try:
            now = float(self._views._env.data.time)
            # 物理时间 reset 后会归零，故 force 用于每个 reset/回合末态；其余
            # 情况限制 10 Hz，避免四路离屏渲染反过来拖慢接触仿真。
            # 用墙钟限制 10 Hz；加速播放时不能按仿真时钟无限提高渲染频率。
            # reset/倒带和末帧强制刷新，暂停期间 UI 仍保持响应。
            wall = time.perf_counter()
            important = force and (
                now <= self._last_refresh_s or info.get("is_success", False)
            )
            if important or wall - self._last_refresh_wall >= 0.10:
                image = self._image_module.fromarray(self._views.render())
                self._photo = self._image_tk.PhotoImage(image=image)
                self._image_label.configure(image=self._photo)
                self._last_refresh_s = now
                self._last_refresh_wall = time.perf_counter()
            audit = info.get("audit", {})
            metrics = info.get("metrics", {})
            if self._history and now < self._history[-1][0]:
                self._history.clear()
            if not self._history or now > self._history[-1][0]:
                self._history.append((now, float(audit.get("cable_tension_n", 0))))
            cfg = self._env.scene.config.get("control", {})
            self._curve.delete("all")
            ceiling = (
                max(
                    1.0,
                    cfg.get("warning_tension_n", 0),
                    max((v for _, v in self._history), default=0),
                )
                * 1.1
            )
            if len(self._history) > 1:
                start = self._history[0][0]
                span = max(1.0, now - start)
                points = [
                    coordinate
                    for t, v in self._history
                    for coordinate in (
                        30 + (t - start) / span * 590,
                        100 - v / ceiling * 80,
                    )
                ]
                self._curve.create_line(*points, fill="#45d9ea", width=2)
            for field, color in [
                ("target_tension_n", "#4bea81"),
                ("warning_tension_n", "#ffbd4b"),
            ]:
                if field in cfg:
                    y = 100 - cfg[field] / ceiling * 80
                    self._curve.create_line(30, y, 620, y, fill=color, dash=(4, 3))
            self._curve.create_text(
                8,
                8,
                anchor="nw",
                fill="white",
                text=f"张力 N | 时间 {now:.2f} s | 绿：目标 / 黄：卸载阈值",
            )
            self._status_label.configure(
                text=(
                    f"状态：{status}    阶段：{info.get('stage', 'unknown')}    "
                    f"成功：{info.get('is_success', False)}\n"
                    f"夹持：左={metrics.get('left_grasp', False)}  "
                    f"右={metrics.get('right_grasp', False)}    "
                    f"线束拉力：{float(audit.get('cable_tension_n', 0.0)):.2f} N  "
                    f"峰值：{float(audit.get('peak_cable_tension_n', 0.0)):.2f} N\n"
                    f"安全：超拉={audit.get('damage_flags', {}).get('cable_over_tension', False)}  "
                    f"碰撞={audit.get('self_collision_flags', {}).get('left_right', False)}  "
                    f"弯曲报警={audit.get('material', {}).get('bend_alarm', False)}  "
                    f"控制={info.get('controller_phase', '--')}"
                )
            )
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._open = False
        return self._open

    def close(self) -> None:
        self._views.close()
        if self._open:
            self._close_window()


def try_create_dashboard(
    env: "CableHandoverEnv", run_requested: threading.Event, controls=None
) -> MultiViewDashboard | None:
    """Create the optional dashboard, returning ``None`` on unavailable desktop GUI."""

    try:
        return MultiViewDashboard(env, run_requested, controls)
    except Exception as error:  # GUI availability varies across CI/remote desktops.
        print(f"Four-view dashboard unavailable; using MuJoCo viewer only: {error}")
        return None
