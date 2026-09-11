"""桌面/线程/渲染异常的资源顺序；替身测试不冒充真实 GUI 验收。"""

from types import SimpleNamespace
import tkinter as tk
import pytest
import ibero.industrial_review as module


class DummyTrace:
    def __init__(self, env):
        self.infos = []

    def append(self, info):
        self.infos.append(info)

    def load(self, path):
        raise ValueError("trace identity mismatch")


def setup_fakes(monkeypatch, *, reset_error=False, draw_error=False):
    calls = []

    class Widget:
        def __init__(self, *args, **kwargs):
            pass

        def pack(self, **kwargs):
            pass

        def bind(self, *args):
            pass

    class Root(Widget):
        def __init__(self):
            calls.append("window")

        def title(self, *args):
            pass

        def protocol(self, *args):
            pass

        def after(self, *args):
            pass

        def destroy(self):
            calls.append("destroy")

        def mainloop(self):
            raise RuntimeError("loop failed")

    class Pool:
        def __init__(self, **kwargs):
            calls.append("pool")

        def shutdown(self, *, wait, cancel_futures):
            assert wait and cancel_futures
            calls.append("joined")

    class Views:
        def close(self):
            calls.append("renderer_closed")

    class Reviewer(module.IndustrialReviewer):
        def _make_policy(self):
            return None

        def _make_trace(self):
            return DummyTrace(self.env)

        def _bind_model(self):
            self.views = Views()

        def draw(self):
            if draw_error:
                raise RuntimeError("draw failed")

    def reset(**kwargs):
        calls.append("reset")
        if reset_error:
            raise ValueError("bad seed")

    monkeypatch.setattr(tk, "Tk", Root)
    for name in ("Label", "Frame", "Button", "StringVar"):
        monkeypatch.setattr(tk, name, Widget)
    monkeypatch.setattr(module, "ThreadPoolExecutor", Pool)
    env = SimpleNamespace(max_episode_steps=10, reset=reset, last_info={})
    return Reviewer, env, calls


@pytest.mark.parametrize("bad_seed", [True, False])
def test_input_failure_before_desktop_or_pool(monkeypatch, bad_seed):
    reviewer, env, calls = setup_fakes(monkeypatch, reset_error=bad_seed)
    with pytest.raises(ValueError, match="bad seed|identity mismatch"):
        reviewer(env, replay=None if bad_seed else "other-source.npz")
    assert calls == ["reset"]


def test_draw_failure_releases_resources_in_order(monkeypatch):
    reviewer, env, calls = setup_fakes(monkeypatch, draw_error=True)
    with pytest.raises(RuntimeError, match="draw failed"):
        reviewer(env)
    assert calls[-3:] == ["joined", "renderer_closed", "destroy"]


def test_mainloop_failure_and_double_close(monkeypatch):
    reviewer, env, calls = setup_fakes(monkeypatch)
    app = reviewer(env)
    with pytest.raises(RuntimeError, match="loop failed"):
        app.run()
    app.close()
    assert calls.count("joined") == calls.count("destroy") == 1
    assert calls[-3:] == ["joined", "renderer_closed", "destroy"]


def test_renderer_close_error_still_destroys_window(monkeypatch):
    reviewer, env, calls = setup_fakes(monkeypatch)
    app = reviewer(env)

    def fail_close():
        raise RuntimeError("GL close failed")

    app.views.close = fail_close
    with pytest.raises(RuntimeError, match="GL close failed"):
        app.close()
    assert calls[-2:] == ["joined", "destroy"]
