"""可变材料显示必须隔离模型、碰撞和账本；此处几何提交仅为显示单测夹具。"""

import numpy as np
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.core.stock_trace import StockTrace
from ibero.milling_review import MillingViews


def test_viewer_history_and_visual_chips_never_change_live_material_or_physics():
    env = PlateMillingEnv()
    trace = StockTrace(env)
    trace.append(env.last_info)
    event = env.stock.prepare_removal(np.arange(12), "visual-only-test-fixture")
    env.binding.commit(event, env.data)
    trace.append(env.last_info)
    q = env.data.qpos.copy()
    mask = env.model.geom_contype.copy()
    rgba = env.model.geom_rgba.copy()
    current_hash = env.stock.state_hash()
    views = MillingViews(env)
    try:
        views.set_frame(trace, 0)
        assert views.stock.version == 0 and env.stock.version == 1
        views.render(trace.infos[:1])
        views.set_frame(trace, 1)
        assert (
            views.stock.state_hash() == current_hash and len(views.chip_positions) > 0
        )
        visible = views.render(trace.infos)
        views.visual_chips = False
        hidden = views.render(trace.infos)
        assert visible.size == hidden.size == (960, 640)
        np.testing.assert_array_equal(env.data.qpos, q)
        np.testing.assert_array_equal(env.model.geom_contype, mask)
        np.testing.assert_array_equal(env.model.geom_rgba, rgba)
        assert env.stock.state_hash() == current_hash
        env.binding.ensure_consistent()
    finally:
        views.close()
