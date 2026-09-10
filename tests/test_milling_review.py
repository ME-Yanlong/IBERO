"""可变材料显示必须隔离模型、碰撞和账本；此处几何提交仅为显示单测夹具。"""

import numpy as np
import mujoco
import pytest
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
    groups = env.model.geom_group.copy()
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
        assert not views.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]
        assert not views.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]
        # 最后一格原生渲染视角只保留实际材料，刀具隐藏是显示过滤而非几何删除。
        assert views.top_option.geomgroup.tolist() == [0, 0, 0, 0, 0, 1]
        assert np.all(views.model.geom_group[env.binding.geom_ids] == 5)
        assert visible.size == hidden.size == (960, 640)
        assert np.any(np.asarray(visible) != np.asarray(hidden))
        views.render_quality = "quality"
        views.render(trace.infos)
        assert views.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]
        assert views.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]
        np.testing.assert_array_equal(env.data.qpos, q)
        np.testing.assert_array_equal(env.model.geom_contype, mask)
        np.testing.assert_array_equal(env.model.geom_rgba, rgba)
        np.testing.assert_array_equal(env.model.geom_group, groups)
        assert env.stock.state_hash() == current_hash
        env.binding.ensure_consistent()
    finally:
        views.close()


def test_invalid_quality_rejected_before_allocating_viewer():
    with pytest.raises(ValueError, match="render_quality"):
        MillingViews(None, render_quality="pretend_physics_fast_forward")
