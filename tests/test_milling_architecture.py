"""具体加工编译与运行时边界；模型产物没有虚假的线束参数。"""

from pathlib import Path
import mujoco
import numpy as np
import pytest

from ibero.benches.milling import MillingFixture, add_spindle
from ibero.core.milling_builder import CompiledMilling
from ibero.core.milling_runtime import MillingRuntime
from ibero.core.scene_compiler import SceneCompiler
from ibero.core.scene_loader import SceneLoader
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.robots.spindle import add_spindle as tool_builder

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["milling_bench", "plate_milling"])
def test_compiler_owns_concrete_stock_model(kind):
    scene = SceneLoader().validate(ROOT / "scenes" / kind)
    result = SceneCompiler().compile(scene)
    assert isinstance(result, CompiledMilling)
    assert result.scene is scene
    assert not hasattr(result, "cable_parameters")
    assert result.model.geom("mill_blade").id >= 0
    runtime = MillingRuntime()
    runtime._initialize_runtime(result)
    assert runtime.stock is result.stock and runtime.binding is result.binding
    assert runtime.model is result.model and isinstance(runtime.data, mujoco.MjData)
    # 初始化共同状态不触发子类 reset / 策略，也不对材料删格。
    assert not hasattr(runtime, "last_info")
    assert result.stock.version == 0
    np.testing.assert_array_equal(
        result.model.opt.gravity, scene.config["physics"]["gravity_m_s2"]
    )


def test_robot_no_longer_inherits_fixture_but_shares_step_transaction(monkeypatch):
    assert not issubclass(PlateMillingEnv, MillingFixture)
    assert MillingFixture.step is MillingRuntime.step
    original = MillingRuntime.step
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(type(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MillingRuntime, "step", counted)
    env = PlateMillingEnv()
    env.step(env.start, 0, substeps=1)
    assert calls == [PlateMillingEnv]
    assert add_spindle is tool_builder  # 原 Python 导入兼容，实体工具只有一份实现。


def test_scene_factory_compiles_once(monkeypatch):
    original = SceneCompiler.compile
    calls = []

    def counted(self, scene):
        calls.append(scene.config["kind"])
        return original(self, scene)

    monkeypatch.setattr(SceneCompiler, "compile", counted)
    env = MillingFixture.from_scene(ROOT / "scenes/milling_bench")
    assert calls == ["milling_bench"]
    assert env.data.time == 0 and env.seed_value == 0
    assert env.scene.config["kind"] == "milling_bench"
