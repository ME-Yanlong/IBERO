"""S1：新台架参数、旧场景兼容、实际材料力叠加及施力生命周期。"""

import copy
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from ibero.core.industrial_config import validate_industrial
from ibero.core.loads import AppliedLoads
from ibero.core.scene_compiler import SceneCompiler
from ibero.core.scene_loader import SceneLoader
from ibero.materials.parameters import BeamParameters

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name", ["cable_handover", "cable_stretch", "latch_bench", "stock_bench"]
)
def test_recipes_validate(name):
    assert SceneLoader().validate(ROOT / "scenes" / name).scene_hash


@pytest.mark.parametrize(
    "change",
    [
        {"young_pa": float("nan")},
        {"density_kg_m3": True},
        {"length_m": -1},
        {"parameter_source": ""},
        {"thickness_m": 0.1},
        {"length_m": 1e200, "thickness_m": 1e190},
        {"young_pa": 1e-320},
    ],
)
def test_invalid_beam(change):
    with pytest.raises(ValueError):
        BeamParameters(**change)


def test_strict_schema_and_empty_fixture():
    scene = SceneLoader().validate(ROOT / "scenes/latch_bench")
    cfg = copy.deepcopy(scene.config)
    cfg["materials"]["beam"]["fracture"] = True
    with pytest.raises(ValueError):
        validate_industrial(cfg, scene.constraints)
    cfg = copy.deepcopy(scene.config)
    cfg.update(kind="empty_bench", materials={}, mechanism={}, numerics={})
    validate_industrial(cfg, scene.constraints)
    compiled = SceneCompiler().compile(replace(scene, config=cfg))
    assert compiled.model.nv == 0
    assert not hasattr(compiled, "cable_parameters")


def test_real_cable_force_accumulation_and_clean_substeps():
    from ibero.materials.cable import CableParameters, _flex_subspec
    from ibero.materials.mechanics import CableMechanics

    scene = SceneLoader().validate(ROOT / "scenes/cable_stretch")
    params = CableParameters.from_scene(scene.config["materials"]["cable"])
    model = _flex_subspec(params, np.array([1.0, 0, 0])).compile()
    data = mujoco.MjData(model)
    mechanics = CableMechanics(model, params)
    mechanics.apply(data)
    reference = data.qfrc_applied.copy()
    loads = AppliedLoads(model)
    loads.begin()
    with pytest.raises(RuntimeError):
        loads.begin()
    with pytest.raises(ValueError):
        loads.add_generalized([-1], [1])
    mechanics.apply(data, loads)
    loads.add_generalized([0], [2.0])
    loads.commit(data)
    reference[0] += 2
    np.testing.assert_allclose(data.qfrc_applied, reference, atol=1e-12)
    with pytest.raises(RuntimeError):
        loads.commit(data)
    data.xfrc_applied[1, 0] = 3
    loads.begin()
    loads.commit(data)
    assert not data.qfrc_applied.any()
    assert data.xfrc_applied[1, 0] == 3  # 独立校准外载不受汇总器擅自清除。


def test_wrench_application_includes_moment_arm():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    loads = AppliedLoads(model)
    loads.begin()
    loads.add_wrench(data, 1, [0, 2, 0], [0, 0, 0.3], [0.1, 0, 0])
    loads.commit(data)
    np.testing.assert_allclose(data.qfrc_applied, [0, 2, 0, 0, 0, 0.5], atol=1e-12)
