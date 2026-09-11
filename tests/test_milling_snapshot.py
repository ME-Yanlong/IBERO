"""跨源码比较用数值摘要，不修改正式回放身份校验。"""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np


def test_model_digest_is_readonly_and_detects_capability_change():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "runtime_snapshot_test", root / "scripts/snapshot_milling_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="q"/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody>'
        '<actuator><motor joint="q" forcelimited="true" forcerange="-1 1"/>'
        '</actuator></mujoco>'
    )
    before = model.actuator_forcerange.copy()
    first = module.model_arrays(model)
    assert first == module.model_arrays(model)
    np.testing.assert_array_equal(before, model.actuator_forcerange)
    model.actuator_forcerange[0, 1] = 2
    changed = module.model_arrays(model)
    assert changed["actuator_forcerange"] != first["actuator_forcerange"]
    assert changed["body_mass"] == first["body_mass"]
