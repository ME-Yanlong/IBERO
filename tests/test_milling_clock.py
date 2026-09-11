"""场景时长必须由共同物理步进约束，不能只依靠某个 demo 的外层循环。"""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import mujoco
import numpy as np
import pytest
import yaml

from ibero.benches.milling import MillingFixture
from ibero.envs.plate_milling import PlateMillingEnv
from ibero.review import Trace


def make_recipe(tmp_path, kind, limit):
    root = Path(__file__).resolve().parents[1] / "scenes" / kind
    scene = tmp_path / kind
    shutil.copytree(root, scene)
    path = scene / "constraints.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["task"]["max_seconds"] = limit
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return scene


@pytest.mark.parametrize("kind", ["milling_bench", "plate_milling"])
@pytest.mark.parametrize(
    "limit,expected_steps", [(0.005, 50), (0.00005, 0), (0.00505, 50)]
)
def test_scene_horizon_bounds_actual_physics_and_latches_timeout(
    tmp_path, kind, limit, expected_steps
):
    scene = make_recipe(tmp_path, kind, limit)
    cls = MillingFixture if kind == "milling_bench" else PlateMillingEnv
    env = cls.from_scene(scene)
    original = np.empty(mujoco.mj_stateSize(env.model, Trace.spec))
    mujoco.mj_getState(env.model, env.data, original, Trace.spec)
    info = env.step(env.start, 6000, substeps=100)
    assert env.data.time <= limit + 1e-12
    assert info["integrated_substeps"] == expected_steps
    assert info["invalid_reason"] == "episode_time_limit" and env._done
    assert env.stock.version == 0
    if expected_steps == 0:
        current = np.empty_like(original)
        mujoco.mj_getState(env.model, env.data, current, Trace.spec)
        np.testing.assert_array_equal(current, original)
        assert not info["last_substep_load_applied"]
        assert info["peak_applied_cutting_force_step_n"] == 0
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.start, 6000)
    env.reset()
    assert not env._done and env.process.invalid_reason is None and env.data.time == 0


def test_headless_demo_cannot_override_scene_horizon_with_large_step_budget(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scene = make_recipe(tmp_path, "plate_milling", 0.005)
    manifest = tmp_path / "terminal.json"
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "ibero.demo",
            "--env",
            "plate_milling",
            "--scene",
            str(scene),
            "--steps",
            "1000000",
            "--save-manifest",
            str(manifest),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 1, result.stderr
    info = json.loads(manifest.read_text(encoding="utf-8"))["final_info"]
    assert info["time_s"] <= 0.005 + 1e-12
    assert info["failure_reason"] == "episode_time_limit" and not info["success"]
