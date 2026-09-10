"""P4 配置校验与独立形状判定，不能用完成轨迹或相同体积代替真孔。"""

from pathlib import Path
import copy
from dataclasses import replace
import pytest
from ibero.core.scene_loader import SceneLoader
from ibero.core.milling_config import validate_milling_bench
from ibero.benches.milling import MillingFixture
from ibero.materials.parameters import StockParameters
from ibero.materials.stock import VoxelStock
from ibero.processes.shape_check import MachiningTarget, inspect_shape

ROOT = Path(__file__).resolve().parents[1]


def test_recipe_loads_strict_material_tool_and_process():
    scene = SceneLoader().validate(ROOT / "scenes/milling_bench")
    env = MillingFixture.from_scene(scene.root)
    assert env.process.nphi == 128 and env.model.nv == 4
    assert env.stock.shape == (24, 20, 4)
    assert env.manifest()["coefficients"]["source"] == "provisional"
    first = env.manifest()["scene_hash"]
    env.model.opt.timestep /= 2
    assert env.manifest()["scene_hash"] != first


@pytest.mark.parametrize(
    "case", ["extra", "missing", "grade", "capacity", "gravity", "teeth", "rpm", "gate"]
)
def test_recipe_rejects_unsupported_or_inconsistent_settings(case):
    scene = SceneLoader().validate(ROOT / "scenes/milling_bench")
    cfg, constraints = copy.deepcopy(scene.config), copy.deepcopy(scene.constraints)
    if case == "extra":
        cfg["tool"]["magic_cutting"] = True
    elif case == "missing":
        del cfg["process"]["coefficients"]["face_axial_pa"]
    elif case == "grade":
        cfg["materials"]["stock"]["material_grade"] = "aluminum"
    elif case == "capacity":
        cfg["numerics"]["max_cells"] = 1
    elif case == "gravity":
        cfg["physics"]["gravity_m_s2"] = [0, 0, -9.81]
    elif case == "teeth":
        cfg["process"]["limits"]["teeth"] = True
    elif case == "rpm":
        cfg["control"]["rpm"] = 0
    else:
        constraints["task"]["volume_error_fraction"] = 0.5
    with pytest.raises(ValueError):
        validate_milling_bench(cfg, constraints)


@pytest.mark.parametrize(
    "shape,a,b,depth",
    [
        ("through_hole", 0, 0, 0.004),
        ("slot", 0.004, 0, 0.002),
        ("pocket", 0.004, 0.002, 0.002),
    ],
)
def test_independent_shape_accepts_reference_and_rejects_under_over_offset(
    shape, a, b, depth
):
    target = MachiningTarget(shape, (0, 0), (a, b), 0.004, depth)
    for fault in (None, "under", "over", "offset"):
        stock = VoxelStock(StockParameters(size_m=(0.024, 0.020, 0.004)), 0.001)
        actual = target
        if fault == "offset":
            actual = replace(target, center_xy_m=(0.003, 0))
        elif fault == "over":
            actual = replace(target, corner_radius_m=0.006)
        elif fault == "under":
            actual = replace(target, depth_m=depth / 2)
        mask = (actual.planar_sdf(stock.centers[:, :2]) <= 0) & (
            stock.centers[:, 2] >= 0.002 - actual.depth_m
        )
        import numpy as np

        stock.commit(
            stock.prepare_removal(np.flatnonzero(mask), "synthetic-independent")
        )
        report = inspect_shape(stock, target)
        assert report["passed"] is (fault is None), report


def test_small_pocket_perimeter_covers_target_without_dense_repeated_paths():
    from ibero.control.milling_bench import MillingBenchScript, bench_target
    from ibero.processes.tools import swept_cells, ToolPose

    env = MillingFixture.from_scene(ROOT / "scenes/milling_bench")
    target = bench_target("pocket")
    policy = MillingBenchScript(env, target)
    assert sum(phase == "cut" for _, phase in policy.waypoints) == 4
    start = env.start
    for i, (point, _) in enumerate(policy.waypoints):
        # 仅用作路径覆盖的几何单测，不是过程级切削成功。
        ids = swept_cells(env.stock, env.tool, ToolPose(start), ToolPose(point))
        env.stock.commit(env.stock.prepare_removal(ids, f"path-coverage-{i}"))
        start = point
    assert inspect_shape(env.stock, target)["passed"]


def test_bench_reads_stricter_shape_criteria_and_inspector_rejects_relaxation():
    import numpy as np
    from ibero.control.milling_bench import bench_target

    env = MillingFixture.from_scene(ROOT / "scenes/milling_bench")
    target = bench_target("through_hole")
    ids = np.flatnonzero(target.planar_sdf(env.stock.centers[:, :2]) <= 0)
    env.binding.commit(env.stock.prepare_removal(ids, "synthetic-criteria"), env.data)
    assert env.inspect_target(target)["passed"]
    env.scene.constraints["task"]["volume_error_fraction"] = 0.01
    report = env.inspect_target(target)
    assert not report["passed"] and report["volume_error_limit_fraction"] == 0.01
    for options in (
        {"volume_error_fraction": True},
        {"volume_error_fraction": float("nan")},
        {"volume_error_fraction": 0.5},
        {"boundary_error_cells": 3},
    ):
        with pytest.raises(ValueError):
            inspect_shape(env.stock, target, **options)
