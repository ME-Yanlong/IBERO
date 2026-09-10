"""几何测试不声称已实现加工载荷；解析参考独立于去除算法。"""

from dataclasses import replace
import math
import numpy as np
import pytest
from ibero.materials.parameters import StockParameters
from ibero.materials.stock import VoxelStock
from ibero.processes.tools import (
    EndMillGeometry,
    ToolPose,
    swept_cells,
    quaternion_matrix,
)


def test_boundary_cell_volume_coordinate_index_and_capacity():
    p = StockParameters(size_m=(0.0055, 0.0045, 0.0023))
    rotation = quaternion_matrix([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    stock = VoxelStock(p, 0.001, origin=(0.2, -0.1, 0.3), rotation=rotation)
    assert stock.shape == (6, 5, 3)
    assert stock.volume_m3 == pytest.approx(math.prod(p.size_m))
    assert stock.mass_kg == pytest.approx(math.prod(p.size_m) * p.density_kg_m3)
    world = stock.local_to_world(stock.centers)
    np.testing.assert_allclose(stock.world_to_local(world), stock.centers, atol=1e-15)
    np.testing.assert_array_equal(stock.point_indices(world), np.arange(len(world)))
    assert not stock.occupied_at(stock.local_to_world([0.006, 0, 0]))
    with pytest.raises(ValueError, match="capacity"):
        VoxelStock(p, 0.00001, max_cells=100)


def test_idempotent_monotonic_atomic_event_and_replay():
    stock = VoxelStock(StockParameters(), 0.002)
    volume = stock.volume_m3
    event = stock.prepare_removal([1, 2, 2, 3], "a")
    assert stock.version == 0 and stock.volume_m3 == volume
    assert stock.commit(event) > 0
    after = stock.volume_m3
    assert stock.commit(event) == 0 and stock.volume_m3 == after
    assert stock.prepare_removal([3, 2, 1], "a") == event
    with pytest.raises(ValueError, match="reused"):
        stock.prepare_removal([5], "a")
    empty = stock.prepare_removal([1, 2, 3], "empty-cut")
    assert stock.commit(empty) == 0 and stock.version == 1
    stale = stock.prepare_removal([4], "stale")
    stock.commit(stock.prepare_removal([5], "new"))
    before = stock.state_hash()
    with pytest.raises(ValueError, match="Stale"):
        stock.commit(stale)
    assert stock.state_hash() == before
    events = list(stock.events)
    clone = VoxelStock(StockParameters(), 0.002)
    for e in events:
        clone.commit(e)
    assert clone.state_hash() == stock.state_hash()
    stock.reset()
    assert stock.volume_m3 == volume and not stock.events and stock.version == 0


def test_corrupt_ledger_and_bad_indices_rejected_without_mutation():
    stock = VoxelStock(StockParameters(), 0.002)
    event = stock.prepare_removal([0], "a")
    before = stock.state_hash()
    with pytest.raises(ValueError, match="ledger"):
        stock.commit(replace(event, mass_kg=10))
    for bad in ([-1], [len(stock.centers)], [1.2], [True]):
        with pytest.raises(ValueError):
            stock.prepare_removal(bad, "bad")
    assert stock.state_hash() == before


def test_query_extreme_points_and_empty_multidimensional_ids_are_bounded():
    stock = VoxelStock(StockParameters(), 0.002)
    with np.errstate(all="raise"):
        assert stock.point_indices([1e300, 0, 0]) == -1
    for value in (1, [], [1, 2], [[1, 2]]):
        with pytest.raises(ValueError):
            stock.point_indices(value)
    with pytest.raises(ValueError):
        stock.prepare_removal(np.empty((0, 3), dtype=int), "malformed-empty")
    event = stock.prepare_removal([0], "oversized-id")
    with pytest.raises(ValueError):
        stock.validate_event(replace(event, removed_ids=(10**100,)))


def test_pose_rotation_cache_is_immutable():
    pose = ToolPose((0, 0, 0))
    assert pose.rotation is pose.rotation
    with pytest.raises(ValueError):
        pose.rotation[0, 0] = 2


def test_cylinder_shank_does_not_cut_and_fast_path_has_no_gap():
    tool = EndMillGeometry(0.004, 0.004, 0.01, 0.02)
    pose = ToolPose((0, 0, 0))
    np.testing.assert_array_equal(
        tool.contains_cutting_points(
            [[0, 0, 0.002], [0, 0, 0.006], [0.006, 0, 0.002]], pose
        ),
        [True, False, False],
    )
    stock = VoxelStock(StockParameters(), 0.001)
    ids = swept_cells(stock, tool, ToolPose((-0.02, 0, 0)), ToolPose((0.02, 0, 0)))
    event = stock.prepare_removal(ids, "geometry-only")
    stock.commit(event)
    xs = np.linspace(-0.018, 0.018, 73)
    assert not stock.occupied_at(
        np.column_stack((xs, np.zeros(73), np.ones(73) * 0.002))
    ).any()


def test_world_rotation_does_not_change_removed_local_cells():
    p = StockParameters(size_m=(0.03, 0.02, 0.006))
    q = (math.cos(0.3), 0, math.sin(0.3), 0)
    rotation = quaternion_matrix(q)
    plain = VoxelStock(p, 0.001)
    turned = VoxelStock(p, 0.001, origin=(0.3, -0.2, 0.4), rotation=rotation)
    tool = EndMillGeometry(0.0043, 0.01, 0.006, 0.03)
    start, end = (-0.008, 0, -0.0031), (0.008, 0, -0.0031)
    a = swept_cells(plain, tool, ToolPose(start), ToolPose(end))
    b = swept_cells(
        turned,
        tool,
        ToolPose(tuple(turned.local_to_world(start)), q),
        ToolPose(tuple(turned.local_to_world(end)), q),
    )
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cell", [0.002, 0.001, 0.0005])
def test_analytical_through_hole_volume(cell):
    stock = VoxelStock(StockParameters(), cell)
    radius = 0.008
    tool = EndMillGeometry(radius, 0.02, radius, 0.03)
    pose = ToolPose((0, 0, -0.005))
    event = stock.prepare_removal(swept_cells(stock, tool, pose, pose), "hole")
    stock.commit(event)
    expected = math.pi * radius**2 * 0.01
    assert abs(event.volume_m3 / expected - 1) <= 0.05
