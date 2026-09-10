"""S5 独立几何验收：不启用机器人、不提供切削授权、不生成伪物理力。"""

import math
import numpy as np
from PIL import Image, ImageDraw
from ibero.materials.parameters import StockParameters
from ibero.materials.stock import VoxelStock
from ibero.processes.tools import EndMillGeometry, ToolPose, swept_cells


def benchmark_shape(name, cell_size_m):
    """解析二维截面×独立深度，与圆柱扫掠算法分开实现参考。"""
    stock = VoxelStock(StockParameters(), cell_size_m)
    depth = 0.01 if name == "through_hole" else 0.004
    z = 0.005 - depth
    paths = []
    if name == "through_hole":
        radius, a, b = 0.008, 0, 0
        paths = [((0, 0, z), (0, 0, z))]
        area = math.pi * radius**2
    elif name == "slot":
        radius, a, b = 0.008, 0.012, 0
        paths = [((-a, 0, z), (a, 0, z))]
        area = 4 * radius * a + math.pi * radius**2
    elif name == "pocket":
        radius, a, b = 0.004, 0.01, 0.006
        paths = [
            ((-a, y, z), (a, y, z))
            for y in np.linspace(-b, b, math.ceil(2 * b / (cell_size_m / 2)) + 1)
        ]
        area = 4 * a * b + 4 * radius * (a + b) + math.pi * radius**2
    else:
        raise ValueError("Unknown stock geometry benchmark")
    tool = EndMillGeometry(radius, 0.02, radius, 0.03)
    for index, (start, end) in enumerate(paths):
        event = stock.prepare_removal(
            swept_cells(stock, tool, ToolPose(start), ToolPose(end)),
            f"geometry-only-{index}",
        )
        stock.commit(event)
        assert stock.commit(event) == 0
    expected = area * depth
    removed = stock.initial_volume_m3 - stock.volume_m3
    # 从实际体素侧边采样边界端点/中点，核对独立圆角矩形 SDF。
    mask = (~stock.occupied).reshape(stock.shape).any(axis=2)
    samples = []
    h = cell_size_m / 2
    for i, j in np.argwhere(mask):
        center = stock.centers[np.ravel_multi_index((i, j, 0), stock.shape)][:2]
        for axis in (0, 1):
            for sign in (-1, 1):
                other = [i, j]
                other[axis] += sign
                if 0 <= other[axis] < stock.shape[axis] and mask[tuple(other)]:
                    continue
                for along in (-h, 0, h):
                    point = center.copy()
                    point[axis] += sign * h
                    point[1 - axis] += along
                    samples.append(point)
    delta = np.abs(np.asarray(samples)) - [a, b]
    sdf = (
        np.linalg.norm(np.maximum(delta, 0), axis=1)
        + np.minimum(np.max(delta, axis=1), 0)
        - radius
    )
    boundary = float(np.max(np.abs(sdf)))
    return stock, {
        "shape": name,
        "cell_size_m": cell_size_m,
        "cells": len(stock.centers),
        "expected_volume_m3": expected,
        "removed_volume_m3": removed,
        "volume_relative_error": abs(removed / expected - 1),
        "boundary_error_m": boundary,
        "boundary_error_cells": boundary / cell_size_m,
        "removed_mass_kg": removed * stock.params.density_kg_m3,
        "state_hash": stock.state_hash(),
        "version": stock.version,
        "event_count": len(stock.events),
    }


def section_image(stock, path):
    """由占据状态生成俯视及两个截面；显示没有自己的去除规则。"""
    occ = stock.occupied.reshape(stock.shape)
    slices = [occ[:, :, -1], occ[:, occ.shape[1] // 2, :], occ[occ.shape[0] // 2, :, :]]
    canvas = Image.new("RGB", (900, 350), (20, 25, 32))
    draw = ImageDraw.Draw(canvas)
    for index, data in enumerate(slices):
        rgb = np.where(data.T[::-1, :, None], [150, 170, 185], [25, 35, 45]).astype(
            np.uint8
        )
        tile = Image.fromarray(rgb)
        tile.thumbnail((280, 280))
        tile = tile.resize((280, 280), Image.Resampling.NEAREST)
        canvas.paste(tile, (10 + 300 * index, 40))
        draw.text(
            (10 + 300 * index, 10),
            ["TOP", "X-Z SECTION", "Y-Z SECTION"][index],
            fill="white",
        )
    canvas.save(path)
