"""无物理、无去除事件的加工网格预估；不是机器人或工艺验收。"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from ibero.control.milling_bench import bench_target
from ibero.core.scene_loader import SceneLoader
from ibero.core.task_loading import load_task_spec
from ibero.core.reproducibility import simulation_source_hash
from ibero.core.milling_config import validate_milling_bench
from ibero.core.plate_milling_config import validate_plate_milling
from ibero.materials.parameters import StockParameters, strict_parameters
from ibero.materials.stock import VoxelStock
from ibero.processes.shape_check import inspect_shape


def forecast(scene, cell_sizes):
    cfg = scene.config
    if cfg["kind"] not in {"milling_bench", "plate_milling"}:
        raise ValueError("Expected milling recipe")
    task = load_task_spec(scene) if cfg["kind"] == "plate_milling" else None
    rows = []
    for cell in cell_sizes:
        candidate = copy.deepcopy(cfg)
        candidate["numerics"]["cell_size_m"] = cell
        recipe_error = None
        try:
            validate = validate_plate_milling if task else validate_milling_bench
            validate(candidate, scene.constraints)
        except ValueError as error:
            # 保留几何预估，但明确某格距不能直接用于原机器人安装/控制包络。
            recipe_error = str(error)
        stock = VoxelStock(
            strict_parameters(StockParameters, cfg["materials"]["stock"]),
            cell,
            max_cells=cfg["numerics"]["max_cells"],
        )
        for shape in ("slot", "pocket", "through_hole"):
            target = task.make_target(shape) if task else bench_target(shape)
            inspect_shape(stock, target)  # 复用只读范围校验，域外 P4 目标不输出伪预估。
            bottom = stock.params.size_m[2] / 2 - target.depth_m
            mask = (target.planar_sdf(stock.centers[:, :2]) <= 0) & (
                stock.centers[:, 2] >= bottom
            )
            # 只对体积权重求和；绝不提交目标掩码到真实 stock / 碰撞模型。
            volume = float(stock.volumes[mask].sum())
            error = abs(volume / target.volume_m3 - 1)
            rows.append(
                dict(
                    shape=shape,
                    cell_size_m=cell,
                    candidate_recipe_valid=recipe_error is None,
                    candidate_recipe_error=recipe_error,
                    cells=len(stock.centers),
                    nominal_center_classification_volume_m3=volume,
                    target_volume_m3=target.volume_m3,
                    relative_volume_error=error,
                    within_volume_criterion=error
                    <= scene.constraints["task"]["volume_error_fraction"],
                    material_events=len(stock.events),
                )
            )
    return dict(
        scope="geometric_quantization_forecast_no_physics",
        source_hash=simulation_source_hash(),
        original_scene_hash=scene.scene_hash,
        physical_steps=0,
        results=rows,
        note="Nominal center classification, not a proof of best possible geometry, actual cutter sweep, load convergence, or successful machining. Finer grids need not improve this error monotonically.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("scenes/milling_bench"))
    parser.add_argument(
        "--cell-sizes-m",
        type=float,
        nargs="+",
        default=[0.001, 0.00075, 0.002 / 3, 0.0005],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = (
        args.output
        or Path("artifacts/industrial_core01/stock/grid_forecast")
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        / "report.json"
    )
    if out.exists():
        parser.error("Refusing to overwrite existing evidence")
    result = forecast(SceneLoader().validate(args.scene), args.cell_sizes_m)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(out, "forecast only; no physics executed", flush=True)


if __name__ == "__main__":
    main()
