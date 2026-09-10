"""带来源的力—位移参考导入与拟合；开发样本与留出样本不得重叠。"""

import csv
import json
from pathlib import Path
import numpy as np


def load_force_curve(csv_path, metadata_path):
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or set(meta) != {"source", "kind", "units"}:
        raise ValueError("Reference metadata requires source, kind and units")
    if meta["kind"] not in {
        "synthetic",
        "analytical",
        "numerical_reference",
        "measured",
    }:
        raise ValueError("Unknown reference kind")
    if (
        not isinstance(meta["source"], str)
        or not meta["source"].strip()
        or meta["units"] != {"displacement": "m", "force": "N"}
    ):
        raise ValueError("Reference provenance and SI units are required")
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["displacement_m", "force_n"]:
            raise ValueError("CSV requires displacement_m,force_n columns")
        values = np.asarray(
            [[float(r["displacement_m"]), float(r["force_n"])] for r in reader]
        )
    if values.ndim != 2 or values.shape[0] < 4 or not np.isfinite(values).all():
        raise ValueError("At least four finite reference samples required")
    return values, meta


def fit_linear_stiffness(values, train_indices, validation_indices):
    """仅适用于小变形加载曲线，不将滞回/塑性数据强行当线弹性。"""
    values = np.asarray(values, dtype=float)
    train, validation = np.asarray(train_indices), np.asarray(validation_indices)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or not np.isfinite(values).all()
        or train.ndim != 1
        or validation.ndim != 1
        or train.dtype.kind not in "iu"
        or validation.dtype.kind not in "iu"
        or min(len(train), len(validation)) < 2
        or set(train) & set(validation)
        or len(set(train)) != len(train)
        or len(set(validation)) != len(validation)
        or np.any(train < 0)
        or np.any(validation < 0)
        or np.any(train >= len(values))
        or np.any(validation >= len(values))
    ):
        raise ValueError(
            "Finite samples and nonoverlapping, unique train/validation indices required"
        )
    x, y = values[train].T
    if np.dot(x, x) <= 0:
        raise ValueError("Training displacement cannot be all zero")
    # 归一化避免有限但数量级过大的输入在点积中溢出，不能返回 NaN 标定结果。
    x_scale, y_scale = np.max(np.abs(x)), np.max(np.abs(y))
    if y_scale == 0:
        raise ValueError("Positive passive stiffness required")
    xn, yn = x / x_scale, y / y_scale
    stiffness = float((np.dot(xn, yn) / np.dot(xn, xn)) * (y_scale / x_scale))
    if not np.isfinite(stiffness) or stiffness <= 0:
        raise ValueError("Positive passive stiffness required")
    predicted = stiffness * values[validation, 0]
    scale = np.linalg.norm(values[validation, 1])
    if scale == 0 or not np.isfinite(scale) or not np.isfinite(predicted).all():
        raise ValueError("Nonzero validation reference scale required")
    return {
        "stiffness_n_m": stiffness,
        "validation_relative_l2": float(
            np.linalg.norm(predicted - values[validation, 1]) / scale
        ),
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
    }
