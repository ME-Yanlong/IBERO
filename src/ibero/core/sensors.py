"""Named sensor helpers with explicit frame metadata."""

from __future__ import annotations

import mujoco
import numpy as np


def sensor_vector(
    model: mujoco.MjModel, data: mujoco.MjData, sensor_id: int
) -> np.ndarray:
    """Return a copied sensor vector, independent of MuJoCo's data buffer."""

    address = model.sensor_adr[sensor_id]
    dimension = model.sensor_dim[sensor_id]
    return data.sensordata[address : address + dimension].copy()


def wrist_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    force_sensor_id: int,
    torque_sensor_id: int,
) -> np.ndarray:
    """Return `[Fx, Fy, Fz, Tx, Ty, Tz]` in the named wrist site frame."""

    return np.concatenate(
        (
            sensor_vector(model, data, force_sensor_id),
            sensor_vector(model, data, torque_sensor_id),
        )
    )
