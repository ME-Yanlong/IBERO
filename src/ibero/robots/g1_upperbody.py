"""Build the fixed-base G1 + two grippers model used by Core-0.

The implementation deliberately composes the original Menagerie MJCF files at
runtime with :class:`mujoco.MjSpec`.  This preserves asset licenses and keeps
the source G1 model unmodified.  The pelvis is already world-fixed in the
Menagerie model; lower-body and waist actuators are held at their reset pose,
so the exposed task is an upper-body bimanual workcell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "vendor"
G1_XML = ASSET_ROOT / "unitree_g1" / "g1.xml"
GRIPPER_XML = ASSET_ROOT / "robotiq_2f85" / "2f85.xml"

ARM_JOINTS = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)


@dataclass(frozen=True)
class ModelHandles:
    """Named MuJoCo ids required by the controller and environment."""

    arm_actuator_ids: np.ndarray
    hold_actuator_ids: np.ndarray
    gripper_actuator_ids: np.ndarray
    left_anchor_site_id: int
    right_anchor_site_id: int
    left_wrist_body_id: int
    right_wrist_body_id: int
    cable_tendon_id: int
    cable_mocap_ids: np.ndarray


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise RuntimeError(f"Expected {kind.name} named {name!r} in assembled model")
    return result


def _spec_from_snapshot(xml_path: Path) -> mujoco.MjSpec:
    """Load an MJCF snapshot without passing a non-ASCII path to MuJoCo.

    MuJoCo's Windows file loader does not reliably handle the Chinese workspace
    path used by this project.  Feeding the XML and its mesh bytes through an
    ``MjSpec`` asset dictionary is both portable and self-contained.
    """

    spec = mujoco.MjSpec.from_string(xml_path.read_text(encoding="utf-8"))
    asset_dir = xml_path.parent / "assets"
    for asset_path in asset_dir.rglob("*"):
        if asset_path.is_file():
            relative = asset_path.relative_to(xml_path.parent).as_posix()
            spec.assets[relative] = asset_path.read_bytes()
    return spec


def _add_gripper(spec: mujoco.MjSpec, side: str) -> tuple[str, str]:
    """Attach a Menagerie 2F-85 to a G1 wrist and return anchor names."""

    wrist = spec.body(f"{side}_wrist_yaw_link")
    # The tool frame extends beyond the G1 wrist-yaw link in the local +x
    # direction.  It is also the physical cable attachment point in Core-0.
    mount = wrist.add_site(
        name=f"{side}_gripper_mount",
        pos=[0.065, 0.0, 0.0],
        size=[0.006],
        rgba=[0.15, 0.85, 0.95, 0.8],
    )
    anchor = wrist.add_site(
        name=f"{side}_cable_anchor",
        pos=[0.155, 0.0, 0.0],
        size=[0.009],
        rgba=[1.0, 0.55, 0.05, 1.0],
    )
    gripper = _spec_from_snapshot(GRIPPER_XML)
    spec.attach(gripper, site=mount, prefix=f"{side}_")
    return mount.name, anchor.name


def _add_cable(spec: mujoco.MjSpec, segments: int, length: float) -> None:
    """Add a stable Core-0 cable surrogate.

    A spatial tendon provides the actual extension force.  A row of mocap
    capsules visualises the cable with a controlled sag profile.  This avoids
    making the first controller test depend on unstable closed-chain contact;
    Flex/elasticity remains a later fidelity upgrade.
    """

    tendon = spec.add_tendon(
        name="cable_tendon", stiffness=110.0, damping=2.0, springlength=[length, length]
    )
    tendon.wrap_site("left_cable_anchor")
    tendon.wrap_site("right_cable_anchor")

    segment_half_length = (length / segments) * 0.58
    for index in range(segments):
        visual = spec.worldbody.add_body(name=f"cable_visual_{index}", mocap=True)
        visual.add_geom(
            name=f"cable_visual_geom_{index}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.010, segment_half_length],
            rgba=[0.95, 0.20, 0.04, 1.0],
            contype=0,
            conaffinity=0,
        )


def build_g1_cable_model(
    *, cable_segments: int = 12, cable_length: float = 0.48
) -> tuple[mujoco.MjModel, ModelHandles]:
    """Compile the Core-0 workcell and expose stable handles for it."""

    if not G1_XML.exists() or not GRIPPER_XML.exists():
        raise FileNotFoundError(
            f"Missing vendored Menagerie assets. Expected {G1_XML} and {GRIPPER_XML}."
        )

    spec = _spec_from_snapshot(G1_XML)
    spec.modelname = "ibero_g1_cable_tension"
    # Menagerie ships a locomotion model with a floating pelvis.  Core-0 is a
    # fixed industrial workcell, so remove only that free joint; all original
    # links and collision geometry remain available for later safety checks.
    spec.delete(spec.joint("floating_base_joint"))
    spec.option.timestep = 0.002  # 500 Hz physics; policy control is 20 Hz.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_RK4
    spec.worldbody.add_geom(
        name="workcell_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.1],
        pos=[0.0, 0.0, -0.74],
        rgba=[0.12, 0.13, 0.16, 1.0],
        contype=1,
        conaffinity=1,
    )
    spec.worldbody.add_light(
        name="workcell_key_light",
        pos=[0.3, -0.5, 1.6],
        dir=[-0.2, 0.35, -1.0],
        diffuse=[0.9, 0.9, 0.9],
    )

    _add_gripper(spec, "left")
    _add_gripper(spec, "right")
    _add_cable(spec, cable_segments, cable_length)
    model = spec.compile()

    arm_actuators = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINTS],
        dtype=np.int32,
    )
    gripper_actuators = np.asarray(
        [
            _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_fingers_actuator"),
            _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_fingers_actuator"),
        ],
        dtype=np.int32,
    )
    held = np.asarray(
        [index for index in range(model.nu) if index not in set(arm_actuators)],
        dtype=np.int32,
    )
    held = held[~np.isin(held, gripper_actuators)]
    handles = ModelHandles(
        arm_actuator_ids=arm_actuators,
        hold_actuator_ids=held,
        gripper_actuator_ids=gripper_actuators,
        left_anchor_site_id=_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_cable_anchor"),
        right_anchor_site_id=_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_cable_anchor"),
        left_wrist_body_id=_id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
        right_wrist_body_id=_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        ),
        cable_tendon_id=_id(model, mujoco.mjtObj.mjOBJ_TENDON, "cable_tendon"),
        cable_mocap_ids=np.asarray(
            [
                model.body_mocapid[
                    _id(model, mujoco.mjtObj.mjOBJ_BODY, f"cable_visual_{index}")
                ]
                for index in range(cable_segments)
            ],
            dtype=np.int32,
        ),
    )
    return model, handles
