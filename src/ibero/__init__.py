"""IBERO public package surface for Core-0 and Core-0.1 environments."""

from ibero.envs.cable_handover import CableHandoverEnv
from ibero.envs.cable_tension import CableTensionEnv


def make(env_id: str, **kwargs) -> CableTensionEnv | CableHandoverEnv:
    """Construct a supported IBERO environment.

    The registry intentionally remains small: a Core-0 force smoke test and
    the first Core-0.1 contribution scene.
    """

    if env_id == "ibero/CableTension-v0":
        return CableTensionEnv(**kwargs)
    if env_id == "ibero/LatchRelease-v0":
        from ibero.envs.latch_release import LatchReleaseEnv

        return LatchReleaseEnv(**kwargs)
    if env_id == "ibero/HarnessUnplug-v0":
        from ibero.envs.harness_unplug import HarnessUnplugEnv

        return HarnessUnplugEnv(**kwargs)
    if env_id == "ibero/CableHandover-v0":
        return CableHandoverEnv(**kwargs)
    if env_id == "ibero/CableStretch-v0":
        from ibero.envs.cable_handover import DEFAULT_SCENE_PATH

        kwargs.setdefault("scene_path", DEFAULT_SCENE_PATH.parent / "cable_stretch")
        return CableHandoverEnv(**kwargs)
    if env_id in {
        "ibero/CableClipInsert-v0": "cable_clip_insert",
        "ibero/CableRouteStraight-v0": "cable_route_straight",
        "ibero/CableRouteL-v0": "cable_route_l",
        "ibero/CableRouteRectangle-v0": "cable_route_rectangle",
    }:
        from pathlib import Path
        from ibero.envs.cable_routing import CableRoutingEnv

        scene_name = {
            "ibero/CableClipInsert-v0": "cable_clip_insert",
            "ibero/CableRouteStraight-v0": "cable_route_straight",
            "ibero/CableRouteL-v0": "cable_route_l",
            "ibero/CableRouteRectangle-v0": "cable_route_rectangle",
        }[env_id]
        kwargs.setdefault(
            "scene_path", Path(__file__).resolve().parents[2] / "scenes" / scene_name
        )
        return CableRoutingEnv(**kwargs)
    raise ValueError(f"Unsupported IBERO environment: {env_id!r}")


__all__ = ["CableHandoverEnv", "CableTensionEnv", "make"]
