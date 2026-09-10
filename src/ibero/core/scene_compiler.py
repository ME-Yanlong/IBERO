"""Compile a validated Core-0.1 scene into its concrete MuJoCo workcell."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from ibero.core.scene_loader import ValidatedScene
from ibero.materials.cable import CableParameters
from ibero.robots.g1_upperbody_2f85 import HandoverModelHandles, build_g1_handover_model


@dataclass(frozen=True)
class CompiledScene:
    """Concrete backend objects behind an already validated scene recipe."""

    scene: ValidatedScene
    model: mujoco.MjModel
    handles: HandoverModelHandles
    cable_parameters: CableParameters


class SceneCompiler:
    """The narrow Core-0.1 configuration-to-backend compilation seam."""

    def compile(self, scene: ValidatedScene) -> CompiledScene:
        model, handles, cable_parameters = build_g1_handover_model(
            scene.config, scene.constraints
        )
        return CompiledScene(
            scene=scene,
            model=model,
            handles=handles,
            cable_parameters=cable_parameters,
        )
