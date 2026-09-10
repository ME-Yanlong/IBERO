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


@dataclass(frozen=True)
class CompiledFixture:
    """独立台架没有机器人/线束句柄，不构造虚假的 cable 参数。"""

    scene: ValidatedScene
    model: mujoco.MjModel


def fixture_spec(config):
    """空台架只负责求解时钟、重力和照明；对象由相应 builder 添加。"""
    spec = mujoco.MjSpec()
    spec.modelname = config["id"]
    spec.option.timestep = config["physics"]["timestep_s"]
    spec.option.gravity = config["physics"]["gravity_m_s2"]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_light(pos=[0, 0, 1])
    return spec


class SceneCompiler:
    """The narrow Core-0.1 configuration-to-backend compilation seam."""

    def compile(self, scene: ValidatedScene) -> CompiledScene | CompiledFixture:
        if scene.config.get("schema_version") == "ibero.industrial/v0.1":
            if scene.config["kind"] == "latch_bench":
                from ibero.benches.latch import build_latch_fixture
                from ibero.materials.parameters import (
                    BeamParameters,
                    LatchParameters,
                    strict_parameters,
                )

                spec, _ = build_latch_fixture(
                    strict_parameters(
                        BeamParameters, scene.config["materials"]["beam"]
                    ),
                    strict_parameters(LatchParameters, scene.config["mechanism"]),
                    segments=scene.config["numerics"]["segments"],
                    timestep=scene.config["physics"]["timestep_s"],
                )
                spec.option.gravity = scene.config["physics"]["gravity_m_s2"]
                return CompiledFixture(scene, spec.compile())
            if scene.config["kind"] != "empty_bench":
                raise NotImplementedError(
                    f"Builder not implemented: {scene.config['kind']}"
                )
            return CompiledFixture(scene, fixture_spec(scene.config).compile())
        model, handles, cable_parameters = build_g1_handover_model(
            scene.config, scene.constraints
        )
        return CompiledScene(
            scene=scene,
            model=model,
            handles=handles,
            cable_parameters=cable_parameters,
        )
