"""Compile a validated Core-0.1 scene into its concrete MuJoCo workcell."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from ibero.core.scene_loader import ValidatedScene
from ibero.materials.cable import CableParameters
from ibero.robots.g1_upperbody_2f85 import HandoverModelHandles, build_g1_handover_model
from ibero.robots.g1_industrial import IndustrialRobotHandles
from ibero.mechanisms.snap_latch import LatchNames
from ibero.materials.harness import HarnessParameters, add_harness


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


@dataclass(frozen=True)
class CompiledLatchCell:
    """实际卡扣工作站产物，不扩张成任意引擎/材料容器。"""

    scene: ValidatedScene
    model: mujoco.MjModel
    handles: IndustrialRobotHandles
    latch_names: LatchNames


@dataclass(frozen=True)
class CompiledHarnessCell(CompiledLatchCell):
    harness_parameters: HarnessParameters


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

    def compile(
        self, scene: ValidatedScene
    ) -> CompiledScene | CompiledFixture | CompiledLatchCell:
        if scene.config.get("schema_version") == "ibero.industrial/v0.1":
            if scene.config["kind"] in {"latch_release", "harness_unplug"}:
                from ibero.robots.g1_industrial import robot_spec, robot_handles
                from ibero.mechanisms.snap_latch import add_latch
                from ibero.materials.parameters import (
                    BeamParameters,
                    LatchParameters,
                    strict_parameters,
                )

                cfg = scene.config
                spec = robot_spec(
                    timestep=cfg["physics"]["timestep_s"],
                    press_tcp_offset_m=cfg["robot"]["press_tcp_offset_m"],
                    press_stem_height_m=cfg["robot"]["press_stem_height_m"],
                )
                spec.option.gravity = cfg["physics"]["gravity_m_s2"]
                names = add_latch(
                    spec,
                    strict_parameters(BeamParameters, cfg["materials"]["beam"]),
                    strict_parameters(LatchParameters, cfg["mechanism"]),
                    segments=cfg["numerics"]["segments"],
                    fixture=False,
                    origin=cfg["initialization"]["origin_m"],
                    quaternion=cfg["initialization"]["quaternion_wxyz"],
                )
                if cfg["kind"] == "harness_unplug":
                    params = strict_parameters(
                        HarnessParameters, cfg["materials"]["cable"]
                    )
                    add_harness(
                        spec,
                        params,
                        origin=cfg["initialization"]["origin_m"],
                        quaternion=cfg["initialization"]["quaternion_wxyz"],
                        tail_offset_m=cfg["initialization"]["tail_offset_m"],
                    )
                    receiver = cfg["workcell"]["receiver"]
                    spec.worldbody.add_geom(
                        name="harness_receiver",
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        solref=[0.001, 1],
                        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
                        pos=receiver["center_m"],
                        size=receiver["half_size_m"],
                        friction=[0.6, 0.005, 0.0001],
                        rgba=[0.15, 0.45, 0.35, 1],
                    )
                    model = spec.compile()
                    return CompiledHarnessCell(
                        scene, model, robot_handles(model), names, params
                    )
                model = spec.compile()
                return CompiledLatchCell(scene, model, robot_handles(model), names)
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
