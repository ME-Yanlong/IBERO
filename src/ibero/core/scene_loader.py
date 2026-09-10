"""Strict loading for the small Core-0.1 scene declaration format."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


class SceneValidationError(ValueError):
    """Raised for a scene declaration that violates the Core-0.1 contract."""


@dataclass(frozen=True)
class ValidatedScene:
    """Validated YAML documents and content hashes for a contribution scene."""

    root: Path
    config: Mapping[str, Any]
    constraints: Mapping[str, Any]
    config_hash: str
    constraints_hash: str
    task_path: Path

    @property
    def scene_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.config_hash.encode("ascii"))
        digest.update(self.constraints_hash.encode("ascii"))
        digest.update(self.task_path.read_bytes())
        return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SceneValidationError(f"Required scene file is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SceneValidationError(f"{path.name} must contain a YAML mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    path: str,
    required: set[str] | None = None,
) -> None:
    extra = set(value) - allowed
    missing = (required or set()) - set(value)
    if extra:
        raise SceneValidationError(f"Unknown fields at {path}: {sorted(extra)}")
    if missing:
        raise SceneValidationError(f"Missing fields at {path}: {sorted(missing)}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SceneValidationError(f"{path} must be a mapping")
    return value


class SceneLoader:
    """Validate a Core-0.1 scene without compiling a simulator."""

    _config_keys = {
        "schema_version",
        "id",
        "suite",
        "backend",
        "robot",
        "physics",
        "materials",
        "initialization",
        "sensors",
        "render",
        "control",
        "workcell",
    }

    def validate(self, root: str | Path) -> ValidatedScene:
        root = Path(root).resolve()
        config_path = root / "scene_config.yaml"
        constraints_path = root / "constraints.yaml"
        task_path = root / "task_spec.py"
        config = _read_mapping(config_path)
        constraints = _read_mapping(constraints_path)
        try:
            if config.get("schema_version") == "ibero.industrial/v0.1":
                # 新台架走独立严格校验，不放松已经验收的线束菜谱规则。
                from ibero.core.industrial_config import validate_industrial

                validate_industrial(config, constraints)
            else:
                self._validate_config(config)
                self._validate_constraints(constraints)
                self._validate_physics(config, constraints)
        except (TypeError, ValueError, KeyError) as error:
            if isinstance(error, SceneValidationError):
                raise
            raise SceneValidationError(f"Invalid scene value/type: {error}") from error
        self._validate_task(task_path)
        return ValidatedScene(
            root=root,
            config=config,
            constraints=constraints,
            config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            constraints_hash=hashlib.sha256(constraints_path.read_bytes()).hexdigest(),
            task_path=task_path,
        )

    def _validate_config(self, config: Mapping[str, Any]) -> None:
        _require_exact_keys(
            config,
            allowed=self._config_keys,
            required=self._config_keys - {"control", "workcell"},
            path="scene_config",
        )
        if config["schema_version"] != "ibero.scene/v0.1":
            raise SceneValidationError("schema_version must be 'ibero.scene/v0.1'")
        if config["backend"] != "mujoco":
            raise SceneValidationError("Core-0.1 supports only backend: mujoco")
        robot = _mapping(config["robot"], "robot")
        _require_exact_keys(
            robot,
            allowed={"preset", "end_effector", "arm_qpos"},
            required={"preset", "end_effector"},
            path="robot",
        )
        # 末端执行器必须由菜谱显式声明，避免把 G1 原始手部外观与外接夹爪
        # 无意叠加。旧 preset 名仅为兼容早期 Core-0.1 菜谱，不代表强制 2F-85。
        if robot["preset"] not in {"g1_upperbody_v0", "g1_upperbody_2f85_v0"}:
            raise SceneValidationError("Core-0.1 supports robot preset g1_upperbody_v0")
        end_effector = _mapping(robot["end_effector"], "robot.end_effector")
        _require_exact_keys(
            end_effector,
            allowed={"type"},
            required={"type"},
            path="robot.end_effector",
        )
        if end_effector["type"] not in {"robotiq_2f85", "g1_native_hand"}:
            raise SceneValidationError(
                "robot.end_effector.type must be robotiq_2f85 or g1_native_hand"
            )
        physics = _mapping(config["physics"], "physics")
        _require_exact_keys(
            physics,
            allowed={"timestep_s", "control_hz", "max_episode_steps"},
            required={"timestep_s", "control_hz", "max_episode_steps"},
            path="physics",
        )
        if float(physics["timestep_s"]) <= 0 or float(physics["control_hz"]) <= 0:
            raise SceneValidationError(
                "physics timestep_s and control_hz must be positive"
            )
        materials = _mapping(config["materials"], "materials")
        _require_exact_keys(
            materials, allowed={"cable"}, required={"cable"}, path="materials"
        )
        cable = _mapping(materials.get("cable"), "materials.cable")
        cable_keys = {
            "end_layout",
            "initial_span_m",
            "representation",
            "length_m",
            "outer_diameter_m",
            "linear_density_kg_m",
            "minimum_bend_radius_m",
            "terminal",
            "axial_stiffness_n_m",
            "bending_stiffness_nm2",
            "damping",
            "friction",
            "parameter_source",
            "segments",
        }
        _require_exact_keys(
            cable,
            allowed=cable_keys,
            required=cable_keys - {"end_layout", "initial_span_m"},
            path="materials.cable",
        )
        if cable["representation"] != "flex":
            raise SceneValidationError(
                "This revision supports representation: flex; the former capsule_chain alias was removed because it was not an independent backend"
            )
        if float(cable["length_m"]) <= 0 or float(cable["outer_diameter_m"]) <= 0:
            raise SceneValidationError(
                "Cable length_m and outer_diameter_m must be positive"
            )
        terminal = _mapping(cable["terminal"], "materials.cable.terminal")
        _require_exact_keys(
            terminal,
            allowed={"length_m", "width_m", "height_m", "mass_kg"},
            required={"length_m", "width_m", "height_m", "mass_kg"},
            path="materials.cable.terminal",
        )
        for value in terminal.values():
            if float(value) <= 0:
                raise SceneValidationError(
                    "All terminal dimensions and mass must be positive"
                )
        initialization = _mapping(config["initialization"], "initialization")
        _require_exact_keys(
            initialization,
            allowed={
                "seed_range",
                "terminal_pose",
                "tail_direction",
                "target_region",
                "target_jitter_m",
            },
            required={
                "seed_range",
                "terminal_pose",
                "tail_direction",
                "target_region",
                "target_jitter_m",
            },
            path="initialization",
        )

    def _validate_constraints(self, constraints: Mapping[str, Any]) -> None:
        _require_exact_keys(
            constraints,
            allowed={"cable", "task", "collision"},
            required={"cable", "task", "collision"},
            path="constraints",
        )
        cable = _mapping(constraints["cable"], "constraints.cable")
        _require_exact_keys(
            cable,
            allowed={"tension_limit_n", "limit_duration_s"},
            required={"tension_limit_n", "limit_duration_s"},
            path="constraints.cable",
        )
        if float(cable["tension_limit_n"]) <= 0 or float(cable["limit_duration_s"]) < 0:
            raise SceneValidationError("Cable safety limits must be non-negative")
        task = _mapping(constraints["task"], "constraints.task")
        _require_exact_keys(
            task,
            allowed={"grasp_window_m", "target_radius_m", "release_opening_m"},
            required={"grasp_window_m", "target_radius_m", "release_opening_m"},
            path="constraints.task",
        )
        collision = _mapping(constraints["collision"], "constraints.collision")
        _require_exact_keys(
            collision,
            allowed={"allowed_robot_pairs"},
            required={"allowed_robot_pairs"},
            path="constraints.collision",
        )

    @staticmethod
    def _validate_physics(config, constraints):
        """把配置的 SI 单位、有限值、稳定性范围和互相依赖关系一起检查。"""

        def finite_tree(value, path):
            if isinstance(value, dict):
                for key, child in value.items():
                    finite_tree(child, f"{path}.{key}")
            elif isinstance(value, list):
                for child in value:
                    finite_tree(child, path)
            elif isinstance(value, (float, int)) and not math.isfinite(value):
                raise SceneValidationError(f"{path} must be finite")

        finite_tree(config, "scene")
        finite_tree(constraints, "constraints")
        cable = config["materials"]["cable"]
        phys = config["physics"]

        def number(value, path, positive=False):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or (positive and value <= 0)
            ):
                raise SceneValidationError(
                    f"{path} must be a finite {'positive ' if positive else ''}number"
                )

        for key, value in phys.items():
            number(value, f"physics.{key}", True)
        if not isinstance(phys["max_episode_steps"], int):
            raise SceneValidationError("max_episode_steps must be an integer")
        for key, value in cable["terminal"].items():
            number(value, f"terminal.{key}", True)
        for key, value in constraints["task"].items():
            number(value, f"constraints.task.{key}", True)
        for key, value in constraints["cable"].items():
            number(value, f"constraints.cable.{key}")
        sources = _mapping(cable["parameter_source"], "parameter_source")
        source_keys = {
            "length_m",
            "outer_diameter_m",
            "linear_density_kg_m",
            "minimum_bend_radius_m",
            "terminal",
            "axial_stiffness_n_m",
            "bending_stiffness_nm2",
            "damping",
            "friction",
        }
        _require_exact_keys(
            sources, allowed=source_keys, required=source_keys, path="parameter_source"
        )
        if any(
            not isinstance(v, str)
            or not v.startswith(("provisional", "measured", "datasheet"))
            for v in sources.values()
        ):
            raise SceneValidationError(
                "parameter_source values must identify provisional, measured or datasheet provenance"
            )
        for key in (
            "length_m",
            "outer_diameter_m",
            "linear_density_kg_m",
            "minimum_bend_radius_m",
            "axial_stiffness_n_m",
            "friction",
        ):
            if not isinstance(cable[key], (float, int)) or cable[key] <= 0:
                raise SceneValidationError(f"materials.cable.{key} must be positive")
        for key in ("bending_stiffness_nm2", "damping"):
            if not isinstance(cable[key], (float, int)) or cable[key] < 0:
                raise SceneValidationError(
                    f"materials.cable.{key} must be non-negative"
                )
        n = cable["segments"]
        if not isinstance(n, int) or isinstance(n, bool) or not 4 <= n <= 64:
            raise SceneValidationError("segments must be an integer in [4,64]")
        layout = cable.get("end_layout", "handover")
        if layout not in ("handover", "dual_end"):
            raise SceneValidationError("end_layout must be handover or dual_end")
        span = cable.get("initial_span_m")
        if span is not None:
            number(span, "initial_span_m", True)
        occupied = cable["terminal"]["length_m"] * (1 if layout == "dual_end" else 0.5)
        if span is not None and not occupied < span <= cable["length_m"] + occupied:
            raise SceneValidationError(
                "initial_span_m must leave a positive cable gap no longer than its arc length"
            )
        dt = phys["timestep_s"]
        ratio = 1 / (phys["control_hz"] * dt)
        if abs(ratio - round(ratio)) > 1e-7 or ratio < 1:
            raise SceneValidationError(
                "control period must contain an integer number of physics steps"
            )
        mass = cable["linear_density_kg_m"] * cable["length_m"] / (n + 1)
        bound = 0.5 * math.sqrt(mass / (n * cable["axial_stiffness_n_m"]))
        if cable["bending_stiffness_nm2"] > 0:
            # 显式弯曲力的高频模态也限制步长；这是保守筛查，不替代数值试验。
            bound = min(
                bound,
                0.5
                * math.sqrt(
                    mass
                    * (cable["length_m"] / n) ** 3
                    / (16 * cable["bending_stiffness_nm2"])
                ),
            )
        if cable["damping"] > 0:
            bound = min(bound, 0.5 * mass / (n * cable["damping"]))
        if dt > bound:
            raise SceneValidationError(
                f"timestep_s exceeds conservative material bound {bound:.6g}; reduce timestep or segments"
            )
        q = config["robot"].get("arm_qpos", [0] * 14)
        if len(q) != 14 or not all(
            isinstance(v, (float, int)) and math.isfinite(v) for v in q
        ):
            raise SceneValidationError(
                "robot.arm_qpos must contain 14 finite joint angles"
            )
        for key in (
            "terminal_pose",
            "tail_direction",
            "target_region",
            "target_jitter_m",
        ):
            values = config["initialization"][key]
            if (
                not isinstance(values, list)
                or len(values) != 3
                or not all(isinstance(v, (float, int)) for v in values)
            ):
                raise SceneValidationError(
                    f"initialization.{key} must contain three numbers"
                )
        if sum(v * v for v in config["initialization"]["tail_direction"]) == 0:
            raise SceneValidationError("tail_direction must be nonzero")
        if any(v < 0 for v in config["initialization"]["target_jitter_m"]):
            raise SceneValidationError("target_jitter_m must be nonnegative")
        seed_range = config["initialization"]["seed_range"]
        if (
            not isinstance(seed_range, list)
            or len(seed_range) != 2
            or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 0
                for v in seed_range
            )
            or seed_range[0] > seed_range[1]
        ):
            raise SceneValidationError(
                "seed_range must be two ordered nonnegative integers"
            )
        pairs = constraints["collision"]["allowed_robot_pairs"]
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(v, str) and v for v in pair)
            for pair in pairs
        ):
            raise SceneValidationError(
                "allowed_robot_pairs must list pairs of body names"
            )
        for key, allowed in [
            ("sensors", {"required"}),
            ("render", {"cameras"}),
            ("workcell", {"receiver"}),
        ]:
            if key in config:
                _require_exact_keys(config[key], allowed=allowed, path=key)
        if "receiver" in config.get("workcell", {}):
            receiver = config["workcell"]["receiver"]
            _require_exact_keys(
                receiver,
                allowed={"offset_m", "half_size_m"},
                required={"offset_m", "half_size_m"},
                path="workcell.receiver",
            )
            if any(len(receiver[k]) != 3 for k in receiver) or any(
                v <= 0 for v in receiver["half_size_m"]
            ):
                raise SceneValidationError(
                    "receiver requires 3D offset and positive half sizes"
                )
        if layout == "dual_end":
            cfg = config.get("control", {})
            keys = {
                "target_tension_n",
                "tension_tolerance_n",
                "hold_seconds",
                "admittance_m_ns",
                "max_speed_m_s",
                "filter_time_s",
                "warning_tension_n",
                "recovery_tension_n",
                "disturbance_time_s",
                "disturbance_duration_s",
                "disturbance_force_n",
            }
            _require_exact_keys(
                cfg,
                allowed=keys | {"disturbance_jitter_fraction"},
                required=keys,
                path="control",
            )
            if not 0 <= cfg.get("disturbance_jitter_fraction", 0) <= 1:
                raise SceneValidationError(
                    "disturbance_jitter_fraction must be in [0,1]"
                )
            for key in keys - {"disturbance_force_n"}:
                if not isinstance(cfg[key], (float, int)) or cfg[key] < 0:
                    raise SceneValidationError(f"control.{key} must be nonnegative")
            if (
                not 0
                < cfg["target_tension_n"]
                < cfg["recovery_tension_n"]
                < cfg["warning_tension_n"]
                < constraints["cable"]["tension_limit_n"]
            ):
                raise SceneValidationError(
                    "Require target < recovery < warning < damage tension"
                )
            if (
                cfg["hold_seconds"] <= 0
                or cfg["max_speed_m_s"] <= 0
                or cfg["admittance_m_ns"] <= 0
            ):
                raise SceneValidationError(
                    "hold_seconds, speed and admittance must be positive"
                )
            if len(cfg["disturbance_force_n"]) != 3:
                raise SceneValidationError(
                    "disturbance_force_n must have three components"
                )
            for value in cfg["disturbance_force_n"]:
                number(value, "disturbance_force_n")
        elif config.get("control"):
            raise SceneValidationError(
                "This handover baseline has no scene control overrides; unknown control fields are not ignored"
            )

    @staticmethod
    def _validate_task(task_path: Path) -> None:
        if not task_path.is_file():
            raise SceneValidationError(f"Required scene file is missing: {task_path}")
        source = task_path.read_text(encoding="utf-8")
        forbidden = ("import mujoco", "from mujoco", "MjModel", "MjData")
        if any(token in source for token in forbidden):
            raise SceneValidationError(
                "task_spec.py must use TaskState and must not access MuJoCo directly"
            )
        if "TASK_SPEC" not in source:
            raise SceneValidationError("task_spec.py must expose a TASK_SPEC instance")
