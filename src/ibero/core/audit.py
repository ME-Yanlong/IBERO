"""Safety event collection and reproducibility manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
import math


@dataclass(frozen=True)
class AuditSnapshot:
    """Immutable safety state visible to a :class:`~ibero.core.TaskSpec`."""

    cable_damage: bool
    self_collision: bool
    cable_over_limit_seconds: float
    peak_cable_tension_n: float
    events: tuple[Mapping[str, Any], ...]


@dataclass
class SafetyAuditor:
    """Latch non-destructive-operation failures across an episode."""

    tension_limit_n: float
    tension_limit_duration_s: float
    _over_limit_seconds: float = 0.0
    _peak_tension_n: float = 0.0
    _cable_damage: bool = False
    _self_collision: bool = False
    _events: list[dict[str, Any]] = field(default_factory=list)
    _time_s: float = 0.0

    def reset(self) -> None:
        self._over_limit_seconds = 0.0
        self._peak_tension_n = 0.0
        self._cable_damage = False
        self._self_collision = False
        self._events.clear()
        self._time_s = 0.0

    def record(self, event_type, **details):
        self._events.append({"type": event_type, "time_s": self._time_s, **details})

    def observe(
        self,
        *,
        cable_tension_n: float,
        self_collision: bool,
        dt: float,
        collision_pairs=(),
    ) -> None:
        """Record a physics tick.  Damage is latched and cannot be cleared."""

        if not math.isfinite(cable_tension_n) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Safety audit requires finite tension and positive dt")
        self._time_s += dt

        self._peak_tension_n = max(self._peak_tension_n, float(cable_tension_n))
        if cable_tension_n > self.tension_limit_n:
            self._over_limit_seconds += dt
            if (
                not self._cable_damage
                and self._over_limit_seconds >= self.tension_limit_duration_s
            ):
                self._cable_damage = True
                self._events.append(
                    {
                        "type": "cable_damage",
                        "time_s": self._time_s,
                        "tension_n": float(cable_tension_n),
                        "duration_s": self._over_limit_seconds,
                    }
                )
        else:
            self._over_limit_seconds = 0.0

        if self_collision and not self._self_collision:
            self._self_collision = True
            self.record(
                "self_collision", pairs=[list(pair) for pair in collision_pairs]
            )

    def snapshot(self) -> AuditSnapshot:
        return AuditSnapshot(
            cable_damage=self._cable_damage,
            self_collision=self._self_collision,
            cable_over_limit_seconds=self._over_limit_seconds,
            peak_cable_tension_n=self._peak_tension_n,
            events=tuple(self._events),
        )

    def manifest(self, *, base: Mapping[str, Any]) -> dict[str, Any]:
        """Return serialisable run metadata without imposing a dataset format."""

        result = dict(base)
        result["audit"] = asdict(self.snapshot())
        return result
