"""Backend-neutral task semantics for contribution scenes.

Scene task files receive :class:`TaskState`, never a mutable MuJoCo model or
data object.  This keeps scene semantics inspectable and makes the eventual
backend boundary real from the first scene.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

import numpy as np


@dataclass(frozen=True)
class TaskState:
    """Read-only task-relevant simulator snapshot."""

    body_positions: Mapping[str, np.ndarray]
    site_positions: Mapping[str, np.ndarray]
    gripper_openings: Mapping[str, float]
    grasps: Mapping[str, bool]
    contacts: frozenset[tuple[str, str]]
    cable_tension_n: float
    elapsed_steps: int
    max_episode_steps: int
    extra: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    """Semantic task outcome for a single control tick."""

    reward: float
    stage: str
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    failure_reason: str | None = None
    metrics: Mapping[str, float | bool | str] = field(default_factory=dict)


class TaskSpec(ABC):
    """Scene-owned task state machine with no direct physics access."""

    id: str
    instruction: str

    @abstractmethod
    def reset(self) -> None:
        """Reset transient stage state before a new episode."""

    @abstractmethod
    def evaluate(self, state: TaskState, audit: "AuditSnapshot") -> TaskResult:
        """Evaluate one immutable simulator snapshot."""


# Avoid a runtime import cycle while retaining useful type information.
if TYPE_CHECKING:
    from ibero.core.audit import AuditSnapshot
