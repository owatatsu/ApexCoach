from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Action(str, Enum):
    NONE = "NONE"
    HEAL = "HEAL"
    RETREAT = "RETREAT"
    PUSH = "PUSH"


@dataclass(slots=True)
class FramePacket:
    frame_index: int
    timestamp: float
    frame: "np.ndarray"


@dataclass(slots=True)
class ParsedStatus:
    hp_pct: float | None = None
    shield_pct: float | None = None
    allies_alive: int | None = None
    allies_down: int | None = None


@dataclass(slots=True)
class ParsedNotifications:
    enemy_knock: bool = False
    ally_knock: bool = False


@dataclass(slots=True)
class FrameEvents:
    timestamp: float
    damage_delta: float = 0.0
    enemy_knock: bool = False
    ally_knock: bool = False


@dataclass(slots=True)
class GameState:
    timestamp: float
    hp_pct: float = 1.0
    shield_pct: float = 1.0
    allies_alive: int = 3
    allies_down: int = 0
    recent_damage_1s: float = 0.0
    recent_damage_3s: float = 0.0
    under_fire: bool = False
    enemy_knock_recent: bool = False
    ally_knock_recent: bool = False
    last_action: Action = Action.NONE
    last_action_time: float | None = None


@dataclass(slots=True)
class Decision:
    action: Action
    reason: str
    confidence: float = 0.5
    meta: dict[str, str | float | int | bool] = field(default_factory=dict)


@dataclass(slots=True)
class ArbiterResult:
    action: Action
    emitted: bool
    reason: str
    source_action: Action
    debug_notes: list[str] = field(default_factory=list)
