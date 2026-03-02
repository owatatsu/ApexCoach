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
    TAKE_COVER = "TAKE_COVER"
    TAKE_HIGH_GROUND = "TAKE_HIGH_GROUND"
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
    hp_confidence: float = 0.0
    shield_confidence: float = 0.0
    allies_alive: int | None = None
    allies_down: int | None = None


@dataclass(slots=True)
class ParsedTactical:
    low_ground_disadvantage: bool | None = None
    low_ground_confidence: float = 0.0
    exposed_no_cover: bool | None = None
    exposed_confidence: float = 0.0
    is_moving: bool | None = None
    movement_score: float = 0.0


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
    vitals_confidence: float = 1.0
    retreat_low_hp_streak: int = 0
    heal_low_hp_streak: int = 0
    is_moving: bool = False
    movement_score: float = 0.0
    moving_recent_frames: int = 0
    stationary_frames: int = 0
    allies_alive: int = 3
    allies_down: int = 0
    recent_damage_1s: float = 0.0
    recent_damage_3s: float = 0.0
    under_fire: bool = False
    enemy_knock_recent: bool = False
    ally_knock_recent: bool = False
    low_ground_disadvantage: bool = False
    low_ground_confidence: float = 0.0
    exposed_no_cover: bool = False
    exposed_confidence: float = 0.0
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
