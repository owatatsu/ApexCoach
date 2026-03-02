from __future__ import annotations

from collections import deque

from apexcoach.models import Action, FrameEvents, GameState, ParsedStatus


class StateAggregator:
    def __init__(
        self,
        knock_recent_seconds: float = 2.5,
        under_fire_damage_1s: float = 0.03,
        max_history_seconds: float = 5.0,
    ) -> None:
        self.knock_recent_seconds = knock_recent_seconds
        self.under_fire_damage_1s = under_fire_damage_1s
        self.max_history_seconds = max_history_seconds

        self._damage_events: deque[tuple[float, float]] = deque()
        self._enemy_knocks: deque[float] = deque()
        self._ally_knocks: deque[float] = deque()

        self._hp_pct = 1.0
        self._shield_pct = 1.0
        self._allies_alive = 3
        self._allies_down = 0
        self._last_action = Action.NONE
        self._last_action_time: float | None = None

    def update(self, status: ParsedStatus, events: FrameEvents) -> GameState:
        timestamp = events.timestamp
        if status.hp_pct is not None:
            self._hp_pct = status.hp_pct
        if status.shield_pct is not None:
            self._shield_pct = status.shield_pct
        if status.allies_alive is not None:
            self._allies_alive = status.allies_alive
        if status.allies_down is not None:
            self._allies_down = status.allies_down

        if events.damage_delta > 0.0:
            self._damage_events.append((timestamp, events.damage_delta))
        if events.enemy_knock:
            self._enemy_knocks.append(timestamp)
        if events.ally_knock:
            self._ally_knocks.append(timestamp)

        self._trim(timestamp)

        recent_damage_1s = self._sum_damage_since(timestamp - 1.0)
        recent_damage_3s = self._sum_damage_since(timestamp - 3.0)

        return GameState(
            timestamp=timestamp,
            hp_pct=self._hp_pct,
            shield_pct=self._shield_pct,
            allies_alive=self._allies_alive,
            allies_down=self._allies_down,
            recent_damage_1s=recent_damage_1s,
            recent_damage_3s=recent_damage_3s,
            under_fire=recent_damage_1s >= self.under_fire_damage_1s,
            enemy_knock_recent=self._has_recent(self._enemy_knocks, timestamp),
            ally_knock_recent=self._has_recent(self._ally_knocks, timestamp),
            last_action=self._last_action,
            last_action_time=self._last_action_time,
        )

    def record_action(self, action: Action, timestamp: float) -> None:
        self._last_action = action
        self._last_action_time = timestamp

    def _trim(self, now: float) -> None:
        cutoff = now - self.max_history_seconds
        while self._damage_events and self._damage_events[0][0] < cutoff:
            self._damage_events.popleft()
        while self._enemy_knocks and self._enemy_knocks[0] < cutoff:
            self._enemy_knocks.popleft()
        while self._ally_knocks and self._ally_knocks[0] < cutoff:
            self._ally_knocks.popleft()

    def _sum_damage_since(self, threshold: float) -> float:
        return sum(delta for ts, delta in self._damage_events if ts >= threshold)

    def _has_recent(self, events: deque[float], now: float) -> bool:
        if not events:
            return False
        return events[-1] >= now - self.knock_recent_seconds
