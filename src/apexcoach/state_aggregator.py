from __future__ import annotations

from collections import deque

from apexcoach.models import Action, FrameEvents, GameState, ParsedStatus, ParsedTactical


class StateAggregator:
    def __init__(
        self,
        knock_recent_seconds: float = 2.5,
        under_fire_damage_1s: float = 0.03,
        retreat_low_total_hp_shield: float = 0.45,
        heal_total_hp_shield: float = 0.65,
        vitals_confidence_min: float = 0.3,
        movement_score_threshold: float = 0.045,
        max_history_seconds: float = 5.0,
    ) -> None:
        self.knock_recent_seconds = knock_recent_seconds
        self.under_fire_damage_1s = under_fire_damage_1s
        self.retreat_low_total_hp_shield = retreat_low_total_hp_shield
        self.heal_total_hp_shield = heal_total_hp_shield
        self.vitals_confidence_min = vitals_confidence_min
        self.movement_score_threshold = movement_score_threshold
        self.max_history_seconds = max_history_seconds

        self._damage_events: deque[tuple[float, float]] = deque()
        self._enemy_knocks: deque[float] = deque()
        self._ally_knocks: deque[float] = deque()

        self._hp_pct = 1.0
        self._shield_pct = 1.0
        self._vitals_confidence = 1.0
        self._retreat_low_hp_streak = 0
        self._heal_low_hp_streak = 0
        self._is_moving = False
        self._movement_score = 0.0
        self._moving_recent_frames = 0
        self._stationary_frames = 0
        self._allies_alive = 3
        self._allies_down = 0
        self._low_ground_disadvantage = False
        self._low_ground_confidence = 0.0
        self._exposed_no_cover = False
        self._exposed_confidence = 0.0
        self._last_action = Action.NONE
        self._last_action_time: float | None = None

    def update(
        self,
        status: ParsedStatus,
        events: FrameEvents,
        tactical: ParsedTactical | None = None,
    ) -> GameState:
        timestamp = events.timestamp
        if status.hp_pct is not None:
            self._hp_pct = status.hp_pct
        if status.shield_pct is not None:
            self._shield_pct = status.shield_pct
        self._vitals_confidence = min(status.hp_confidence, status.shield_confidence)
        self._update_low_hp_streaks()
        if status.allies_alive is not None:
            self._allies_alive = status.allies_alive
        if status.allies_down is not None:
            self._allies_down = status.allies_down
        if tactical is not None:
            if tactical.low_ground_disadvantage is not None:
                self._low_ground_disadvantage = tactical.low_ground_disadvantage
                self._low_ground_confidence = tactical.low_ground_confidence
            if tactical.exposed_no_cover is not None:
                self._exposed_no_cover = tactical.exposed_no_cover
                self._exposed_confidence = tactical.exposed_confidence
            if tactical.is_moving is not None or tactical.movement_score > 0.0:
                if tactical.is_moving is None:
                    self._is_moving = (
                        tactical.movement_score >= self.movement_score_threshold
                    )
                else:
                    self._is_moving = tactical.is_moving
                self._movement_score = tactical.movement_score

        self._update_movement_streaks()

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
            vitals_confidence=self._vitals_confidence,
            retreat_low_hp_streak=self._retreat_low_hp_streak,
            heal_low_hp_streak=self._heal_low_hp_streak,
            is_moving=self._is_moving,
            movement_score=self._movement_score,
            moving_recent_frames=self._moving_recent_frames,
            stationary_frames=self._stationary_frames,
            allies_alive=self._allies_alive,
            allies_down=self._allies_down,
            recent_damage_1s=recent_damage_1s,
            recent_damage_3s=recent_damage_3s,
            under_fire=recent_damage_1s >= self.under_fire_damage_1s,
            enemy_knock_recent=self._has_recent(self._enemy_knocks, timestamp),
            ally_knock_recent=self._has_recent(self._ally_knocks, timestamp),
            low_ground_disadvantage=self._low_ground_disadvantage,
            low_ground_confidence=self._low_ground_confidence,
            exposed_no_cover=self._exposed_no_cover,
            exposed_confidence=self._exposed_confidence,
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

    def _update_low_hp_streaks(self) -> None:
        if self._vitals_confidence < self.vitals_confidence_min:
            self._retreat_low_hp_streak = 0
            self._heal_low_hp_streak = 0
            return

        total = self._hp_pct + self._shield_pct
        if total <= self.retreat_low_total_hp_shield:
            self._retreat_low_hp_streak += 1
        else:
            self._retreat_low_hp_streak = 0

        if total <= self.heal_total_hp_shield:
            self._heal_low_hp_streak += 1
        else:
            self._heal_low_hp_streak = 0

    def _update_movement_streaks(self) -> None:
        if self._is_moving:
            self._moving_recent_frames += 1
            self._stationary_frames = 0
        else:
            self._moving_recent_frames = 0
            self._stationary_frames += 1
