from __future__ import annotations

from collections import deque

from apexcoach.models import Action, FrameEvents, GameState, ParsedStatus, ParsedTactical
from apexcoach.vitals import combine_vitals_confidence


class StateAggregator:
    def __init__(
        self,
        knock_recent_seconds: float = 2.5,
        under_fire_damage_1s: float = 0.03,
        under_fire_release_damage_1s: float = 0.015,
        retreat_low_total_hp_shield: float = 0.45,
        heal_total_hp_shield: float = 0.65,
        vitals_confidence_min: float = 0.3,
        movement_score_threshold: float = 0.045,
        movement_release_score_threshold: float = 0.03,
        low_ground_confidence_min: float = 0.55,
        low_ground_confidence_off: float = 0.4,
        exposed_confidence_min: float = 0.65,
        exposed_confidence_off: float = 0.5,
        vitals_ema_alpha: float = 0.55,
        tactical_ema_alpha: float = 0.4,
        movement_ema_alpha: float = 0.45,
        max_history_seconds: float = 5.0,
    ) -> None:
        self.knock_recent_seconds = knock_recent_seconds
        self.under_fire_damage_1s = under_fire_damage_1s
        self.under_fire_release_damage_1s = under_fire_release_damage_1s
        self.retreat_low_total_hp_shield = retreat_low_total_hp_shield
        self.heal_total_hp_shield = heal_total_hp_shield
        self.vitals_confidence_min = vitals_confidence_min
        self.movement_score_threshold = movement_score_threshold
        self.movement_release_score_threshold = movement_release_score_threshold
        self.low_ground_confidence_min = low_ground_confidence_min
        self.low_ground_confidence_off = low_ground_confidence_off
        self.exposed_confidence_min = exposed_confidence_min
        self.exposed_confidence_off = exposed_confidence_off
        self.vitals_ema_alpha = max(0.0, min(1.0, float(vitals_ema_alpha)))
        self.tactical_ema_alpha = max(0.0, min(1.0, float(tactical_ema_alpha)))
        self.movement_ema_alpha = max(0.0, min(1.0, float(movement_ema_alpha)))
        self.max_history_seconds = max_history_seconds

        self._damage_events: deque[tuple[float, float]] = deque()
        self._enemy_knocks: deque[float] = deque()
        self._ally_knocks: deque[float] = deque()

        self._hp_pct = 1.0
        self._shield_pct = 1.0
        self._has_hp_sample = False
        self._has_shield_sample = False
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
        self._under_fire = False
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
            self._hp_pct = _ema_update(
                current=self._hp_pct,
                target=status.hp_pct,
                alpha=self.vitals_ema_alpha,
                has_sample=self._has_hp_sample,
            )
            self._has_hp_sample = True
        if status.shield_pct is not None:
            self._shield_pct = _ema_update(
                current=self._shield_pct,
                target=status.shield_pct,
                alpha=self.vitals_ema_alpha,
                has_sample=self._has_shield_sample,
            )
            self._has_shield_sample = True
        self._vitals_confidence = combine_vitals_confidence(
            hp_pct=status.hp_pct,
            shield_pct=status.shield_pct,
            hp_confidence=status.hp_confidence,
            shield_confidence=status.shield_confidence,
        )
        self._update_low_hp_streaks()
        if status.allies_alive is not None:
            self._allies_alive = status.allies_alive
        if status.allies_down is not None:
            self._allies_down = status.allies_down
        if tactical is not None:
            self._low_ground_confidence = _smoothed_confidence(
                current=self._low_ground_confidence,
                detected=tactical.low_ground_disadvantage,
                score=tactical.low_ground_confidence,
                alpha=self.tactical_ema_alpha,
            )
            self._low_ground_disadvantage = _apply_hysteresis(
                current=self._low_ground_disadvantage,
                score=self._low_ground_confidence,
                on_threshold=self.low_ground_confidence_min,
                off_threshold=self.low_ground_confidence_off,
            )

            self._exposed_confidence = _smoothed_confidence(
                current=self._exposed_confidence,
                detected=tactical.exposed_no_cover,
                score=tactical.exposed_confidence,
                alpha=self.tactical_ema_alpha,
            )
            self._exposed_no_cover = _apply_hysteresis(
                current=self._exposed_no_cover,
                score=self._exposed_confidence,
                on_threshold=self.exposed_confidence_min,
                off_threshold=self.exposed_confidence_off,
            )

            movement_target = max(0.0, min(1.0, float(tactical.movement_score)))
            if tactical.is_moving is True:
                movement_target = max(movement_target, self.movement_score_threshold)
            elif tactical.is_moving is False:
                movement_target = 0.0
            self._movement_score = _ema_update(
                current=self._movement_score,
                target=movement_target,
                alpha=self.movement_ema_alpha,
                has_sample=True,
            )
            self._is_moving = _apply_hysteresis(
                current=self._is_moving,
                score=self._movement_score,
                on_threshold=self.movement_score_threshold,
                off_threshold=self.movement_release_score_threshold,
            )

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
        self._under_fire = _apply_hysteresis(
            current=self._under_fire,
            score=recent_damage_1s,
            on_threshold=self.under_fire_damage_1s,
            off_threshold=self.under_fire_release_damage_1s,
        )

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
            under_fire=self._under_fire,
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


def _ema_update(current: float, target: float, alpha: float, has_sample: bool) -> float:
    clipped_target = max(0.0, min(1.0, float(target)))
    if not has_sample:
        return clipped_target
    mix = max(0.0, min(1.0, float(alpha)))
    return (mix * clipped_target) + ((1.0 - mix) * current)


def _smoothed_confidence(
    current: float,
    detected: bool | None,
    score: float,
    alpha: float,
) -> float:
    raw_score = max(0.0, min(1.0, float(score)))
    effective_alpha = alpha
    if detected is False:
        raw_score = 0.0
        effective_alpha = alpha * 0.5
    elif detected is None and raw_score <= 0.0:
        raw_score = 0.0
        effective_alpha = alpha * 0.5
    return _ema_update(
        current=current,
        target=raw_score,
        alpha=effective_alpha,
        has_sample=True,
    )


def _apply_hysteresis(
    current: bool,
    score: float,
    on_threshold: float,
    off_threshold: float,
) -> bool:
    value = max(0.0, float(score))
    if current:
        return value >= float(off_threshold)
    return value >= float(on_threshold)
