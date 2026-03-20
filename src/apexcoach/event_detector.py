from __future__ import annotations

from apexcoach.models import FrameEvents, ParsedNotifications, ParsedStatus
from apexcoach.vitals import combine_vitals_confidence


class EventDetector:
    def __init__(
        self,
        vitals_confidence_min: float = 0.6,
        min_damage_event_delta: float = 0.04,
        damage_confirmation_frames: int = 2,
        damage_burst_multiplier: float = 2.0,
        damage_confirmation_window_seconds: float = 0.35,
        partial_damage_floor_ratio: float = 0.45,
    ) -> None:
        self._prev_total: float | None = None
        self._vitals_confidence_min = vitals_confidence_min
        self._min_damage_event_delta = max(0.0, float(min_damage_event_delta))
        self._damage_confirmation_frames = max(1, int(damage_confirmation_frames))
        self._damage_burst_multiplier = max(1.0, float(damage_burst_multiplier))
        self._damage_confirmation_window_seconds = max(
            0.0,
            float(damage_confirmation_window_seconds),
        )
        self._partial_damage_floor = self._min_damage_event_delta * max(
            0.0,
            min(1.0, float(partial_damage_floor_ratio)),
        )
        self._pending_damage = 0.0
        self._pending_damage_frames = 0
        self._pending_damage_timestamp: float | None = None

    def detect(
        self,
        status: ParsedStatus,
        notifications: ParsedNotifications,
        timestamp: float,
    ) -> FrameEvents:
        damage_delta = 0.0
        current_total = _total_hp_shield(status)
        vitals_conf = combine_vitals_confidence(
            hp_pct=status.hp_pct,
            shield_pct=status.shield_pct,
            hp_confidence=status.hp_confidence,
            shield_confidence=status.shield_confidence,
        )
        reliable = vitals_conf >= self._vitals_confidence_min

        self._expire_pending_damage(timestamp)

        if reliable and current_total is not None and self._prev_total is not None:
            raw_drop = max(0.0, self._prev_total - current_total)
            burst_threshold = self._min_damage_event_delta * self._damage_burst_multiplier
            if raw_drop >= burst_threshold:
                damage_delta = raw_drop
                self._reset_pending_damage()
            elif raw_drop >= self._min_damage_event_delta:
                damage_delta = raw_drop
                self._reset_pending_damage()
            elif raw_drop >= self._partial_damage_floor:
                self._pending_damage += raw_drop
                self._pending_damage_frames += 1
                self._pending_damage_timestamp = timestamp
                if (
                    self._pending_damage >= self._min_damage_event_delta
                    or self._pending_damage_frames >= self._damage_confirmation_frames
                ):
                    damage_delta = self._pending_damage
                    self._reset_pending_damage()
            elif raw_drop <= 0.0:
                self._expire_pending_damage(timestamp, force=False)

        if reliable and current_total is not None:
            self._prev_total = current_total
        elif not reliable:
            self._reset_pending_damage()

        return FrameEvents(
            timestamp=timestamp,
            damage_delta=damage_delta,
            enemy_knock=notifications.enemy_knock,
            ally_knock=notifications.ally_knock,
        )

    def _reset_pending_damage(self) -> None:
        self._pending_damage = 0.0
        self._pending_damage_frames = 0
        self._pending_damage_timestamp = None

    def _expire_pending_damage(self, timestamp: float, force: bool = True) -> None:
        if force:
            if (
                self._pending_damage_timestamp is not None
                and timestamp - self._pending_damage_timestamp
                > self._damage_confirmation_window_seconds
            ):
                self._reset_pending_damage()
            return

        if (
            self._pending_damage_timestamp is not None
            and timestamp - self._pending_damage_timestamp
            > self._damage_confirmation_window_seconds
        ):
            self._reset_pending_damage()


def _total_hp_shield(status: ParsedStatus) -> float | None:
    if status.hp_pct is None or status.shield_pct is None:
        return None
    return status.hp_pct + status.shield_pct
