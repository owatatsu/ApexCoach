from __future__ import annotations

from apexcoach.models import FrameEvents, ParsedNotifications, ParsedStatus


class EventDetector:
    def __init__(
        self,
        vitals_confidence_min: float = 0.6,
        min_damage_event_delta: float = 0.04,
    ) -> None:
        self._prev_total: float | None = None
        self._vitals_confidence_min = vitals_confidence_min
        self._min_damage_event_delta = max(0.0, float(min_damage_event_delta))

    def detect(
        self,
        status: ParsedStatus,
        notifications: ParsedNotifications,
        timestamp: float,
    ) -> FrameEvents:
        damage_delta = 0.0
        current_total = _total_hp_shield(status)
        vitals_conf = min(status.hp_confidence, status.shield_confidence)
        reliable = vitals_conf >= self._vitals_confidence_min

        if reliable and current_total is not None and self._prev_total is not None:
            damage_delta = max(0.0, self._prev_total - current_total)
            if damage_delta < self._min_damage_event_delta:
                damage_delta = 0.0

        if reliable and current_total is not None:
            self._prev_total = current_total

        return FrameEvents(
            timestamp=timestamp,
            damage_delta=damage_delta,
            enemy_knock=notifications.enemy_knock,
            ally_knock=notifications.ally_knock,
        )


def _total_hp_shield(status: ParsedStatus) -> float | None:
    if status.hp_pct is None or status.shield_pct is None:
        return None
    return status.hp_pct + status.shield_pct
