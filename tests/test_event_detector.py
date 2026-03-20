from apexcoach.event_detector import EventDetector
from apexcoach.models import ParsedNotifications, ParsedStatus


def test_small_damage_delta_is_ignored() -> None:
    det = EventDetector(vitals_confidence_min=0.3, min_damage_event_delta=0.05)
    notes = ParsedNotifications()
    det.detect(
        status=ParsedStatus(
            hp_pct=0.8,
            shield_pct=0.8,
            hp_confidence=0.9,
            shield_confidence=0.9,
        ),
        notifications=notes,
        timestamp=0.0,
    )
    ev = det.detect(
        status=ParsedStatus(
            hp_pct=0.79,
            shield_pct=0.8,
            hp_confidence=0.9,
            shield_confidence=0.9,
        ),
        notifications=notes,
        timestamp=0.1,
    )
    assert ev.damage_delta == 0.0


def test_medium_damage_is_emitted_immediately() -> None:
    det = EventDetector(
        vitals_confidence_min=0.3,
        min_damage_event_delta=0.04,
        damage_confirmation_frames=2,
    )
    notes = ParsedNotifications()

    det.detect(
        status=ParsedStatus(
            hp_pct=0.9,
            shield_pct=0.9,
            hp_confidence=0.9,
            shield_confidence=0.9,
        ),
        notifications=notes,
        timestamp=0.0,
    )
    ev = det.detect(
        status=ParsedStatus(
            hp_pct=0.86,
            shield_pct=0.9,
            hp_confidence=0.9,
            shield_confidence=0.9,
        ),
        notifications=notes,
        timestamp=0.1,
    )

    assert ev.damage_delta >= 0.04


def test_partial_damage_accumulates_within_short_window() -> None:
    det = EventDetector(
        vitals_confidence_min=0.3,
        min_damage_event_delta=0.04,
        damage_confirmation_frames=2,
        damage_confirmation_window_seconds=0.35,
        partial_damage_floor_ratio=0.45,
    )
    notes = ParsedNotifications()

    det.detect(
        status=ParsedStatus(
            hp_pct=0.9,
            shield_pct=0.9,
            hp_confidence=0.9,
            shield_confidence=0.9,
        ),
        notifications=notes,
        timestamp=0.0,
    )
    first = det.detect(
        status=ParsedStatus(
            hp_pct=0.878,
            shield_pct=0.9,
            hp_confidence=0.9,
            shield_confidence=0.9,
        ),
        notifications=notes,
        timestamp=0.1,
    )
    second = det.detect(
        status=ParsedStatus(
            hp_pct=0.856,
            shield_pct=0.9,
            hp_confidence=0.9,
            shield_confidence=0.9,
        ),
        notifications=notes,
        timestamp=0.2,
    )

    assert first.damage_delta == 0.0
    assert second.damage_delta >= 0.04


def test_large_damage_burst_is_emitted_immediately() -> None:
    det = EventDetector(
        vitals_confidence_min=0.3,
        min_damage_event_delta=0.04,
        damage_confirmation_frames=2,
        damage_burst_multiplier=2.0,
    )
    notes = ParsedNotifications()

    det.detect(
        status=ParsedStatus(
            hp_pct=0.9,
            shield_pct=0.9,
            hp_confidence=0.9,
            shield_confidence=0.9,
        ),
        notifications=notes,
        timestamp=0.0,
    )
    ev = det.detect(
        status=ParsedStatus(
            hp_pct=0.8,
            shield_pct=0.9,
            hp_confidence=0.9,
            shield_confidence=0.9,
        ),
        notifications=notes,
        timestamp=0.1,
    )

    assert ev.damage_delta >= 0.099
