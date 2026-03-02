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
