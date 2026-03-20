from apexcoach.models import FrameEvents, ParsedStatus, ParsedTactical
from apexcoach.state_aggregator import StateAggregator


def test_under_fire_uses_hysteresis_release_threshold() -> None:
    agg = StateAggregator(
        under_fire_damage_1s=0.03,
        under_fire_release_damage_1s=0.015,
    )

    base_status = ParsedStatus(
        hp_pct=1.0,
        shield_pct=1.0,
        hp_confidence=1.0,
        shield_confidence=1.0,
    )
    tactical = ParsedTactical()

    state = agg.update(
        status=base_status,
        events=FrameEvents(timestamp=0.0, damage_delta=0.04),
        tactical=tactical,
    )
    assert state.under_fire is True

    state = agg.update(
        status=base_status,
        events=FrameEvents(timestamp=0.6, damage_delta=0.0),
        tactical=tactical,
    )
    assert state.under_fire is True

    state = agg.update(
        status=base_status,
        events=FrameEvents(timestamp=1.2, damage_delta=0.0),
        tactical=tactical,
    )
    assert state.under_fire is False


def test_low_ground_confidence_is_smoothed_with_hysteresis() -> None:
    agg = StateAggregator(
        low_ground_confidence_min=0.55,
        low_ground_confidence_off=0.4,
        tactical_ema_alpha=0.5,
    )
    status = ParsedStatus(
        hp_pct=1.0,
        shield_pct=1.0,
        hp_confidence=1.0,
        shield_confidence=1.0,
    )

    first = agg.update(
        status=status,
        events=FrameEvents(timestamp=0.0),
        tactical=ParsedTactical(
            low_ground_disadvantage=True,
            low_ground_confidence=0.9,
        ),
    )
    second = agg.update(
        status=status,
        events=FrameEvents(timestamp=0.1),
        tactical=ParsedTactical(
            low_ground_disadvantage=True,
            low_ground_confidence=0.9,
        ),
    )
    third = agg.update(
        status=status,
        events=FrameEvents(timestamp=0.2),
        tactical=ParsedTactical(
            low_ground_disadvantage=False,
            low_ground_confidence=0.0,
        ),
    )

    assert first.low_ground_disadvantage is False
    assert second.low_ground_disadvantage is True
    assert third.low_ground_disadvantage is True


def test_vitals_are_smoothed_before_low_hp_logic() -> None:
    agg = StateAggregator(
        retreat_low_total_hp_shield=0.45,
        heal_total_hp_shield=0.65,
        vitals_ema_alpha=0.5,
        vitals_confidence_min=0.3,
    )

    high = ParsedStatus(
        hp_pct=1.0,
        shield_pct=1.0,
        hp_confidence=1.0,
        shield_confidence=1.0,
    )
    low = ParsedStatus(
        hp_pct=0.1,
        shield_pct=0.1,
        hp_confidence=1.0,
        shield_confidence=1.0,
    )

    agg.update(status=high, events=FrameEvents(timestamp=0.0), tactical=ParsedTactical())
    state = agg.update(
        status=low,
        events=FrameEvents(timestamp=0.1),
        tactical=ParsedTactical(),
    )

    assert state.hp_pct > 0.1
    assert state.shield_pct > 0.1
    assert state.heal_low_hp_streak == 0
