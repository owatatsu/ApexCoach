from apexcoach.vitals import combine_vitals_confidence


def test_combine_vitals_confidence_keeps_empty_bar_usable() -> None:
    confidence = combine_vitals_confidence(
        hp_pct=0.08,
        shield_pct=0.0,
        hp_confidence=0.0,
        shield_confidence=0.0,
    )
    assert confidence >= 0.3


def test_combine_vitals_confidence_averages_normal_values() -> None:
    confidence = combine_vitals_confidence(
        hp_pct=0.8,
        shield_pct=0.7,
        hp_confidence=0.9,
        shield_confidence=0.7,
    )
    assert 0.79 <= confidence <= 0.81
