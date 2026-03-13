from __future__ import annotations


def combine_vitals_confidence(
    *,
    hp_pct: float | None,
    shield_pct: float | None,
    hp_confidence: float,
    shield_confidence: float,
) -> float:
    values: list[float] = []
    if hp_pct is not None:
        values.append(_effective_bar_confidence(hp_pct, hp_confidence))
    if shield_pct is not None:
        values.append(_effective_bar_confidence(shield_pct, shield_confidence))
    if not values:
        return 0.0
    return max(0.0, min(1.0, sum(values) / len(values)))


def _effective_bar_confidence(ratio: float, confidence: float) -> float:
    conf = max(0.0, min(1.0, float(confidence)))
    if conf > 0.0:
        return conf

    clipped_ratio = max(0.0, min(1.0, float(ratio)))
    if clipped_ratio <= 0.10:
        return 0.35
    return 0.0
