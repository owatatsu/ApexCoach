from __future__ import annotations

from collections.abc import Iterable, Sequence

from apexcoach.aim_diagnosis.models import AimLabel, ClipAnalysisMetrics


def score_labels(
    metrics: ClipAnalysisMetrics,
    *,
    allowed_labels: Sequence[AimLabel | str] | None = None,
) -> dict[AimLabel, float]:
    allowed = _normalize_allowed_labels(allowed_labels)
    scores = {
        AimLabel.SLOW_INITIAL_ADJUSTMENT: _slow_initial_adjustment_score(metrics),
        AimLabel.OVERFLICK: _overflick_score(metrics),
        AimLabel.TRACKING_DELAY: _tracking_delay_score(metrics),
        AimLabel.RECOIL_BREAKDOWN: _recoil_breakdown_score(metrics),
        AimLabel.CLOSE_RANGE_INSTABILITY: _close_range_instability_score(metrics),
        AimLabel.ADS_JUDGMENT_ISSUE: _ads_judgment_issue_score(metrics),
    }
    if allowed is None:
        return {label: round(score, 3) for label, score in scores.items()}
    return {
        label: round(score, 3)
        for label, score in scores.items()
        if label in allowed
    }


def infer_labels(
    metrics: ClipAnalysisMetrics,
    *,
    allowed_labels: Sequence[AimLabel | str] | None = None,
    min_score: float = 0.55,
    max_labels: int = 3,
) -> tuple[list[AimLabel], float]:
    scores = score_labels(metrics, allowed_labels=allowed_labels)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
    selected = [label for label, score in ranked if score >= float(min_score)]
    selected = selected[: max(0, int(max_labels))]

    if not selected:
        fallback = ranked[0][1] if ranked else 0.0
        return [], round(max(0.2, min(0.49, fallback)), 3)

    selected_scores = [score for label, score in ranked if label in selected]
    confidence = sum(selected_scores) / len(selected_scores)
    return selected, round(max(0.0, min(1.0, confidence)), 3)


def _slow_initial_adjustment_score(metrics: ClipAnalysisMetrics) -> float:
    ms = float(metrics.time_to_first_shot_ms or 0)
    variance = float(metrics.aim_path_variance or 0.0)
    return _clamp01(((ms - 220.0) / 240.0) * 0.75 + variance * 0.25)


def _overflick_score(metrics: ClipAnalysisMetrics) -> float:
    variance = float(metrics.aim_path_variance or 0.0)
    tracking = float(metrics.tracking_error_score or 0.0)
    return _clamp01(max(0.0, variance - 0.48) * 1.2 + max(0.0, 0.55 - tracking) * 0.35)


def _tracking_delay_score(metrics: ClipAnalysisMetrics) -> float:
    tracking = float(metrics.tracking_error_score or 0.0)
    ms = float(metrics.time_to_first_shot_ms or 0)
    return _clamp01(tracking * 0.8 + max(0.0, (ms - 260.0) / 400.0) * 0.2)


def _recoil_breakdown_score(metrics: ClipAnalysisMetrics) -> float:
    recoil = float(metrics.recoil_error_score or 0.0)
    variance = float(metrics.aim_path_variance or 0.0)
    return _clamp01(recoil * 0.8 + variance * 0.2)


def _close_range_instability_score(metrics: ClipAnalysisMetrics) -> float:
    close = float(metrics.close_range_score or 0.0)
    variance = float(metrics.aim_path_variance or 0.0)
    return _clamp01(close * 0.65 + variance * 0.35)


def _ads_judgment_issue_score(metrics: ClipAnalysisMetrics) -> float:
    close = float(metrics.close_range_score or 0.0)
    ads = float(metrics.ads_usage_score or 0.0)
    return _clamp01(min(close, ads) * 0.8 + abs(close - ads) * 0.1)


def _normalize_allowed_labels(
    allowed_labels: Sequence[AimLabel | str] | None,
) -> set[AimLabel] | None:
    if allowed_labels is None:
        return None

    normalized: set[AimLabel] = set()
    for value in allowed_labels:
        label = _coerce_label(value)
        if label is not None:
            normalized.add(label)
    return normalized


def _coerce_label(value: AimLabel | str) -> AimLabel | None:
    if isinstance(value, AimLabel):
        return value
    try:
        return AimLabel(str(value))
    except ValueError:
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
