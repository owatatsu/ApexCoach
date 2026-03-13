from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from apexcoach.aim_diagnosis.models import AimLabel, ClipAnalysisMetrics, ClipDiagnosis


@dataclass(frozen=True, slots=True)
class _CommentTemplate:
    summary: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    next_focus: str


_GENERIC_TEMPLATE = _CommentTemplate(
    summary="This clip shows a few aim issues, but the signal is limited.",
    strengths=("The engagement still contains usable aim data.",),
    weaknesses=("The main weakness is not yet clear from this clip alone.",),
    next_focus="Review whether the opening aim or mid-fight tracking breaks first.",
)

_TEMPLATES: dict[AimLabel, _CommentTemplate] = {
    AimLabel.SLOW_INITIAL_ADJUSTMENT: _CommentTemplate(
        summary="The first adjustment onto the target looks a bit slow.",
        strengths=("Once the fight starts, your aim does not fully collapse.",),
        weaknesses=(
            "Your first correction takes an extra step before settling.",
            "The opening crosshair move is larger than necessary.",
        ),
        next_focus="Move onto the target in one clean adjustment instead of one large swing.",
    ),
    AimLabel.OVERFLICK: _CommentTemplate(
        summary="You tend to overshoot the target on the first correction.",
        strengths=("Your raw reaction speed looks decent.",),
        weaknesses=(
            "The crosshair passes the target before settling.",
            "Stopping control is weaker than the initial snap.",
        ),
        next_focus="Prioritize stopping on target over snapping faster.",
    ),
    AimLabel.TRACKING_DELAY: _CommentTemplate(
        summary="Your tracking falls slightly behind the target during the fight.",
        strengths=("The opening acquisition is relatively quick.",),
        weaknesses=(
            "You are following the enemy instead of matching their movement early.",
            "Tracking consistency drops once the target keeps strafing.",
        ),
        next_focus="Trace slightly ahead of the target instead of chasing their current position.",
    ),
    AimLabel.RECOIL_BREAKDOWN: _CommentTemplate(
        summary="Spray control becomes unstable as the burst continues.",
        strengths=("The first few bullets are more controlled than the rest.",),
        weaknesses=(
            "The latter half of the spray spreads too much.",
            "You are spending more effort correcting than maintaining the spray.",
        ),
        next_focus="Watch the second half of the spray and keep the pattern controlled longer.",
    ),
    AimLabel.CLOSE_RANGE_INSTABILITY: _CommentTemplate(
        summary="Your aim gets unstable in close-range movement.",
        strengths=("Target acquisition itself is still fairly quick.",),
        weaknesses=(
            "Close-range strafes are pulling your crosshair around too much.",
            "Your correction count rises sharply once the target is nearby.",
        ),
        next_focus="Keep the crosshair calmer at chest level and avoid overreacting to every strafe.",
    ),
    AimLabel.ADS_JUDGMENT_ISSUE: _CommentTemplate(
        summary="Your ADS choice may be making close-range aim harder.",
        strengths=("The distance read is not completely off.",),
        weaknesses=(
            "ADS appears to make tracking feel tighter than necessary in this range.",
            "Hipfire would likely give a steadier correction window here.",
        ),
        next_focus="In close fights, check whether hipfire would keep your tracking steadier.",
    ),
}


def build_clip_diagnosis(
    *,
    clip_id: str,
    labels: Sequence[AimLabel | str],
    confidence: float,
    metrics: ClipAnalysisMetrics | None = None,
) -> ClipDiagnosis:
    normalized_labels = _normalize_labels(labels)
    primary_label = normalized_labels[0] if normalized_labels else None
    primary_template = _TEMPLATES.get(primary_label, _GENERIC_TEMPLATE)

    strengths = list(primary_template.strengths[:1])
    weaknesses = _merge_weaknesses(normalized_labels)
    summary = _hedge_summary(primary_template.summary, confidence)
    next_focus = primary_template.next_focus

    return ClipDiagnosis(
        clip_id=clip_id,
        summary=summary,
        strengths=strengths,
        weaknesses=weaknesses,
        labels=normalized_labels,
        confidence=max(0.0, min(1.0, float(confidence))),
        next_focus=next_focus,
        metrics=metrics or ClipAnalysisMetrics(),
    )


def build_clip_diagnoses(
    *,
    clip_labels: dict[str, Sequence[AimLabel | str]],
    confidence_by_clip: dict[str, float] | None = None,
    metrics_by_clip: dict[str, ClipAnalysisMetrics] | None = None,
) -> list[ClipDiagnosis]:
    confidence_by_clip = confidence_by_clip or {}
    metrics_by_clip = metrics_by_clip or {}
    diagnoses: list[ClipDiagnosis] = []
    for clip_id, labels in clip_labels.items():
        diagnoses.append(
            build_clip_diagnosis(
                clip_id=clip_id,
                labels=labels,
                confidence=confidence_by_clip.get(clip_id, 0.65),
                metrics=metrics_by_clip.get(clip_id),
            )
        )
    return diagnoses


def _normalize_labels(labels: Iterable[AimLabel | str]) -> list[AimLabel]:
    out: list[AimLabel] = []
    seen: set[AimLabel] = set()
    for raw in labels:
        label = _coerce_label(raw)
        if label is None or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _coerce_label(value: AimLabel | str) -> AimLabel | None:
    if isinstance(value, AimLabel):
        return value
    try:
        return AimLabel(str(value))
    except ValueError:
        return None


def _merge_weaknesses(labels: Sequence[AimLabel]) -> list[str]:
    if not labels:
        return list(_GENERIC_TEMPLATE.weaknesses)

    merged: list[str] = []
    seen: set[str] = set()
    for label in labels[:2]:
        template = _TEMPLATES[label]
        for weakness in template.weaknesses:
            if weakness in seen:
                continue
            seen.add(weakness)
            merged.append(weakness)
            if len(merged) >= 2:
                return merged
    return merged


def _hedge_summary(summary: str, confidence: float) -> str:
    score = max(0.0, min(1.0, float(confidence)))
    if score >= 0.75:
        return summary
    if score >= 0.5:
        return f"Likely: {summary}"
    return f"Possible: {summary}"
