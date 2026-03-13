from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from apexcoach.aim_diagnosis.models import AimLabel, ClipDiagnosis, TrainingDrill, TrainingPlan


_LABEL_PRIORITY = {
    AimLabel.TRACKING_DELAY: 0,
    AimLabel.CLOSE_RANGE_INSTABILITY: 1,
    AimLabel.SLOW_INITIAL_ADJUSTMENT: 2,
    AimLabel.OVERFLICK: 3,
    AimLabel.RECOIL_BREAKDOWN: 4,
    AimLabel.ADS_JUDGMENT_ISSUE: 5,
}

_DRILL_LIBRARY: dict[AimLabel, TrainingDrill] = {
    AimLabel.SLOW_INITIAL_ADJUSTMENT: TrainingDrill(
        label=AimLabel.SLOW_INITIAL_ADJUSTMENT,
        title="Reaction + micro flick",
        minutes=3,
        instruction="Practice quick target acquisition with small corrections.",
        check_point="Avoid a large first swing after spotting the target.",
    ),
    AimLabel.OVERFLICK: TrainingDrill(
        label=AimLabel.OVERFLICK,
        title="Micro flick stop control",
        minutes=3,
        instruction="Focus on stopping on target instead of snapping past it.",
        check_point="Prioritize stop accuracy over flick speed.",
    ),
    AimLabel.TRACKING_DELAY: TrainingDrill(
        label=AimLabel.TRACKING_DELAY,
        title="Mid-range tracking",
        minutes=3,
        instruction="Track steady lateral movement without trailing behind.",
        check_point="Trace slightly ahead of the target instead of chasing it.",
    ),
    AimLabel.RECOIL_BREAKDOWN: TrainingDrill(
        label=AimLabel.RECOIL_BREAKDOWN,
        title="Sustained recoil control",
        minutes=3,
        instruction="Keep the spray stable after the first few bullets.",
        check_point="Watch for late-spray instability instead of only the opening burst.",
    ),
    AimLabel.CLOSE_RANGE_INSTABILITY: TrainingDrill(
        label=AimLabel.CLOSE_RANGE_INSTABILITY,
        title="Close-range hipfire stability",
        minutes=3,
        instruction="Keep the crosshair stable at chest level during close fights.",
        check_point="Do not overreact to every strafe at close range.",
    ),
    AimLabel.ADS_JUDGMENT_ISSUE: TrainingDrill(
        label=AimLabel.ADS_JUDGMENT_ISSUE,
        title="ADS vs hipfire review",
        minutes=2,
        instruction="Review close-range fights and favor hipfire in cramped duels.",
        check_point="Check whether ADS is slowing your close-range tracking.",
    ),
}


def build_training_plan(
    *,
    video_id: str,
    clip_diagnoses: Sequence[ClipDiagnosis],
    max_priority_labels: int = 2,
    max_drills: int = 3,
) -> TrainingPlan:
    label_counts: Counter[AimLabel] = Counter()
    for diagnosis in clip_diagnoses:
        label_counts.update(_dedupe_labels(diagnosis.labels))
    return build_training_plan_from_label_counts(
        video_id=video_id,
        label_counts=label_counts,
        max_priority_labels=max_priority_labels,
        max_drills=max_drills,
    )


def build_training_plan_from_label_counts(
    *,
    video_id: str,
    label_counts: Mapping[AimLabel, int] | Mapping[str, int],
    max_priority_labels: int = 2,
    max_drills: int = 3,
) -> TrainingPlan:
    normalized_counts = _normalize_label_counts(label_counts)
    ranked_labels = _rank_labels(normalized_counts, max_priority_labels=max_priority_labels)
    drills = [_copy_drill(_DRILL_LIBRARY[label]) for label in ranked_labels[:max_drills]]
    return TrainingPlan(
        video_id=video_id,
        priority_labels=ranked_labels,
        drills=drills,
        total_minutes=sum(drill.minutes for drill in drills),
    )


def _normalize_label_counts(
    label_counts: Mapping[AimLabel, int] | Mapping[str, int],
) -> Counter[AimLabel]:
    normalized: Counter[AimLabel] = Counter()
    for raw_label, raw_count in label_counts.items():
        if raw_count <= 0:
            continue
        label = _coerce_label(raw_label)
        if label is None:
            continue
        normalized[label] += int(raw_count)
    return normalized


def _rank_labels(
    label_counts: Mapping[AimLabel, int],
    *,
    max_priority_labels: int,
) -> list[AimLabel]:
    limit = max(0, int(max_priority_labels))
    if limit == 0:
        return []

    ranked = sorted(
        label_counts.items(),
        key=lambda item: (-item[1], _LABEL_PRIORITY.get(item[0], 999), item[0].value),
    )
    return [label for label, _count in ranked[:limit]]


def _dedupe_labels(labels: Iterable[AimLabel]) -> list[AimLabel]:
    seen: set[AimLabel] = set()
    ordered: list[AimLabel] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def _coerce_label(value: AimLabel | str) -> AimLabel | None:
    if isinstance(value, AimLabel):
        return value
    try:
        return AimLabel(str(value))
    except ValueError:
        return None


def _copy_drill(drill: TrainingDrill) -> TrainingDrill:
    return TrainingDrill(
        label=drill.label,
        title=drill.title,
        minutes=drill.minutes,
        instruction=drill.instruction,
        check_point=drill.check_point,
    )
