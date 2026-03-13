from apexcoach.aim_diagnosis.comment_builder import build_clip_diagnosis, build_clip_diagnoses
from apexcoach.aim_diagnosis.models import AimLabel, ClipAnalysisMetrics


def test_build_clip_diagnosis_uses_primary_label_template() -> None:
    diagnosis = build_clip_diagnosis(
        clip_id="clip_1",
        labels=[AimLabel.TRACKING_DELAY, AimLabel.CLOSE_RANGE_INSTABILITY],
        confidence=0.82,
        metrics=ClipAnalysisMetrics(tracking_error_score=0.66),
    )

    assert diagnosis.clip_id == "clip_1"
    assert diagnosis.summary == "Your tracking falls slightly behind the target during the fight."
    assert diagnosis.labels == [
        AimLabel.TRACKING_DELAY,
        AimLabel.CLOSE_RANGE_INSTABILITY,
    ]
    assert len(diagnosis.weaknesses) == 2
    assert diagnosis.metrics.tracking_error_score == 0.66


def test_build_clip_diagnosis_hedges_low_confidence_summary() -> None:
    diagnosis = build_clip_diagnosis(
        clip_id="clip_2",
        labels=[AimLabel.OVERFLICK],
        confidence=0.41,
    )

    assert diagnosis.summary.startswith("Possible: ")


def test_build_clip_diagnoses_accepts_string_labels() -> None:
    diagnoses = build_clip_diagnoses(
        clip_labels={
            "clip_1": ["slow_initial_adjustment"],
            "clip_2": ["ads_judgment_issue"],
        },
        confidence_by_clip={"clip_1": 0.7},
    )

    assert [d.clip_id for d in diagnoses] == ["clip_1", "clip_2"]
    assert diagnoses[0].labels == [AimLabel.SLOW_INITIAL_ADJUSTMENT]
    assert diagnoses[0].summary.startswith("Likely: ")
