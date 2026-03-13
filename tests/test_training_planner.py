from apexcoach.aim_diagnosis.models import AimLabel, ClipDiagnosis
from apexcoach.aim_diagnosis.training_planner import (
    build_training_plan,
    build_training_plan_from_label_counts,
)


def test_build_training_plan_picks_top_two_labels() -> None:
    diagnoses = [
        ClipDiagnosis(
            clip_id="clip_1",
            summary="tracking issue",
            labels=[AimLabel.TRACKING_DELAY, AimLabel.CLOSE_RANGE_INSTABILITY],
        ),
        ClipDiagnosis(
            clip_id="clip_2",
            summary="tracking again",
            labels=[AimLabel.TRACKING_DELAY],
        ),
        ClipDiagnosis(
            clip_id="clip_3",
            summary="overflick once",
            labels=[AimLabel.OVERFLICK],
        ),
    ]

    plan = build_training_plan(video_id="vid_1", clip_diagnoses=diagnoses)

    assert plan.video_id == "vid_1"
    assert plan.priority_labels == [
        AimLabel.TRACKING_DELAY,
        AimLabel.CLOSE_RANGE_INSTABILITY,
    ]
    assert [drill.label for drill in plan.drills] == plan.priority_labels
    assert plan.total_minutes == 6


def test_build_training_plan_from_label_counts_accepts_string_keys() -> None:
    plan = build_training_plan_from_label_counts(
        video_id="vid_2",
        label_counts={
            "slow_initial_adjustment": 2,
            "overflick": 2,
            "unknown_label": 99,
        },
    )

    assert plan.priority_labels == [
        AimLabel.SLOW_INITIAL_ADJUSTMENT,
        AimLabel.OVERFLICK,
    ]
    assert [drill.title for drill in plan.drills] == [
        "Reaction + micro flick",
        "Micro flick stop control",
    ]


def test_build_training_plan_dedupes_labels_within_single_clip() -> None:
    diagnoses = [
        ClipDiagnosis(
            clip_id="clip_1",
            summary="duplicate labels",
            labels=[
                AimLabel.RECOIL_BREAKDOWN,
                AimLabel.RECOIL_BREAKDOWN,
                AimLabel.ADS_JUDGMENT_ISSUE,
            ],
        )
    ]

    plan = build_training_plan(video_id="vid_3", clip_diagnoses=diagnoses)

    assert plan.priority_labels == [
        AimLabel.RECOIL_BREAKDOWN,
        AimLabel.ADS_JUDGMENT_ISSUE,
    ]


def test_build_training_plan_empty_input_returns_empty_plan() -> None:
    plan = build_training_plan(video_id="vid_4", clip_diagnoses=[])

    assert plan.priority_labels == []
    assert plan.drills == []
    assert plan.total_minutes == 0
