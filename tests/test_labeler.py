from apexcoach.aim_diagnosis.labeler import infer_labels, score_labels
from apexcoach.aim_diagnosis.models import AimLabel, ClipAnalysisMetrics


def test_infer_labels_identifies_tracking_and_close_range_instability() -> None:
    metrics = ClipAnalysisMetrics(
        time_to_first_shot_ms=340,
        aim_path_variance=0.6,
        tracking_error_score=0.8,
        recoil_error_score=0.25,
        close_range_score=0.7,
        ads_usage_score=0.45,
    )

    labels, confidence = infer_labels(metrics)

    assert labels[:2] == [
        AimLabel.TRACKING_DELAY,
        AimLabel.CLOSE_RANGE_INSTABILITY,
    ]
    assert confidence >= 0.6


def test_infer_labels_respects_allowed_labels_filter() -> None:
    metrics = ClipAnalysisMetrics(
        time_to_first_shot_ms=420,
        aim_path_variance=0.55,
        tracking_error_score=0.82,
        close_range_score=0.68,
    )

    labels, _confidence = infer_labels(
        metrics,
        allowed_labels=[AimLabel.SLOW_INITIAL_ADJUSTMENT, AimLabel.OVERFLICK],
    )

    assert labels == [AimLabel.SLOW_INITIAL_ADJUSTMENT]


def test_score_labels_can_trigger_ads_judgment_issue() -> None:
    metrics = ClipAnalysisMetrics(
        time_to_first_shot_ms=200,
        aim_path_variance=0.35,
        tracking_error_score=0.25,
        close_range_score=0.82,
        ads_usage_score=0.88,
    )

    scores = score_labels(metrics)

    assert scores[AimLabel.ADS_JUDGMENT_ISSUE] >= 0.65
