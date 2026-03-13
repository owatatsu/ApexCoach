from apexcoach.aim_diagnosis.models import AimLabel, ClipRange
from apexcoach.aim_diagnosis.models import ClipAnalysisMetrics
from apexcoach.aim_diagnosis.service import AimDiagnosisService


def test_service_registers_video_and_clips() -> None:
    service = AimDiagnosisService()
    video = service.register_video(file_path="match.mp4", duration_sec=120.0, video_id="vid_1")

    clips = service.save_clips(
        video_id=video.id,
        clips=[
            ClipRange(id="clip_1", video_id=video.id, start_sec=10.0, end_sec=18.5),
            ClipRange(id="clip_2", video_id=video.id, start_sec=40.0, end_sec=49.0),
        ],
    )

    assert video.id == "vid_1"
    assert [clip.id for clip in clips] == ["clip_1", "clip_2"]


def test_service_builds_results_and_training_plan() -> None:
    service = AimDiagnosisService()
    video = service.register_video(file_path="match.mp4", duration_sec=120.0, video_id="vid_2")
    service.save_clips(
        video_id=video.id,
        clips=[
            ClipRange(id="clip_1", video_id=video.id, start_sec=10.0, end_sec=18.5),
            ClipRange(id="clip_2", video_id=video.id, start_sec=40.0, end_sec=49.0),
        ],
    )
    service.start_analysis(
        video_id=video.id,
        labels=[AimLabel.TRACKING_DELAY, AimLabel.CLOSE_RANGE_INSTABILITY],
        job_id="job_1",
    )
    service.build_and_save_diagnoses(
        video_id=video.id,
        clip_labels={
            "clip_1": [AimLabel.TRACKING_DELAY],
            "clip_2": [AimLabel.TRACKING_DELAY, AimLabel.CLOSE_RANGE_INSTABILITY],
        },
        confidence_by_clip={"clip_1": 0.8, "clip_2": 0.66},
    )

    result = service.get_results(video_id=video.id)

    assert result["status"] == "completed"
    assert len(result["clips"]) == 2
    assert result["clips"][0]["labels"] == ["tracking_delay"]
    assert result["training_plan"]["priority_labels"] == [
        "tracking_delay",
        "close_range_instability",
    ]
    assert result["training_plan"]["total_minutes"] == 6


def test_service_analyze_saved_clips_uses_labeler_flow(monkeypatch) -> None:
    service = AimDiagnosisService()
    video = service.register_video(file_path="match.mp4", duration_sec=120.0, video_id="vid_3")
    service.save_clips(
        video_id=video.id,
        clips=[ClipRange(id="clip_1", video_id=video.id, start_sec=10.0, end_sec=18.5)],
    )
    service.start_analysis(
        video_id=video.id,
        labels=[AimLabel.TRACKING_DELAY, AimLabel.CLOSE_RANGE_INSTABILITY],
        job_id="job_2",
    )

    def _fake_analyze_clips(*, video_path, clips, config):
        assert video_path == "match.mp4"
        assert len(clips) == 1
        return {
            "clip_1": ClipAnalysisMetrics(
                time_to_first_shot_ms=320,
                aim_path_variance=0.62,
                tracking_error_score=0.84,
                close_range_score=0.71,
                ads_usage_score=0.4,
            )
        }

    monkeypatch.setattr("apexcoach.aim_diagnosis.service.analyze_clips", _fake_analyze_clips)
    diagnoses = service.analyze_saved_clips(video_id=video.id)
    result = service.get_results(video_id=video.id)

    assert diagnoses[0].labels[0] == AimLabel.TRACKING_DELAY
    assert result["status"] == "completed"
    assert result["clips"][0]["labels"][0] == "tracking_delay"
