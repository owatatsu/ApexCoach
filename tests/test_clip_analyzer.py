from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from apexcoach.aim_diagnosis.clip_analyzer import ClipAnalyzerConfig, analyze_clip, analyze_clips
from apexcoach.aim_diagnosis.models import ClipRange


def _write_tracking_video(path: Path, *, fps: int = 20, frame_count: int = 40) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (160, 120))
    assert writer.isOpened()
    try:
        for idx in range(frame_count):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            x = 28 + idx * 2
            y = 48 + (idx % 3)
            cv2.rectangle(frame, (x, y), (x + 28, y + 28), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_analyze_clip_extracts_nonzero_metrics_from_tracking_video(tmp_path: Path) -> None:
    video_path = tmp_path / "tracking.mp4"
    _write_tracking_video(video_path)

    metrics = analyze_clip(
        video_path=video_path,
        clip=ClipRange(id="clip_1", video_id="vid_1", start_sec=0.0, end_sec=1.5),
        config=ClipAnalyzerConfig(sample_fps=10, min_frames=3),
    )

    assert metrics.time_to_first_shot_ms is not None
    assert metrics.close_range_score is not None
    assert metrics.close_range_score > 0.0
    assert metrics.aim_path_variance is not None


def test_analyze_clips_returns_map_for_multiple_clips(tmp_path: Path) -> None:
    video_path = tmp_path / "tracking_multi.mp4"
    _write_tracking_video(video_path)

    metrics_by_clip = analyze_clips(
        video_path=video_path,
        clips=[
            ClipRange(id="clip_1", video_id="vid_1", start_sec=0.0, end_sec=1.0),
            ClipRange(id="clip_2", video_id="vid_1", start_sec=0.5, end_sec=1.5),
        ],
        config=ClipAnalyzerConfig(sample_fps=10, min_frames=3),
    )

    assert set(metrics_by_clip.keys()) == {"clip_1", "clip_2"}
