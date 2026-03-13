from pathlib import Path

from apexcoach.capture_service import (
    VideoCaptureService,
    VideoMetadata,
    _parse_frame_rate,
)


def test_parse_frame_rate_fraction() -> None:
    assert round(_parse_frame_rate("30000/1001"), 3) == 29.97
    assert _parse_frame_rate("0/0") == 0.0
    assert _parse_frame_rate("") == 0.0


def test_video_capture_prefers_ffmpeg_for_av1(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"placeholder")
    calls: list[str] = []

    monkeypatch.setattr(
        "apexcoach.capture_service.probe_video_metadata",
        lambda path: VideoMetadata(width=1920, height=1080, fps=60.0, codec_name="av1"),
    )

    def fake_open_ffmpeg(self, path: Path, metadata: VideoMetadata) -> bool:
        calls.append("ffmpeg")
        self._backend = "ffmpeg"
        self.width = metadata.width
        self.height = metadata.height
        self.source_fps = metadata.fps
        return True

    def fake_open_cv2(self, path: Path) -> bool:
        calls.append("cv2")
        return False

    monkeypatch.setattr(VideoCaptureService, "_open_ffmpeg", fake_open_ffmpeg)
    monkeypatch.setattr(VideoCaptureService, "_open_cv2", fake_open_cv2)

    capture = VideoCaptureService(video_path)
    capture.open()

    assert capture._backend == "ffmpeg"
    assert calls == ["ffmpeg"]


def test_video_capture_falls_back_to_ffmpeg_when_cv2_fails(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"placeholder")
    calls: list[str] = []

    monkeypatch.setattr(
        "apexcoach.capture_service.probe_video_metadata",
        lambda path: VideoMetadata(width=1280, height=720, fps=30.0, codec_name="h264"),
    )

    def fake_open_ffmpeg(self, path: Path, metadata: VideoMetadata) -> bool:
        calls.append("ffmpeg")
        self._backend = "ffmpeg"
        self.width = metadata.width
        self.height = metadata.height
        self.source_fps = metadata.fps
        return True

    def fake_open_cv2(self, path: Path) -> bool:
        calls.append("cv2")
        return False

    monkeypatch.setattr(VideoCaptureService, "_open_ffmpeg", fake_open_ffmpeg)
    monkeypatch.setattr(VideoCaptureService, "_open_cv2", fake_open_cv2)

    capture = VideoCaptureService(video_path)
    capture.open()

    assert capture._backend == "ffmpeg"
    assert calls == ["cv2", "ffmpeg"]
