from pathlib import Path
from types import SimpleNamespace

import pytest

from apexcoach.capture_service import VideoCaptureService
from apexcoach.config import YoloConfig
from apexcoach.enemy_detector import YoloEnemyDetector

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor([[10, 12, 50, 70], [120, 20, 180, 90]])
        self.conf = _FakeTensor([0.9, 0.8])
        self.cls = _FakeTensor([0, 0])
        self.id = _FakeTensor([3, 7])


class _FakeResult:
    def __init__(self):
        self.boxes = _FakeBoxes()
        self.names = {0: "person"}


class _FakeYOLO:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def track(self, **kwargs):
        return [_FakeResult()]

    def predict(self, **kwargs):
        return [_FakeResult()]


def test_yolo_detector_returns_summary_from_static_frame(monkeypatch) -> None:
    monkeypatch.setattr("apexcoach.enemy_detector.YOLO", _FakeYOLO)
    detector = YoloEnemyDetector(
        YoloConfig(enabled=True, model_name="fake.pt", track_enabled=True)
    )
    frame = np.zeros((120, 200, 3), dtype=np.uint8)

    state = detector.infer(frame, timestamp=0.1, color_format="bgr")

    assert state.available is True
    assert state.enemy_count == 2
    assert state.enemy_left == 1
    assert state.enemy_center == 0
    assert state.enemy_right == 1
    assert state.tracked_enemy_ids == [3, 7]
    assert "enemy_count=2" in state.summary_lines


def test_yolo_detector_can_process_video_frames(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("apexcoach.enemy_detector.YOLO", _FakeYOLO)
    detector = YoloEnemyDetector(
        YoloConfig(enabled=True, model_name="fake.pt", track_enabled=False)
    )
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (160, 90),
    )
    for _ in range(3):
        writer.write(np.zeros((90, 160, 3), dtype=np.uint8))
    writer.release()

    enemy_counts: list[int] = []
    with VideoCaptureService(video_path=video_path, target_fps=5) as capture:
        for packet in capture.iter_frames():
            state = detector.infer(packet.frame, timestamp=packet.timestamp, color_format="bgr")
            enemy_counts.append(state.enemy_count)

    assert enemy_counts
    assert all(count == 2 for count in enemy_counts)


def test_yolo_detector_disabled_keeps_default_state() -> None:
    detector = YoloEnemyDetector(YoloConfig(enabled=False))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    state = detector.infer(frame, timestamp=0.0)

    assert state.available is False
    assert state.enemy_count == 0
    assert state.summary_lines == []
