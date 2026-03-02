from __future__ import annotations

from pathlib import Path
from typing import Iterator

from apexcoach.models import FramePacket

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


class VideoCaptureService:
    def __init__(self, video_path: str | Path, target_fps: int = 10) -> None:
        self.video_path = str(Path(video_path).expanduser())
        self.target_fps = max(0, int(target_fps))
        self._cap = None
        self.source_fps: float = 0.0
        self.width: int = 0
        self.height: int = 0

    def __enter__(self) -> "VideoCaptureService":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for capture.")

        if self._cap is not None:
            return

        video_path = Path(self.video_path)
        if not video_path.exists():
            raise RuntimeError(
                "Could not open video because file does not exist: "
                f"{video_path.resolve()}"
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open video. Path exists but OpenCV failed to decode: "
                f"{video_path.resolve()}"
            )

        self._cap = cap
        self.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def iter_frames(self) -> Iterator[FramePacket]:
        if self._cap is None:
            self.open()

        assert self._cap is not None
        frame_index = 0
        target_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        next_emit_ts = 0.0

        while True:
            ok, frame = self._cap.read()
            if not ok:
                break

            position_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if position_ms > 0.0:
                timestamp = position_ms / 1000.0
            elif self.source_fps > 0.0:
                timestamp = frame_index / self.source_fps
            else:
                timestamp = float(frame_index)

            if target_interval > 0.0:
                if frame_index == 0:
                    next_emit_ts = timestamp
                if timestamp + 1e-9 < next_emit_ts:
                    frame_index += 1
                    continue
                while next_emit_ts <= timestamp + 1e-9:
                    next_emit_ts += target_interval

            yield FramePacket(frame_index=frame_index, timestamp=timestamp, frame=frame)
            frame_index += 1
