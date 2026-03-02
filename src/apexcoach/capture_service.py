from __future__ import annotations

from pathlib import Path
import time
from typing import Iterator

from apexcoach.models import FramePacket

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None

try:
    import mss
except ImportError:  # pragma: no cover - runtime dependency
    mss = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    np = None


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


class ScreenCaptureService:
    def __init__(
        self,
        target_fps: int = 30,
        monitor_index: int = 1,
        region: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.target_fps = max(0, int(target_fps))
        self.monitor_index = int(monitor_index)
        self.region = region
        self._sct = None
        self._grab_box: dict[str, int] | None = None
        self.source_fps: float = float(self.target_fps)
        self.width: int = 0
        self.height: int = 0

    def __enter__(self) -> "ScreenCaptureService":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if mss is None or np is None:
            raise RuntimeError("mss and numpy are required for realtime capture.")
        if self._sct is not None:
            return

        sct = mss.mss()
        monitors = sct.monitors
        if self.monitor_index <= 0 or self.monitor_index >= len(monitors):
            sct.close()
            raise RuntimeError(
                f"Invalid monitor index {self.monitor_index}. "
                f"Available monitor indexes: 1..{max(1, len(monitors)-1)}"
            )

        mon = monitors[self.monitor_index]
        if self.region is not None:
            x, y, w, h = self.region
            if w <= 0 or h <= 0:
                sct.close()
                raise RuntimeError("Realtime capture region width/height must be > 0.")
            grab_box = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
        else:
            grab_box = {
                "left": int(mon["left"]),
                "top": int(mon["top"]),
                "width": int(mon["width"]),
                "height": int(mon["height"]),
            }

        self._sct = sct
        self._grab_box = grab_box
        self.width = int(grab_box["width"])
        self.height = int(grab_box["height"])

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
            self._grab_box = None

    def iter_frames(self) -> Iterator[FramePacket]:
        if self._sct is None:
            self.open()

        assert self._sct is not None and self._grab_box is not None
        frame_index = 0
        interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        next_emit = 0.0
        t0 = time.perf_counter()

        while True:
            now = time.perf_counter() - t0
            if interval > 0.0 and now + 1e-9 < next_emit:
                time.sleep(min(0.005, next_emit - now))
                continue

            shot = self._sct.grab(self._grab_box)
            frame = np.asarray(shot)
            # mss returns BGRA, OpenCV expects BGR.
            bgr = frame[:, :, :3].copy()

            if interval > 0.0:
                while next_emit <= now + 1e-9:
                    next_emit += interval

            yield FramePacket(frame_index=frame_index, timestamp=now, frame=bgr)
            frame_index += 1
