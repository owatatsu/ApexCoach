from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
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


@dataclass(slots=True)
class VideoMetadata:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_sec: float = 0.0
    codec_name: str = ""


class VideoCaptureService:
    def __init__(self, video_path: str | Path, target_fps: int = 10) -> None:
        self.video_path = str(Path(video_path).expanduser())
        self.target_fps = max(0, int(target_fps))
        self._cap = None
        self._backend: str | None = None
        self._ffmpeg_proc: subprocess.Popen[bytes] | None = None
        self._frame_byte_size: int = 0
        self.source_fps: float = 0.0
        self.width: int = 0
        self.height: int = 0

    def __enter__(self) -> "VideoCaptureService":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if self._backend is not None:
            return

        video_path = Path(self.video_path)
        if not video_path.exists():
            raise RuntimeError(
                "Could not open video because file does not exist: "
                f"{video_path.resolve()}"
            )

        metadata = probe_video_metadata(video_path)
        prefer_ffmpeg = metadata.codec_name.strip().lower() == "av1"

        if prefer_ffmpeg and self._open_ffmpeg(video_path, metadata):
            return
        if self._open_cv2(video_path):
            return
        if self._open_ffmpeg(video_path, metadata):
            return

        raise RuntimeError(
            "Could not decode video with either OpenCV or ffmpeg. "
            f"Path: {video_path.resolve()}"
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.terminate()
            try:
                self._ffmpeg_proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                self._ffmpeg_proc.kill()
                self._ffmpeg_proc.communicate()
            self._ffmpeg_proc = None
        self._backend = None

    def iter_frames(self) -> Iterator[FramePacket]:
        if self._backend is None:
            self.open()

        if self._backend == "ffmpeg":
            yield from self._iter_ffmpeg_frames()
            return

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

    def _open_cv2(self, video_path: Path) -> bool:
        if cv2 is None:
            return False

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return False

        ok, frame = cap.read()
        if not ok or frame is None or getattr(frame, "size", 0) <= 0:
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._cap = cap
        self._backend = "cv2"
        self.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return True

    def _open_ffmpeg(self, video_path: Path, metadata: VideoMetadata) -> bool:
        if np is None:
            return False
        ffmpeg = _ffmpeg_binary()
        if ffmpeg is None:
            return False

        width = int(metadata.width)
        height = int(metadata.height)
        fps = float(metadata.fps)
        if width <= 0 or height <= 0:
            return False

        proc = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-an",
                "-sn",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-vsync",
                "0",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout is None:
            proc.kill()
            proc.communicate()
            return False

        self._ffmpeg_proc = proc
        self._backend = "ffmpeg"
        self.source_fps = fps
        self.width = width
        self.height = height
        self._frame_byte_size = width * height * 3
        return True

    def _iter_ffmpeg_frames(self) -> Iterator[FramePacket]:
        assert self._ffmpeg_proc is not None and self._ffmpeg_proc.stdout is not None
        frame_index = 0
        target_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        next_emit_ts = 0.0

        while True:
            raw = self._ffmpeg_proc.stdout.read(self._frame_byte_size)
            if len(raw) != self._frame_byte_size:
                break
            timestamp = (
                frame_index / self.source_fps if self.source_fps > 0.0 else float(frame_index)
            )

            if target_interval > 0.0:
                if frame_index == 0:
                    next_emit_ts = timestamp
                if timestamp + 1e-9 < next_emit_ts:
                    frame_index += 1
                    continue
                while next_emit_ts <= timestamp + 1e-9:
                    next_emit_ts += target_interval

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            ).copy()
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


def probe_video_metadata(video_path: str | Path) -> VideoMetadata:
    path = str(Path(video_path).expanduser())
    ffprobe = _ffprobe_binary()
    if ffprobe is not None:
        try:
            completed = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
                    "-of",
                    "json",
                    path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout or "{}")
            streams = payload.get("streams") or []
            stream = streams[0] if streams else {}
            format_info = payload.get("format") or {}
            fps = _parse_frame_rate(
                stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
            )
            duration_sec = _parse_float(stream.get("duration"))
            if duration_sec <= 0.0:
                duration_sec = _parse_float(format_info.get("duration"))
            return VideoMetadata(
                width=int(stream.get("width") or 0),
                height=int(stream.get("height") or 0),
                fps=fps,
                duration_sec=duration_sec,
                codec_name=str(stream.get("codec_name") or ""),
            )
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
            pass

    if cv2 is None:
        return VideoMetadata()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return VideoMetadata()

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration_sec = frame_count / fps if fps > 0.0 and frame_count > 0.0 else 0.0
        return VideoMetadata(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=fps,
            duration_sec=duration_sec,
        )
    finally:
        cap.release()


def _parse_frame_rate(value: str) -> float:
    text = str(value or "").strip()
    if not text or text == "0/0":
        return 0.0
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        num = _parse_float(numerator)
        den = _parse_float(denominator)
        if den <= 0.0:
            return 0.0
        return num / den
    return _parse_float(text)


def _parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ffmpeg_binary() -> str | None:
    return shutil.which("ffmpeg")


def _ffprobe_binary() -> str | None:
    return shutil.which("ffprobe")
