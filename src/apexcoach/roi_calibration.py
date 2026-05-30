from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from apexcoach.capture_service import VideoCaptureService
from apexcoach.config import ApexCoachConfig, Roi
from apexcoach.roi_manager import RoiManager

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


DEFAULT_CALIBRATION_ROIS = ("hp_bar", "shield_bar", "teammate_panel", "kill_feed")


@dataclass(slots=True)
class RoiCalibrationResult:
    frame_width: int
    frame_height: int
    rois: dict[str, Roi]


def calibrate_rois_from_video(
    *,
    video_path: str | Path,
    config: ApexCoachConfig,
    roi_names: Sequence[str] = DEFAULT_CALIBRATION_ROIS,
    frame_sec: float = 0.0,
    output_path: str | Path,
    snapshot_path: str | Path | None = None,
) -> RoiCalibrationResult:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for ROI calibration.")

    frame = _read_calibration_frame(video_path=video_path, frame_sec=frame_sec)
    result = calibrate_rois_from_frame(
        frame=frame,
        config=config,
        roi_names=roi_names,
    )
    write_roi_config(output_path, result)
    if snapshot_path:
        write_roi_snapshot(snapshot_path, frame=frame, result=result)
    return result


def calibrate_rois_from_frame(
    *,
    frame,
    config: ApexCoachConfig,
    roi_names: Sequence[str] = DEFAULT_CALIBRATION_ROIS,
) -> RoiCalibrationResult:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for ROI calibration.")
    if frame is None or getattr(frame, "size", 0) <= 0:
        raise ValueError("Calibration frame is empty.")

    frame_h, frame_w = frame.shape[:2]
    existing_boxes = RoiManager(
        config.rois,
        scale_to_frame=config.scale_rois_to_frame,
        reference_width=config.roi_reference_width,
        reference_height=config.roi_reference_height,
    ).resolve_boxes(frame)

    calibrated: dict[str, Roi] = {}
    names = [name.strip() for name in roi_names if name.strip()]
    if not names:
        raise ValueError("At least one ROI name is required.")

    for name in names:
        preview = _draw_existing_box(frame, name=name, boxes=existing_boxes)
        rect = _select_roi(window_name=f"ApexCoach ROI: {name}", frame=preview)
        x, y, w, h = [int(round(v)) for v in rect]
        if w <= 0 or h <= 0:
            fallback = existing_boxes.get(name)
            if fallback is None:
                continue
            x1, y1, x2, y2 = fallback
            calibrated[name] = Roi(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
            continue
        calibrated[name] = Roi(
            x=max(0, min(frame_w - 1, x)),
            y=max(0, min(frame_h - 1, y)),
            w=max(1, min(frame_w - x, w)),
            h=max(1, min(frame_h - y, h)),
        )

    if not calibrated:
        raise RuntimeError("No ROIs were selected.")
    return RoiCalibrationResult(frame_width=frame_w, frame_height=frame_h, rois=calibrated)


def _select_roi(*, window_name: str, frame) -> tuple[int, int, int, int]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for ROI calibration.")

    state = {
        "dragging": False,
        "start": (0, 0),
        "end": (0, 0),
        "rect": (0, 0, 0, 0),
        "done": False,
    }

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["start"] = (int(x), int(y))
            state["end"] = (int(x), int(y))
            state["rect"] = (0, 0, 0, 0)
            return
        if event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = (int(x), int(y))
            return
        if event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            state["end"] = (int(x), int(y))
            state["rect"] = _normalize_drag_rect(state["start"], state["end"])
            if state["rect"][2] > 0 and state["rect"][3] > 0:
                state["done"] = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            canvas = frame.copy()
            rect = state["rect"]
            if state["dragging"]:
                rect = _normalize_drag_rect(state["start"], state["end"])
            if rect[2] > 0 and rect[3] > 0:
                x, y, w, h = rect
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 220, 255), 2)
            cv2.putText(
                canvas,
                "Drag ROI and release mouse. Press c or Esc to skip.",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (60, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if state["done"]:
                return state["rect"]
            if key in (ord("c"), 27):
                return (0, 0, 0, 0)
            if key in (13, 10, 32) and state["rect"][2] > 0 and state["rect"][3] > 0:
                return state["rect"]
    finally:
        cv2.destroyWindow(window_name)


def _normalize_drag_rect(
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1 = start
    x2, y2 = end
    x = min(x1, x2)
    y = min(y1, y2)
    return (x, y, abs(x2 - x1), abs(y2 - y1))


def write_roi_config(path: str | Path, result: RoiCalibrationResult) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_roi_config(result), encoding="utf-8")


def render_roi_config(result: RoiCalibrationResult) -> str:
    lines = [
        "# Generated by apexcoach --calibrate-rois",
        "scale_rois_to_frame: true",
        f"roi_reference_width: {int(result.frame_width)}",
        f"roi_reference_height: {int(result.frame_height)}",
        "rois:",
    ]
    for name in sorted(result.rois):
        roi = result.rois[name]
        lines.append(f"  {name}: [{roi.x}, {roi.y}, {roi.w}, {roi.h}]")
    return "\n".join(lines) + "\n"


def write_roi_snapshot(
    path: str | Path,
    *,
    frame,
    result: RoiCalibrationResult,
) -> None:
    if cv2 is None:
        return
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    boxes = {
        name: (roi.x, roi.y, roi.x + roi.w, roi.y + roi.h)
        for name, roi in result.rois.items()
    }
    out = _draw_boxes(out, boxes=boxes)
    cv2.imwrite(str(output), out)


def _read_calibration_frame(*, video_path: str | Path, frame_sec: float):
    target = max(0.0, float(frame_sec))
    last_frame = None
    with VideoCaptureService(video_path=video_path, target_fps=0) as capture:
        for packet in capture.iter_frames():
            last_frame = packet.frame
            if packet.timestamp + 1e-9 >= target:
                return packet.frame
    if last_frame is None:
        raise RuntimeError("No frames were read from input video.")
    return last_frame


def _draw_existing_box(frame, *, name: str, boxes: dict[str, tuple[int, int, int, int]]):
    out = frame.copy()
    fallback = boxes.get(name)
    if fallback is not None:
        out = _draw_boxes(out, boxes={name: fallback})
    return out


def _draw_boxes(frame, *, boxes: dict[str, tuple[int, int, int, int]]):
    if cv2 is None:
        return frame
    out = frame.copy()
    for name, (x1, y1, x2, y2) in boxes.items():
        cv2.rectangle(out, (x1, y1), (x2, y2), (70, 220, 255), 2)
        cv2.putText(
            out,
            name,
            (x1 + 4, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (70, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return out
