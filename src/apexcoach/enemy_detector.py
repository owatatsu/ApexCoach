from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from apexcoach.config import YoloConfig
from apexcoach.models import EnemyDetection, EnemyState

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _TrackPoint:
    timestamp: float
    center_x: float
    center_y: float


class YoloEnemyDetector:
    def __init__(self, config: YoloConfig) -> None:
        self.config = config
        self._enabled = bool(config.enabled)
        self._disabled_due_to_error = False
        self._warned_missing_dependency = False
        self._model = None
        self._track_history: dict[int, deque[_TrackPoint]] = {}

    def infer(
        self,
        frame,
        *,
        timestamp: float,
        color_format: str = "bgr",
    ) -> EnemyState:
        if not self._enabled or self._disabled_due_to_error:
            return EnemyState()

        model = self._ensure_model()
        if model is None:
            return EnemyState()

        try:
            prepared = _prepare_frame(frame, color_format=color_format)
        except ValueError as exc:
            LOGGER.warning("YOLO frame preparation failed: %s", exc)
            return EnemyState()
        if prepared is None:
            return EnemyState()

        try:
            if self.config.track_enabled:
                results = model.track(
                    source=prepared,
                    conf=float(self.config.confidence_threshold),
                    iou=float(self.config.iou_threshold),
                    max_det=int(self.config.max_detections),
                    imgsz=int(self.config.imgsz),
                    device=(self.config.device or None),
                    classes=None,
                    tracker=self.config.tracker,
                    persist=bool(self.config.persist_tracks),
                    verbose=False,
                )
            else:
                results = model.predict(
                    source=prepared,
                    conf=float(self.config.confidence_threshold),
                    iou=float(self.config.iou_threshold),
                    max_det=int(self.config.max_detections),
                    imgsz=int(self.config.imgsz),
                    device=(self.config.device or None),
                    classes=None,
                    verbose=False,
                )
        except Exception as exc:  # pragma: no cover - runtime safety
            self._disable_with_warning(f"YOLO inference failed and was disabled: {exc}")
            return EnemyState()

        detections = self._parse_results(results)
        self._update_track_history(detections, timestamp=timestamp)
        return _summarize_enemy_state(
            detections=detections,
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
            history=self._track_history,
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if YOLO is None:
            if not self._warned_missing_dependency:
                LOGGER.warning(
                    "YOLO enemy detection is enabled but ultralytics is not installed. "
                    "Install with `pip install -e .[yolo]`."
                )
                self._warned_missing_dependency = True
            self._disabled_due_to_error = True
            return None
        try:
            self._model = YOLO(self.config.model_name)
        except Exception as exc:  # pragma: no cover - runtime safety
            self._disable_with_warning(f"Failed to load YOLO model and disabled feature: {exc}")
            return None
        return self._model

    def _parse_results(self, results: Any) -> list[EnemyDetection]:
        if not results:
            return []

        first = results[0]
        boxes = getattr(first, "boxes", None)
        if boxes is None:
            return []

        names = getattr(first, "names", {}) or {}
        allowed = {name.strip().lower() for name in self.config.class_names if name.strip()}
        xyxy_list = _to_list(getattr(boxes, "xyxy", None))
        conf_list = _to_list(getattr(boxes, "conf", None))
        cls_list = _to_list(getattr(boxes, "cls", None))
        id_list = _to_list(getattr(boxes, "id", None))

        detections: list[EnemyDetection] = []
        for index, xyxy in enumerate(xyxy_list):
            if not isinstance(xyxy, list) or len(xyxy) < 4:
                continue
            class_id = int(cls_list[index]) if index < len(cls_list) else -1
            class_name = str(names.get(class_id, class_id))
            if allowed and class_name.strip().lower() not in allowed:
                continue
            confidence = float(conf_list[index]) if index < len(conf_list) else 0.0
            track_id = int(id_list[index]) if index < len(id_list) and id_list[index] is not None else None
            detections.append(
                EnemyDetection(
                    x1=int(round(float(xyxy[0]))),
                    y1=int(round(float(xyxy[1]))),
                    x2=int(round(float(xyxy[2]))),
                    y2=int(round(float(xyxy[3]))),
                    confidence=max(0.0, min(1.0, confidence)),
                    class_id=class_id,
                    class_name=class_name,
                    track_id=track_id,
                )
            )
        return detections

    def _update_track_history(
        self,
        detections: list[EnemyDetection],
        *,
        timestamp: float,
    ) -> None:
        max_history = max(2, int(self.config.history_size))
        active_track_ids: set[int] = set()
        for detection in detections:
            if detection.track_id is None:
                continue
            active_track_ids.add(detection.track_id)
            points = self._track_history.setdefault(
                detection.track_id,
                deque(maxlen=max_history),
            )
            points.append(
                _TrackPoint(
                    timestamp=timestamp,
                    center_x=(detection.x1 + detection.x2) / 2.0,
                    center_y=(detection.y1 + detection.y2) / 2.0,
                )
            )

        stale_ids = [
            track_id
            for track_id in self._track_history
            if track_id not in active_track_ids
        ]
        for track_id in stale_ids:
            history = self._track_history.get(track_id)
            if history is None:
                continue
            while history and (timestamp - history[0].timestamp) > 1.5:
                history.popleft()
            if not history:
                self._track_history.pop(track_id, None)

    def _disable_with_warning(self, message: str) -> None:
        LOGGER.warning(message)
        self._disabled_due_to_error = True


def draw_enemy_debug(frame, enemy_state: EnemyState):
    if cv2 is None or not enemy_state.available or not enemy_state.detections:
        return frame

    out = frame.copy()
    for detection in enemy_state.detections:
        color = (60, 180, 255)
        cv2.rectangle(out, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        if detection.track_id is not None:
            label += f" #{detection.track_id}"
        text_y = max(16, detection.y1 - 6)
        cv2.putText(
            out,
            label,
            (detection.x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _prepare_frame(frame, *, color_format: str):
    if cv2 is None or frame is None or getattr(frame, "size", 0) <= 0:
        return None
    lowered = (color_format or "bgr").strip().lower()
    if lowered == "rgb":
        return frame
    if lowered == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported color_format: {color_format!r}")


def _summarize_enemy_state(
    *,
    detections: list[EnemyDetection],
    frame_width: int,
    frame_height: int,
    history: dict[int, deque[_TrackPoint]],
) -> EnemyState:
    enemy_left = 0
    enemy_center = 0
    enemy_right = 0
    tracked_enemy_ids: list[int] = []
    for detection in detections:
        cx = (detection.x1 + detection.x2) / 2.0
        if cx < frame_width / 3.0:
            enemy_left += 1
        elif cx < (frame_width * 2.0 / 3.0):
            enemy_center += 1
        else:
            enemy_right += 1
        if detection.track_id is not None:
            tracked_enemy_ids.append(detection.track_id)

    tracked_enemy_ids = sorted(set(tracked_enemy_ids))
    movement_trend = _infer_movement_trend(tracked_enemy_ids, history)
    summary_lines = [
        f"enemy_count={len(detections)}",
        f"enemy_left={enemy_left}",
        f"enemy_center={enemy_center}",
        f"enemy_right={enemy_right}",
    ]
    if tracked_enemy_ids:
        summary_lines.append(f"tracked_enemy_ids={tracked_enemy_ids}")
    if movement_trend:
        summary_lines.append(f"enemy_movement={movement_trend}")

    return EnemyState(
        available=True,
        detections=detections,
        frame_width=frame_width,
        frame_height=frame_height,
        enemy_count=len(detections),
        enemy_left=enemy_left,
        enemy_center=enemy_center,
        enemy_right=enemy_right,
        tracked_enemy_ids=tracked_enemy_ids,
        movement_trend=movement_trend,
        summary_lines=summary_lines,
    )


def _infer_movement_trend(
    track_ids: list[int],
    history: dict[int, deque[_TrackPoint]],
) -> str:
    deltas: list[float] = []
    for track_id in track_ids:
        points = history.get(track_id)
        if points is None or len(points) < 2:
            continue
        deltas.append(points[-1].center_x - points[0].center_x)
    if not deltas:
        return ""
    positive = sum(1 for delta in deltas if delta >= 12.0)
    negative = sum(1 for delta in deltas if delta <= -12.0)
    if positive and not negative:
        return "left_to_right"
    if negative and not positive:
        return "right_to_left"
    if not positive and not negative:
        return "stable"
    return "mixed"


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
