from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apexcoach.models import (
    FramePacket,
    ParsedNotifications,
    ParsedStatus,
    ParsedTactical,
)

if TYPE_CHECKING:
    import numpy as np

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None
    np = None


class TelemetryReader:
    def __init__(self, path: str | Path) -> None:
        self._by_frame: dict[int, dict[str, Any]] = {}
        self._timeline: list[tuple[float, dict[str, Any]]] = []
        self._times: list[float] = []

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Telemetry file not found: {path_obj}")

        with path_obj.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue

                frame_index = record.get("frame_index")
                if isinstance(frame_index, int):
                    self._by_frame[frame_index] = record

                ts = record.get("timestamp")
                if isinstance(ts, (int, float)):
                    self._timeline.append((float(ts), record))

        self._timeline.sort(key=lambda item: item[0])
        self._times = [item[0] for item in self._timeline]

    def lookup(self, frame_index: int, timestamp: float) -> dict[str, Any]:
        if frame_index in self._by_frame:
            return dict(self._by_frame[frame_index])

        if not self._timeline:
            return {}

        idx = bisect.bisect_right(self._times, timestamp) - 1
        if idx < 0:
            return {}
        return dict(self._timeline[idx][1])

    def lookup_event(
        self, frame_index: int, timestamp: float, max_age_seconds: float = 0.2
    ) -> dict[str, Any]:
        if frame_index in self._by_frame:
            return dict(self._by_frame[frame_index])

        if not self._timeline:
            return {}

        idx = bisect.bisect_right(self._times, timestamp) - 1
        if idx < 0:
            return {}

        event_ts, record = self._timeline[idx]
        if timestamp - event_ts <= max_age_seconds:
            return dict(record)
        return {}


class SimpleUiParser:
    def __init__(self, telemetry: TelemetryReader | None = None) -> None:
        self.telemetry = telemetry
        self._prev_motion_gray = None

    def parse_status(
        self, packet: FramePacket, rois: dict[str, "np.ndarray"]
    ) -> ParsedStatus:
        raw = self._telemetry_row(packet)

        hp_pct = _as_opt_float(raw.get("hp_pct"))
        shield_pct = _as_opt_float(raw.get("shield_pct"))
        hp_conf = 1.0 if hp_pct is not None else 0.0
        shield_conf = 1.0 if shield_pct is not None else 0.0
        allies_alive = _as_opt_int(raw.get("allies_alive"))
        allies_down = _as_opt_int(raw.get("allies_down"))

        if hp_pct is None and "hp_bar" in rois:
            hp_pct, hp_conf = self._estimate_hp(rois["hp_bar"])
        if shield_pct is None and "shield_bar" in rois:
            shield_pct, shield_conf = self._estimate_shield(rois["shield_bar"])

        if allies_alive is None and allies_down is not None:
            allies_alive = max(0, 3 - allies_down)
        if allies_down is None and allies_alive is not None:
            allies_down = max(0, 3 - allies_alive)

        if allies_alive is None:
            allies_alive = 3
        if allies_down is None:
            allies_down = 0

        return ParsedStatus(
            hp_pct=hp_pct,
            shield_pct=shield_pct,
            hp_confidence=hp_conf,
            shield_confidence=shield_conf,
            allies_alive=allies_alive,
            allies_down=allies_down,
        )

    def parse_notifications(
        self, packet: FramePacket, _rois: dict[str, "np.ndarray"]
    ) -> ParsedNotifications:
        raw = self._telemetry_event_row(packet)
        return ParsedNotifications(
            enemy_knock=_as_bool(raw.get("enemy_knock")),
            ally_knock=_as_bool(raw.get("ally_knock")),
        )

    def parse_tactical(
        self, packet: FramePacket, rois: dict[str, "np.ndarray"]
    ) -> ParsedTactical:
        raw = self._telemetry_row(packet)
        low_ground = _as_opt_bool(
            _pick_first(
                raw,
                "low_ground_disadvantage",
                "low_ground",
                "is_low_ground",
            )
        )
        low_ground_conf = _as_opt_float(
            _pick_first(raw, "low_ground_confidence", "low_ground_conf")
        )

        exposed = _as_opt_bool(
            _pick_first(
                raw,
                "exposed_no_cover",
                "is_exposed_no_cover",
                "no_cover",
            )
        )
        exposed_conf = _as_opt_float(
            _pick_first(raw, "exposed_confidence", "no_cover_confidence")
        )
        is_moving = _as_opt_bool(
            _pick_first(raw, "is_moving", "moving", "player_moving")
        )
        movement_score = _as_opt_float(
            _pick_first(raw, "movement_score", "motion_score")
        )

        if low_ground is None or low_ground_conf is None:
            est_low, est_low_conf = self._estimate_low_ground(packet.frame)
            if low_ground is None:
                low_ground = est_low
            if low_ground_conf is None:
                low_ground_conf = est_low_conf

        if exposed is None or exposed_conf is None:
            est_exp, est_exp_conf = self._estimate_exposed_no_cover(packet.frame)
            if exposed is None:
                exposed = est_exp
            if exposed_conf is None:
                exposed_conf = est_exp_conf

        if is_moving is None or movement_score is None:
            est_moving, est_motion = self._estimate_movement(packet.frame)
            if is_moving is None:
                is_moving = est_moving
            if movement_score is None:
                movement_score = est_motion

        return ParsedTactical(
            low_ground_disadvantage=low_ground,
            low_ground_confidence=low_ground_conf or 0.0,
            exposed_no_cover=exposed,
            exposed_confidence=exposed_conf or 0.0,
            is_moving=is_moving,
            movement_score=movement_score or 0.0,
        )

    def _telemetry_row(self, packet: FramePacket) -> dict[str, Any]:
        if self.telemetry is None:
            return {}
        return self.telemetry.lookup(
            frame_index=packet.frame_index, timestamp=packet.timestamp
        )

    def _telemetry_event_row(self, packet: FramePacket) -> dict[str, Any]:
        if self.telemetry is None:
            return {}
        return self.telemetry.lookup_event(
            frame_index=packet.frame_index, timestamp=packet.timestamp
        )

    def _estimate_hp(self, roi: "np.ndarray") -> tuple[float | None, float]:
        return _estimate_color_bar_ratio(roi, target="hp")

    def _estimate_shield(self, roi: "np.ndarray") -> tuple[float | None, float]:
        return _estimate_color_bar_ratio(roi, target="shield")

    def _estimate_low_ground(self, frame: "np.ndarray") -> tuple[bool | None, float]:
        if cv2 is None or np is None or frame.size == 0:
            return None, 0.0

        h, w = frame.shape[:2]
        cx1 = int(w * 0.33)
        cx2 = int(w * 0.67)
        top = frame[int(h * 0.08) : int(h * 0.40), cx1:cx2]
        bottom = frame[int(h * 0.55) : int(h * 0.88), cx1:cx2]
        if top.size == 0 or bottom.size == 0:
            return None, 0.0

        top_gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        bottom_gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        top_edges = cv2.Canny(top_gray, 70, 150).mean() / 255.0
        bottom_edges = cv2.Canny(bottom_gray, 70, 150).mean() / 255.0

        ratio = (top_edges + 1e-6) / (bottom_edges + 1e-6)
        # Looking up often produces richer geometry/edges in upper center.
        score = min(1.0, max(0.0, (ratio - 1.15) / 1.0))
        if score < 0.45:
            return None, score
        return True, score

    def _estimate_exposed_no_cover(
        self, frame: "np.ndarray"
    ) -> tuple[bool | None, float]:
        if cv2 is None or np is None or frame.size == 0:
            return None, 0.0

        h, w = frame.shape[:2]
        # Near player view: lower-center region where immediate cover usually appears.
        near = frame[int(h * 0.52) : int(h * 0.9), int(w * 0.2) : int(w * 0.8)]
        if near.size == 0:
            return None, 0.0

        gray = cv2.cvtColor(near, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140).mean() / 255.0
        texture = float(np.std(gray) / 128.0)
        # Low edge + low texture tends to indicate open space.
        openness = max(0.0, min(1.0, 1.0 - (0.6 * edges + 0.4 * min(1.0, texture))))
        if openness < 0.5:
            return None, openness
        return True, openness

    def _estimate_movement(self, frame: "np.ndarray") -> tuple[bool | None, float]:
        if cv2 is None or np is None or frame.size == 0:
            return None, 0.0

        h, w = frame.shape[:2]
        y1 = int(h * 0.18)
        y2 = int(h * 0.82)
        x1 = int(w * 0.18)
        x2 = int(w * 0.82)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        core = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

        prev = self._prev_motion_gray
        self._prev_motion_gray = gray
        if prev is None:
            return None, 0.0

        diff = cv2.absdiff(gray, prev)
        score = float(diff.mean() / 255.0)
        moving = score >= 0.045
        return moving, max(0.0, min(1.0, score))


def _estimate_color_bar_ratio(
    roi: "np.ndarray", target: str
) -> tuple[float | None, float]:
    if cv2 is None or np is None or roi.size == 0:
        return None, 0.0

    focus = _extract_bar_focus_region(roi)
    hsv = cv2.cvtColor(focus, cv2.COLOR_BGR2HSV)
    mask = _build_bar_mask(hsv, target=target)
    mask = _denoise_bar_mask(mask)

    # Keep mostly contiguous left-to-right fill and ignore scattered UI noise.
    gap = max(2, focus.shape[1] // 50)
    ratio = _left_fill_ratio(mask, min_col_occupancy=0.28, max_gap=gap)
    confidence = _bar_confidence(mask, ratio=ratio)
    if ratio <= 0.02:
        ratio = 0.0
        confidence = max(confidence, _empty_bar_confidence(focus, mask))
    elif ratio >= 0.98:
        ratio = 1.0
    return min(1.0, max(0.0, ratio)), confidence


def _extract_bar_focus_region(roi: "np.ndarray") -> "np.ndarray":
    if np is None or roi.size == 0:
        return roi

    height, width = roi.shape[:2]
    if height < 6 or width < 6:
        return roi

    y1 = max(0, int(round(height * 0.2)))
    y2 = min(height, int(round(height * 0.8)))
    x1 = max(0, int(round(width * 0.01)))
    x2 = min(width, int(round(width * 0.99)))
    if y2 <= y1 or x2 <= x1:
        return roi
    return roi[y1:y2, x1:x2]


def _build_bar_mask(hsv: "np.ndarray", target: str) -> "np.ndarray":
    white = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))

    if target == "hp":
        red1 = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255))
        red2 = cv2.inRange(hsv, (170, 90, 60), (180, 255, 255))
        return cv2.bitwise_or(white, cv2.bitwise_or(red1, red2))

    blue_purple = cv2.inRange(hsv, (95, 70, 60), (155, 255, 255))
    gold = cv2.inRange(hsv, (14, 80, 90), (38, 255, 255))
    red1 = cv2.inRange(hsv, (0, 100, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 100, 70), (180, 255, 255))
    warm = cv2.bitwise_or(gold, cv2.bitwise_or(red1, red2))
    return cv2.bitwise_or(white, cv2.bitwise_or(blue_purple, warm))


def _denoise_bar_mask(mask: "np.ndarray") -> "np.ndarray":
    if cv2 is None:
        return mask

    height, width = mask.shape[:2]
    kernel_w = max(2, width // 40)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, kernel_w // 2), 1))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
    return cleaned


def _left_fill_ratio(mask: "np.ndarray", min_col_occupancy: float, max_gap: int) -> float:
    col_occ = (mask > 0).mean(axis=0)
    filled = col_occ >= min_col_occupancy
    width = int(filled.shape[0])
    if width <= 0:
        return 0.0

    last_filled = -1
    gap = 0
    for idx, is_filled in enumerate(filled):
        if bool(is_filled):
            last_filled = idx
            gap = 0
            continue

        gap += 1
        if gap > max_gap and last_filled >= 0:
            break

    if last_filled < 0:
        return 0.0
    return float(last_filled + 1) / float(width)


def _bar_confidence(mask: "np.ndarray", ratio: float) -> float:
    col_occ = (mask > 0).mean(axis=0)
    width = int(col_occ.shape[0])
    if width <= 0:
        return 0.0

    end = max(1, int(round(ratio * width)))
    prefix = col_occ[:end]
    suffix = col_occ[end:]
    prefix_mean = float(prefix.mean()) if prefix.size else 0.0
    suffix_mean = float(suffix.mean()) if suffix.size else 0.0

    # Good bar: dense prefix, sparse suffix.
    sep = max(0.0, min(1.0, prefix_mean - suffix_mean))
    fill_fraction = float((mask > 0).mean()) if mask.size else 0.0
    conf = (
        0.45 * max(0.0, min(1.0, prefix_mean))
        + 0.35 * sep
        + 0.20 * max(0.0, min(1.0, fill_fraction * 2.5))
    )
    return max(0.0, min(1.0, conf))


def _empty_bar_confidence(roi: "np.ndarray", mask: "np.ndarray") -> float:
    if np is None or roi.size == 0 or mask.size == 0:
        return 0.0

    fill_fraction = float((mask > 0).mean())
    if fill_fraction > 0.03:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    darkness = 1.0 - min(1.0, float(gray.mean()) / 90.0)
    return max(0.0, min(0.45, 0.18 + darkness * 0.27))


def _as_opt_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    return None


def _as_opt_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _as_opt_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return None


def _pick_first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "y"}
    return False
