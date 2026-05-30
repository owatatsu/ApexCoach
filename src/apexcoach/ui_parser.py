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
        self._last_debug: dict[str, Any] = {}

    def debug_snapshot(self) -> dict[str, Any]:
        return dict(self._last_debug)

    def parse_status(
        self, packet: FramePacket, rois: dict[str, "np.ndarray"]
    ) -> ParsedStatus:
        raw = self._telemetry_row(packet)

        hp_pct = _as_opt_float(raw.get("hp_pct"))
        shield_pct = _as_opt_float(raw.get("shield_pct"))
        hp_conf = 1.0 if hp_pct is not None else 0.0
        shield_conf = 1.0 if shield_pct is not None else 0.0
        hp_debug: dict[str, Any] = {"source": "telemetry" if hp_pct is not None else "missing"}
        shield_debug: dict[str, Any] = {
            "source": "telemetry" if shield_pct is not None else "missing"
        }
        allies_alive = _as_opt_int(raw.get("allies_alive"))
        allies_down = _as_opt_int(raw.get("allies_down"))

        if hp_pct is None and "hp_bar" in rois:
            hp_pct, hp_conf, hp_debug = self._estimate_hp(rois["hp_bar"])
        if shield_pct is None and "shield_bar" in rois:
            shield_pct, shield_conf, shield_debug = self._estimate_shield(
                rois["shield_bar"]
            )

        if allies_alive is None and allies_down is not None:
            allies_alive = max(0, 3 - allies_down)
        if allies_down is None and allies_alive is not None:
            allies_down = max(0, 3 - allies_alive)

        if allies_alive is None:
            allies_alive = 3
        if allies_down is None:
            allies_down = 0

        self._last_debug["status"] = {
            "hp": {
                **hp_debug,
                "ratio": None if hp_pct is None else round(float(hp_pct), 3),
                "confidence": round(float(hp_conf), 3),
            },
            "shield": {
                **shield_debug,
                "ratio": None if shield_pct is None else round(float(shield_pct), 3),
                "confidence": round(float(shield_conf), 3),
            },
        }

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
        low_ground_debug: dict[str, Any] = {
            "source": "telemetry"
            if low_ground is not None or low_ground_conf is not None
            else "missing"
        }
        exposed_debug: dict[str, Any] = {
            "source": "telemetry"
            if exposed is not None or exposed_conf is not None
            else "missing"
        }
        movement_debug: dict[str, Any] = {
            "source": "telemetry"
            if is_moving is not None or movement_score is not None
            else "missing"
        }

        low_ground_evidence: list[str] = []
        if low_ground is None or low_ground_conf is None:
            est_low, est_low_conf, low_ground_debug = self._estimate_low_ground(
                packet.frame,
                rois=rois,
                telemetry=raw,
            )
            if low_ground is None:
                low_ground = est_low
            if low_ground_conf is None:
                low_ground_conf = est_low_conf
        low_ground_evidence = list(low_ground_debug.get("evidence", []))

        if exposed is None or exposed_conf is None:
            est_exp, est_exp_conf, exposed_debug = self._estimate_exposed_no_cover(
                packet.frame
            )
            if exposed is None:
                exposed = est_exp
            if exposed_conf is None:
                exposed_conf = est_exp_conf

        if is_moving is None or movement_score is None:
            est_moving, est_motion, movement_debug = self._estimate_movement(
                packet.frame
            )
            if is_moving is None:
                is_moving = est_moving
            if movement_score is None:
                movement_score = est_motion

        self._last_debug["tactical"] = {
            "low_ground": {
                **low_ground_debug,
                "active": low_ground,
                "confidence": round(float(low_ground_conf or 0.0), 3),
                "evidence": low_ground_evidence,
            },
            "exposed": {
                **exposed_debug,
                "active": exposed,
                "confidence": round(float(exposed_conf or 0.0), 3),
            },
            "movement": {
                **movement_debug,
                "active": is_moving,
                "score": round(float(movement_score or 0.0), 3),
            },
        }

        return ParsedTactical(
            low_ground_disadvantage=low_ground,
            low_ground_confidence=low_ground_conf or 0.0,
            low_ground_evidence=low_ground_evidence,
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

    def _estimate_hp(
        self, roi: "np.ndarray"
    ) -> tuple[float | None, float, dict[str, Any]]:
        return _estimate_color_bar(roi, target="hp")

    def _estimate_shield(
        self, roi: "np.ndarray"
    ) -> tuple[float | None, float, dict[str, Any]]:
        return _estimate_color_bar(roi, target="shield")

    def _estimate_low_ground(
        self,
        frame: "np.ndarray",
        rois: dict[str, "np.ndarray"] | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> tuple[bool | None, float, dict[str, Any]]:
        if cv2 is None or np is None or frame.size == 0:
            return None, 0.0, {"source": "estimator", "valid": False}

        rois = rois or {}
        telemetry = telemetry or {}
        h, w = frame.shape[:2]
        cx1 = int(w * 0.33)
        cx2 = int(w * 0.67)
        top = frame[int(h * 0.08) : int(h * 0.40), cx1:cx2]
        bottom = frame[int(h * 0.55) : int(h * 0.88), cx1:cx2]
        if top.size == 0 or bottom.size == 0:
            return None, 0.0, {"source": "estimator", "valid": False}

        top_gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        bottom_gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        top_edges = cv2.Canny(top_gray, 70, 150).mean() / 255.0
        bottom_edges = cv2.Canny(bottom_gray, 70, 150).mean() / 255.0

        ratio = (top_edges + 1e-6) / (bottom_edges + 1e-6)
        edge_score = _clamp01((ratio - 1.15) / 1.0)
        horizon_y_pct, horizon_conf, horizon_debug = _estimate_horizon_y_pct(frame)
        horizon_score = _clamp01(((horizon_y_pct or 0.0) - 0.50) / 0.22) * horizon_conf
        minimap_score, minimap_debug = _estimate_minimap_elevation_signal(
            rois.get("minimap")
        )
        telemetry_score, telemetry_debug = _telemetry_low_ground_score(telemetry)

        signals = [
            ("view_angle_edges", edge_score, 1.0),
            ("horizon_pitch", horizon_score, 0.8),
            ("minimap", minimap_score, 0.45),
            ("telemetry", telemetry_score, 1.0),
        ]
        score = _combine_weighted_scores(signals)
        evidence = [
            name
            for name, raw_score, weight in signals
            if raw_score * weight >= 0.18
        ]
        debug = {
            "source": "multi_signal_estimator",
            "valid": True,
            "top_edges": round(float(top_edges), 4),
            "bottom_edges": round(float(bottom_edges), 4),
            "edge_ratio": round(float(ratio), 4),
            "edge_score": round(float(edge_score), 4),
            "horizon_y_pct": None
            if horizon_y_pct is None
            else round(float(horizon_y_pct), 4),
            "horizon_confidence": round(float(horizon_conf), 4),
            "horizon": horizon_debug,
            "minimap": minimap_debug,
            "telemetry": telemetry_debug,
            "score": round(float(score), 4),
            "evidence": evidence,
        }
        if score < 0.45:
            return None, score, debug
        return True, score, debug

    def _estimate_exposed_no_cover(
        self, frame: "np.ndarray"
    ) -> tuple[bool | None, float, dict[str, Any]]:
        if cv2 is None or np is None or frame.size == 0:
            return None, 0.0, {"source": "estimator", "valid": False}

        h, w = frame.shape[:2]
        # Near player view: lower-center region where immediate cover usually appears.
        near = frame[int(h * 0.52) : int(h * 0.9), int(w * 0.2) : int(w * 0.8)]
        if near.size == 0:
            return None, 0.0, {"source": "estimator", "valid": False}

        gray = cv2.cvtColor(near, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140).mean() / 255.0
        texture = float(np.std(gray) / 128.0)
        # Low edge + low texture tends to indicate open space.
        openness = max(0.0, min(1.0, 1.0 - (0.6 * edges + 0.4 * min(1.0, texture))))
        debug = {
            "source": "estimator",
            "valid": True,
            "edges": round(float(edges), 4),
            "texture": round(float(texture), 4),
            "openness": round(float(openness), 4),
        }
        if openness < 0.5:
            return None, openness, debug
        return True, openness, debug

    def _estimate_movement(
        self, frame: "np.ndarray"
    ) -> tuple[bool | None, float, dict[str, Any]]:
        if cv2 is None or np is None or frame.size == 0:
            return None, 0.0, {"source": "estimator", "valid": False}

        h, w = frame.shape[:2]
        y1 = int(h * 0.18)
        y2 = int(h * 0.82)
        x1 = int(w * 0.18)
        x2 = int(w * 0.82)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0, {"source": "estimator", "valid": False}

        core = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

        prev = self._prev_motion_gray
        self._prev_motion_gray = gray
        if prev is None:
            return None, 0.0, {"source": "estimator", "valid": False, "warmup": True}

        diff = cv2.absdiff(gray, prev)
        score = float(diff.mean() / 255.0)
        moving = score >= 0.045
        return moving, max(0.0, min(1.0, score)), {
            "source": "estimator",
            "valid": True,
            "score": round(float(score), 4),
        }


def _estimate_color_bar(
    roi: "np.ndarray", target: str
) -> tuple[float | None, float, dict[str, Any]]:
    if cv2 is None or np is None or roi.size == 0:
        return None, 0.0, {"source": "roi", "valid": False}

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
    normalized_ratio = min(1.0, max(0.0, ratio))
    debug = {
        "source": "roi",
        "valid": True,
        "focus": focus,
        "mask": mask,
        "fill_fraction": round(float((mask > 0).mean()), 4),
        "ratio": round(float(normalized_ratio), 4),
        "confidence": round(float(confidence), 4),
    }
    return normalized_ratio, confidence, debug


def _estimate_color_bar_ratio(
    roi: "np.ndarray", target: str
) -> tuple[float | None, float]:
    ratio, confidence, _ = _estimate_color_bar(roi, target=target)
    return ratio, confidence


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


def _estimate_horizon_y_pct(frame: "np.ndarray") -> tuple[float | None, float, dict[str, Any]]:
    if cv2 is None or np is None or frame.size == 0:
        return None, 0.0, {"valid": False}

    h, w = frame.shape[:2]
    y1 = int(h * 0.18)
    y2 = int(h * 0.72)
    x1 = int(w * 0.18)
    x2 = int(w * 0.82)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return None, 0.0, {"valid": False}

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=max(40, region.shape[1] // 5),
        maxLineGap=18,
    )
    if lines is None:
        return None, 0.0, {"valid": True, "line_count": 0}

    weighted_y: list[tuple[float, float]] = []
    for raw_line in lines[:80]:
        x_a, y_a, x_b, y_b = [float(v) for v in raw_line[0]]
        dx = x_b - x_a
        dy = y_b - y_a
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 0.0:
            continue
        slope = abs(dy / max(1.0, abs(dx)))
        if slope > 0.22:
            continue
        weighted_y.append(((y_a + y_b) / 2.0, length))

    if not weighted_y:
        return None, 0.0, {"valid": True, "line_count": int(len(lines)), "horizontal_count": 0}

    weighted_y.sort(key=lambda item: item[0])
    total_weight = sum(weight for _, weight in weighted_y)
    midpoint = total_weight / 2.0
    running = 0.0
    median_y = weighted_y[0][0]
    for y, weight in weighted_y:
        running += weight
        if running >= midpoint:
            median_y = y
            break

    y_pct = (float(y1) + median_y) / float(h)
    confidence = _clamp01(len(weighted_y) / 12.0)
    return y_pct, confidence, {
        "valid": True,
        "line_count": int(len(lines)),
        "horizontal_count": len(weighted_y),
        "y_pct": round(float(y_pct), 4),
        "confidence": round(float(confidence), 4),
    }


def _estimate_minimap_elevation_signal(roi: "np.ndarray" | None) -> tuple[float, dict[str, Any]]:
    if cv2 is None or np is None or roi is None or roi.size == 0:
        return 0.0, {"valid": False}

    height, width = roi.shape[:2]
    if height < 20 or width < 20:
        return 0.0, {"valid": False}

    focus = roi[int(height * 0.12) : int(height * 0.88), int(width * 0.12) : int(width * 0.88)]
    if focus.size == 0:
        return 0.0, {"valid": False}

    gray = cv2.cvtColor(focus, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    top = edges[: edges.shape[0] // 2]
    bottom = edges[edges.shape[0] // 2 :]
    top_density = float((top > 0).mean()) if top.size else 0.0
    bottom_density = float((bottom > 0).mean()) if bottom.size else 0.0
    ratio = (top_density + 1e-6) / (bottom_density + 1e-6)
    # Minimap elevation is map/style dependent, so keep this as a weak cue.
    score = _clamp01((ratio - 1.25) / 1.5)
    return score, {
        "valid": True,
        "top_edge_density": round(top_density, 4),
        "bottom_edge_density": round(bottom_density, 4),
        "edge_ratio": round(float(ratio), 4),
        "score": round(float(score), 4),
    }


def _telemetry_low_ground_score(data: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    view_up = _as_opt_float(
        _pick_first(data, "view_pitch_up_score", "camera_pitch_up_score")
    )
    if view_up is None:
        pitch_deg = _as_raw_float(_pick_first(data, "view_pitch_deg", "camera_pitch_deg"))
        if pitch_deg is not None:
            view_up = _clamp01(pitch_deg / 35.0)

    horizon = _as_opt_float(_pick_first(data, "horizon_y_pct", "horizon_ratio"))
    horizon_score = _clamp01(((horizon or 0.0) - 0.50) / 0.22) if horizon is not None else 0.0

    minimap = _as_opt_float(
        _pick_first(
            data,
            "minimap_high_ground_confidence",
            "minimap_elevation_confidence",
        )
    )
    active = _as_opt_bool(
        _pick_first(
            data,
            "minimap_high_ground_disadvantage",
            "minimap_low_ground",
        )
    )
    if active is False:
        minimap = 0.0
    elif active is True and minimap is None:
        minimap = 0.7

    score = max(view_up or 0.0, horizon_score, minimap or 0.0)
    return score, {
        "view_up_score": None if view_up is None else round(float(view_up), 4),
        "horizon_score": round(float(horizon_score), 4),
        "minimap_score": None if minimap is None else round(float(minimap), 4),
        "score": round(float(score), 4),
    }


def _combine_weighted_scores(signals: list[tuple[str, float, float]]) -> float:
    remaining = 1.0
    for _name, score, weight in signals:
        remaining *= 1.0 - _clamp01(score) * max(0.0, min(1.0, float(weight)))
    return _clamp01(1.0 - remaining)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_opt_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    return None


def _as_raw_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
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
