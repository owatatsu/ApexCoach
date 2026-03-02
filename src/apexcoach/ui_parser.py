from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apexcoach.models import FramePacket, ParsedNotifications, ParsedStatus

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

    def parse_status(
        self, packet: FramePacket, rois: dict[str, "np.ndarray"]
    ) -> ParsedStatus:
        raw = self._telemetry_row(packet)

        hp_pct = _as_opt_float(raw.get("hp_pct"))
        shield_pct = _as_opt_float(raw.get("shield_pct"))
        allies_alive = _as_opt_int(raw.get("allies_alive"))
        allies_down = _as_opt_int(raw.get("allies_down"))

        if hp_pct is None and "hp_bar" in rois:
            hp_pct = self._estimate_hp(rois["hp_bar"])
        if shield_pct is None and "shield_bar" in rois:
            shield_pct = self._estimate_shield(rois["shield_bar"])

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

    def _estimate_hp(self, roi: "np.ndarray") -> float | None:
        return _estimate_color_bar_ratio(roi, target="hp")

    def _estimate_shield(self, roi: "np.ndarray") -> float | None:
        return _estimate_color_bar_ratio(roi, target="shield")


def _estimate_color_bar_ratio(roi: "np.ndarray", target: str) -> float | None:
    if cv2 is None or np is None or roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if target == "hp":
        # Red around hue 0 and 180.
        mask1 = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        # Blue/purple-ish for shield.
        mask = cv2.inRange(hsv, (90, 50, 45), (150, 255, 255))

    active_by_col = (mask > 0).mean(axis=0) > 0.25
    ratio = float(active_by_col.sum()) / float(active_by_col.shape[0])
    return min(1.0, max(0.0, ratio))


def _as_opt_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    return None


def _as_opt_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
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
