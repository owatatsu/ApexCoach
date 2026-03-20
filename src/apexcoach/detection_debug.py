from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from apexcoach.config import DetectionDebugConfig
from apexcoach.models import FrameEvents, FramePacket, GameState, ParsedStatus, ParsedTactical

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


class DetectionDebugDumper:
    def __init__(self, config: DetectionDebugConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self.output_dir = Path(config.output_dir)
        self._dumped_frames = 0
        self._metadata_handle = None

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._metadata_handle = (self.output_dir / "metadata.jsonl").open(
                "w",
                encoding="utf-8",
            )

    def close(self) -> None:
        if self._metadata_handle is not None:
            self._metadata_handle.close()
            self._metadata_handle = None

    def maybe_dump(
        self,
        packet: FramePacket,
        rois: dict[str, Any],
        roi_boxes: dict[str, tuple[int, int, int, int]],
        status: ParsedStatus,
        tactical: ParsedTactical,
        state: GameState,
        events: FrameEvents,
        parser_debug: dict[str, Any] | None,
    ) -> None:
        if not self.enabled:
            return
        if self._metadata_handle is None:
            return
        if self._dumped_frames >= max(0, int(self.config.max_frames)):
            return

        interval = max(1, int(self.config.dump_interval_frames))
        if packet.frame_index % interval != 0:
            return

        frame_dir = self.output_dir / f"frame_{packet.frame_index:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "timestamp": round(float(packet.timestamp), 3),
            "frame_index": int(packet.frame_index),
            "roi_boxes": roi_boxes,
            "status": _serialize(status),
            "tactical": _serialize(tactical),
            "state": _serialize(state),
            "events": _serialize(events),
            "parser_debug": _serialize_debug(parser_debug or {}),
        }
        self._metadata_handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        self._metadata_handle.flush()

        if self.config.save_roi_images:
            for name in ("hp_bar", "shield_bar"):
                roi = rois.get(name)
                if roi is not None:
                    _write_image(frame_dir / f"{name}.png", roi)

        if self.config.save_mask_images:
            self._write_status_debug_images(frame_dir, parser_debug or {})

        self._dumped_frames += 1

    def _write_status_debug_images(
        self,
        frame_dir: Path,
        parser_debug: dict[str, Any],
    ) -> None:
        status_debug = parser_debug.get("status", {})
        for name in ("hp", "shield"):
            entry = status_debug.get(name, {})
            focus = entry.get("focus")
            mask = entry.get("mask")
            if focus is not None:
                _write_image(frame_dir / f"{name}_focus.png", focus)
            if mask is not None:
                _write_image(frame_dir / f"{name}_mask.png", mask)


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _serialize_debug(debug: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in debug.items():
        if isinstance(value, dict):
            out[key] = _serialize_debug(value)
            continue
        if _looks_like_array(value):
            continue
        out[key] = _serialize(value)
    return out


def _looks_like_array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _write_image(path: Path, image: Any) -> None:
    if cv2 is None or image is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
