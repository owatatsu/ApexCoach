from __future__ import annotations

from typing import TYPE_CHECKING

from apexcoach.config import Roi

if TYPE_CHECKING:
    import numpy as np


class RoiManager:
    def __init__(
        self,
        rois: dict[str, Roi],
        scale_to_frame: bool = True,
        reference_width: int = 1920,
        reference_height: int = 1080,
    ) -> None:
        self._rois = rois
        self._scale_to_frame = scale_to_frame
        self._reference_width = max(1, int(reference_width))
        self._reference_height = max(1, int(reference_height))

    def crop(
        self,
        frame: "np.ndarray",
        boxes: dict[str, tuple[int, int, int, int]] | None = None,
    ) -> dict[str, "np.ndarray"]:
        out: dict[str, "np.ndarray"] = {}
        resolved = boxes if boxes is not None else self.resolve_boxes(frame)
        for name, (x1, y1, x2, y2) in resolved.items():
            out[name] = frame[y1:y2, x1:x2]
        return out

    def resolve_boxes(self, frame: "np.ndarray") -> dict[str, tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        out: dict[str, tuple[int, int, int, int]] = {}
        sx = float(width) / float(self._reference_width)
        sy = float(height) / float(self._reference_height)

        for name, roi in self._rois.items():
            if self._scale_to_frame:
                x = int(round(roi.x * sx))
                y = int(round(roi.y * sy))
                w = int(round(roi.w * sx))
                h = int(round(roi.h * sy))
            else:
                x = int(roi.x)
                y = int(roi.y)
                w = int(roi.w)
                h = int(roi.h)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            out[name] = (x1, y1, x2, y2)

        return out
