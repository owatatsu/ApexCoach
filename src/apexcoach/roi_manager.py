from __future__ import annotations

from typing import TYPE_CHECKING

from apexcoach.config import Roi

if TYPE_CHECKING:
    import numpy as np


class RoiManager:
    def __init__(self, rois: dict[str, Roi]) -> None:
        self._rois = rois

    def crop(self, frame: "np.ndarray") -> dict[str, "np.ndarray"]:
        height, width = frame.shape[:2]
        out: dict[str, "np.ndarray"] = {}

        for name, roi in self._rois.items():
            x1 = max(0, roi.x)
            y1 = max(0, roi.y)
            x2 = min(width, roi.x + roi.w)
            y2 = min(height, roi.y + roi.h)
            if x2 <= x1 or y2 <= y1:
                continue
            out[name] = frame[y1:y2, x1:x2]

        return out
