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

    def crop(self, frame: "np.ndarray") -> dict[str, "np.ndarray"]:
        height, width = frame.shape[:2]
        out: dict[str, "np.ndarray"] = {}
        sx = float(width) / float(self._reference_width)
        sy = float(height) / float(self._reference_height)

        for name, roi in self._rois.items():
            if self._scale_to_frame:
                x = int(round(roi.x * sx))
                y = int(round(roi.y * sy))
                w = int(round(roi.w * sx))
                h = int(round(roi.h * sy))
            else:
                x = roi.x
                y = roi.y
                w = roi.w
                h = roi.h

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            out[name] = frame[y1:y2, x1:x2]

        return out
