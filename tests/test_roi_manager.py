import numpy as np

from apexcoach.config import Roi
from apexcoach.roi_manager import RoiManager


def test_resolve_boxes_and_crop_with_clipping() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    rois = {
        "in": Roi(10, 10, 20, 15),
        "clip": Roi(110, 90, 30, 20),
        "invalid": Roi(130, 130, 5, 5),
    }
    manager = RoiManager(
        rois=rois,
        scale_to_frame=False,
        reference_width=120,
        reference_height=100,
    )
    boxes = manager.resolve_boxes(frame)
    assert boxes["in"] == (10, 10, 30, 25)
    assert boxes["clip"] == (110, 90, 120, 100)
    assert "invalid" not in boxes

    cropped = manager.crop(frame, boxes=boxes)
    assert cropped["in"].shape[:2] == (15, 20)
    assert cropped["clip"].shape[:2] == (10, 10)
