from pathlib import Path

import numpy as np

from apexcoach.config import DetectionDebugConfig
from apexcoach.detection_debug import DetectionDebugDumper
from apexcoach.models import FrameEvents, FramePacket, GameState, ParsedStatus, ParsedTactical


def test_detection_debug_dumper_writes_metadata_and_images(tmp_path: Path) -> None:
    output_dir = tmp_path / "debug"
    dumper = DetectionDebugDumper(
        DetectionDebugConfig(
            enabled=True,
            output_dir=str(output_dir),
            dump_interval_frames=1,
            max_frames=2,
            save_roi_images=True,
            save_mask_images=True,
        )
    )

    packet = FramePacket(
        frame_index=15,
        timestamp=1.5,
        frame=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    rois = {
        "hp_bar": np.zeros((8, 20, 3), dtype=np.uint8),
        "shield_bar": np.zeros((8, 20, 3), dtype=np.uint8),
    }
    roi_boxes = {
        "hp_bar": (0, 0, 20, 8),
        "shield_bar": (0, 8, 20, 16),
    }
    parser_debug = {
        "status": {
            "hp": {
                "ratio": 0.5,
                "confidence": 0.8,
                "focus": np.zeros((8, 20, 3), dtype=np.uint8),
                "mask": np.zeros((8, 20), dtype=np.uint8),
            },
            "shield": {
                "ratio": 0.2,
                "confidence": 0.7,
                "focus": np.zeros((8, 20, 3), dtype=np.uint8),
                "mask": np.zeros((8, 20), dtype=np.uint8),
            },
        }
    }

    dumper.maybe_dump(
        packet=packet,
        rois=rois,
        roi_boxes=roi_boxes,
        status=ParsedStatus(hp_pct=0.5, shield_pct=0.2, hp_confidence=0.8, shield_confidence=0.7),
        tactical=ParsedTactical(),
        state=GameState(timestamp=1.5, hp_pct=0.5, shield_pct=0.2),
        events=FrameEvents(timestamp=1.5),
        parser_debug=parser_debug,
    )
    dumper.close()

    metadata = (output_dir / "metadata.jsonl").read_text(encoding="utf-8")
    assert "\"frame_index\": 15" in metadata
    assert (output_dir / "frame_000015" / "hp_bar.png").exists()
    assert (output_dir / "frame_000015" / "hp_mask.png").exists()
