from apexcoach.config import Roi
from apexcoach.roi_calibration import RoiCalibrationResult, render_roi_config


def test_render_roi_config_uses_frame_as_reference() -> None:
    result = RoiCalibrationResult(
        frame_width=1920,
        frame_height=1080,
        rois={
            "shield_bar": Roi(42, 982, 420, 22),
            "hp_bar": Roi(42, 1006, 420, 24),
        },
    )

    rendered = render_roi_config(result)

    assert "scale_rois_to_frame: true" in rendered
    assert "roi_reference_width: 1920" in rendered
    assert "roi_reference_height: 1080" in rendered
    assert "  hp_bar: [42, 1006, 420, 24]" in rendered
    assert "  shield_bar: [42, 982, 420, 22]" in rendered
