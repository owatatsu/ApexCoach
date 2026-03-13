import numpy as np

from apexcoach.ui_parser import _estimate_color_bar_ratio, _left_fill_ratio


def _make_bar_roi(
    fill_ratio: float,
    fill_bgr: tuple[int, int, int],
    *,
    width: int = 200,
    height: int = 20,
) -> np.ndarray:
    roi = np.full((height, width, 3), 18, dtype=np.uint8)
    roi[1:-1, 1:-1] = (36, 36, 36)

    inner_w = max(1, width - 2)
    fill_w = int(round(max(0.0, min(1.0, fill_ratio)) * inner_w))
    if fill_w > 0:
        roi[2:-2, 1 : 1 + fill_w] = fill_bgr
    return roi


def test_left_fill_ratio_contiguous_bar() -> None:
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[:, :12] = 255
    ratio = _left_fill_ratio(mask, min_col_occupancy=0.2, max_gap=2)
    assert 0.55 <= ratio <= 0.65


def test_left_fill_ratio_ignores_sparse_noise() -> None:
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[:, :8] = 255
    # Sparse noise near far right should not extend fill much.
    mask[0, 18] = 255
    mask[1, 19] = 255
    ratio = _left_fill_ratio(mask, min_col_occupancy=0.2, max_gap=2)
    assert 0.35 <= ratio <= 0.45


def test_estimate_hp_ratio_for_full_white_bar() -> None:
    roi = _make_bar_roi(1.0, (245, 245, 245))
    ratio, confidence = _estimate_color_bar_ratio(roi, target="hp")
    assert ratio is not None
    assert ratio >= 0.95
    assert confidence >= 0.65


def test_estimate_hp_ratio_for_low_red_bar() -> None:
    roi = _make_bar_roi(0.22, (0, 0, 220))
    ratio, confidence = _estimate_color_bar_ratio(roi, target="hp")
    assert ratio is not None
    assert 0.12 <= ratio <= 0.32
    assert confidence >= 0.3


def test_estimate_shield_ratio_for_empty_bar_with_outline() -> None:
    roi = _make_bar_roi(0.0, (255, 200, 80))
    ratio, confidence = _estimate_color_bar_ratio(roi, target="shield")
    assert ratio is not None
    assert ratio <= 0.05
    assert confidence >= 0.2
