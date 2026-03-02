import numpy as np

from apexcoach.ui_parser import _left_fill_ratio


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
