import numpy as np

from fpsd_reassessment.features import (
    BANDS, CHANNELS, WelchConfig, assert_deterministic, extract_paper95,
    feature_index_rows, flatten_band_major, segment_nonoverlapping,
)


def synthetic_eeg(seconds=10, fs=500):
    time = np.arange(seconds * fs) / fs
    return np.stack([np.sin(2 * np.pi * (2 + index % 10) * time) + 0.01 * index for index in range(19)])


def test_exact_five_second_windows_and_tail_discard():
    windows = segment_nonoverlapping(synthetic_eeg(seconds=11), fs=500)
    assert windows.shape == (2, 19, 2500)


def test_paper95_shape_finite_and_order():
    tensor, flat, _ = extract_paper95(synthetic_eeg(), fs=500)
    assert tensor.shape == (2, 19, 5)
    assert flat.shape == (2, 95)
    assert np.isfinite(flat).all()
    np.testing.assert_array_equal(flat[:, :19], tensor[:, :, 0])
    np.testing.assert_array_equal(flat[:, 19:38], tensor[:, :, 1])


def test_feature_index_band_major():
    rows = feature_index_rows()
    assert len(rows) == 95
    assert [row["channel"] for row in rows[:19]] == list(CHANNELS)
    assert [row["band"] for row in rows[::19]] == list(BANDS)


def test_extraction_is_deterministic():
    assert_deterministic(synthetic_eeg(), fs=500)
