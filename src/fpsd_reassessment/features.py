"""Paper-profile FPSD extraction with explicit shapes and ordering."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import welch

CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
)
BANDS = OrderedDict(
    delta=(0.5, 4.0),
    theta=(4.0, 8.0),
    alpha=(8.0, 13.0),
    beta=(13.0, 25.0),
    gamma=(25.0, 45.0),
)


@dataclass(frozen=True)
class WelchConfig:
    fs: float = 500.0
    window_seconds: float = 5.0
    n_fft: int = 256
    n_per_seg: int = 256
    n_overlap: int = 10
    detrend: str = "constant"
    window: str = "hann"
    fmin: float = 0.5
    fmax: float = 45.0


def segment_nonoverlapping(data: np.ndarray, fs: float, seconds: float = 5.0) -> np.ndarray:
    """Return exact [window, channel, sample] windows and discard only the tail."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected [channel, sample], got {data.shape}")
    samples = int(round(fs * seconds))
    if not np.isclose(samples / fs, seconds):
        raise ValueError("Window duration is not exactly representable at this sampling rate")
    n_windows = data.shape[1] // samples
    if n_windows < 1:
        raise ValueError("Recording contains no complete five-second window")
    trimmed = data[:, : n_windows * samples]
    return trimmed.reshape(data.shape[0], n_windows, samples).transpose(1, 0, 2)


def _band_mask(freqs: np.ndarray, low: float, high: float, final: bool) -> np.ndarray:
    return (freqs >= low) & (freqs <= high if final else freqs < high)


def fpsd_from_windows(windows: np.ndarray, config: WelchConfig) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized mean band power; output is [window, channel, band]."""
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3 or windows.shape[1] != len(CHANNELS):
        raise ValueError(f"Expected [N, 19, samples], got {windows.shape}")
    freqs, psd = welch(
        windows,
        fs=config.fs,
        window=config.window,
        nperseg=config.n_per_seg,
        noverlap=config.n_overlap,
        nfft=config.n_fft,
        detrend=config.detrend,
        axis=-1,
        scaling="density",
    )
    range_mask = (freqs >= config.fmin) & (freqs <= config.fmax)
    denom = psd[..., range_mask].sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(denom)) or np.any(denom <= 0):
        raise ValueError("At least one window/channel has non-positive spectral mass")
    relative = psd / denom
    values = []
    items = list(BANDS.items())
    for index, (_, (low, high)) in enumerate(items):
        mask = _band_mask(freqs, low, high, final=index == len(items) - 1)
        if not mask.any():
            raise ValueError(f"No Welch bins for band [{low}, {high}]")
        values.append(relative[..., mask].mean(axis=-1))
    tensor = np.stack(values, axis=-1).astype(np.float32)
    validate_feature_tensor(tensor)
    return tensor, freqs


def extract_paper95(data: np.ndarray, fs: float = 500.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = WelchConfig(fs=fs)
    windows = segment_nonoverlapping(data, fs, config.window_seconds)
    tensor, freqs = fpsd_from_windows(windows, config)
    return tensor, flatten_band_major(tensor), freqs


def flatten_band_major(tensor: np.ndarray) -> np.ndarray:
    """Flatten as delta(all channels), theta(all channels), ..., gamma."""
    tensor = np.asarray(tensor)
    validate_feature_tensor(tensor)
    flat = tensor.transpose(0, 2, 1).reshape(tensor.shape[0], -1)
    if flat.shape[1] != 95:
        raise AssertionError(f"Expected 95 values, got {flat.shape}")
    return flat.astype(np.float32, copy=False)


def feature_index_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for band, (low, high) in BANDS.items():
        for channel in CHANNELS:
            rows.append({"index": len(rows), "band": band, "channel": channel, "fmin": low, "fmax": high})
    return rows


def validate_feature_tensor(tensor: np.ndarray) -> None:
    if tensor.ndim != 3 or tensor.shape[1:] != (19, 5):
        raise ValueError(f"Expected [N, 19, 5], got {tensor.shape}")
    if not np.isfinite(tensor).all():
        raise ValueError("Features contain NaN or infinity")


def assert_deterministic(data: np.ndarray, fs: float = 500.0) -> None:
    first = extract_paper95(data, fs)[0]
    second = extract_paper95(data, fs)[0]
    np.testing.assert_array_equal(first, second)


def reorder_channels(data: np.ndarray, names: Iterable[str]) -> np.ndarray:
    names = tuple(names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate channel names")
    missing = [name for name in CHANNELS if name not in names]
    if missing:
        raise ValueError(f"Missing canonical channels: {missing}")
    return np.asarray(data)[[names.index(name) for name in CHANNELS]]
