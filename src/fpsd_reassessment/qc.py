"""Per-subject checks for derivative EEG files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import CHANNELS


def subject_qc(subject_id: str, path: str | Path, data: np.ndarray, fs: float, channel_names) -> dict[str, object]:
    if tuple(channel_names) != CHANNELS:
        raise ValueError("Channels are not in canonical order")
    samples_per_window = int(round(5.0 * fs))
    return {
        "subject_id": subject_id,
        "source_file": str(path),
        "source_bytes": Path(path).stat().st_size,
        "sfreq_hz": fs,
        "n_channels": int(data.shape[0]),
        "n_samples": int(data.shape[1]),
        "duration_seconds": float(data.shape[1] / fs),
        "n_complete_windows": int(data.shape[1] // samples_per_window),
        "has_nan": bool(np.isnan(data).any()),
        "has_infinite": bool(np.isinf(data).any()),
        "median_abs_uv": float(np.median(np.abs(data)) * 1e6),
        "p99_abs_uv": float(np.quantile(np.abs(data), 0.99) * 1e6),
        "status": "pass" if np.isfinite(data).all() and fs == 500.0 else "review",
    }
