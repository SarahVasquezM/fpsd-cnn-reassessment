"""Dataset metadata, EEGLAB input, and compact bundle I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CHANNELS, flatten_band_major, reorder_channels


def read_participants(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={"participant_id": str, "Group": str})
    required = {"participant_id", "Group"}
    if not required.issubset(frame.columns):
        raise ValueError(f"participants.tsv lacks {sorted(required - set(frame.columns))}")
    frame["diagnosis"] = frame["Group"].map({"A": "AD", "C": "CN", "F": "FTD"})
    subset = frame[frame["Group"].isin(["A", "C"])].copy()
    counts = subset["diagnosis"].value_counts().to_dict()
    if counts != {"AD": 36, "CN": 29} or len(subset) != 65:
        raise ValueError(f"Expected 36 AD + 29 CN, observed {counts}")
    return subset


def find_preprocessed_set(dataset_root: str | Path, subject_id: str) -> Path:
    root = Path(dataset_root)
    candidates = sorted(root.glob(f"derivatives/**/{subject_id}*eyesclosed*eeg.set"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one derivative .set for {subject_id}, found {candidates}")
    return candidates[0]


def load_eeglab(path: str | Path) -> tuple[np.ndarray, float, tuple[str, ...], object]:
    import mne

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    eeg_indices = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude=[])
    names = tuple(raw.ch_names[index] for index in eeg_indices)
    data = raw.get_data(picks=eeg_indices)
    data = reorder_channels(data, names)
    return data, float(raw.info["sfreq"]), CHANNELS, raw


def save_feature_bundle(output_dir: str | Path, features: np.ndarray, metadata: pd.DataFrame) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = features.astype(np.float32)
    np.savez_compressed(output / "paper95_features.npz", X=features, X_flat=flatten_band_major(features))
    metadata.to_parquet(output / "windows_metadata.parquet", index=False)


def load_feature_bundle(input_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    root = Path(input_dir)
    with np.load(root / "paper95_features.npz") as archive:
        tensor = archive["X"]
        features = archive["X_flat"] if "X_flat" in archive.files else flatten_band_major(tensor)
    metadata = pd.read_parquet(root / "windows_metadata.parquet")
    if len(features) != len(metadata):
        raise ValueError("Feature and metadata row counts differ")
    return features, metadata
