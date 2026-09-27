"""Window-to-subject probability aggregation."""

from __future__ import annotations

import pandas as pd


def aggregate_subjects(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    required = {"subject_id", "y_true", "p_AD"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    cardinality = predictions.groupby("subject_id")["y_true"].nunique()
    if (cardinality != 1).any():
        raise ValueError("A subject has conflicting labels")
    subject = predictions.groupby("subject_id", as_index=False).agg(
        y_true=("y_true", "first"), p_AD=("p_AD", "mean"), n_windows=("p_AD", "size")
    )
    subject["prediction"] = (subject["p_AD"] >= threshold).astype(int)
    return subject
