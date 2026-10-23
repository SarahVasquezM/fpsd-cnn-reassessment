"""Window-to-subject probability aggregation."""

from __future__ import annotations

import pandas as pd


def aggregate_subjects(
    predictions: pd.DataFrame,
    threshold: float = 0.5,
    probability_column: str | None = None,
    positive_label: str = "AD",
) -> pd.DataFrame:
    if probability_column is None:
        probability_column = "p_AD" if "p_AD" in predictions.columns else "p_positive"
    required = {"subject_id", "y_true", probability_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    cardinality = predictions.groupby("subject_id")["y_true"].nunique()
    if (cardinality != 1).any():
        raise ValueError("A subject has conflicting labels")
    subject = predictions.groupby("subject_id", as_index=False).agg(
        y_true=("y_true", "first"),
        p_positive=(probability_column, "mean"),
        n_windows=(probability_column, "size"),
    )
    subject[f"p_{positive_label}"] = subject["p_positive"]
    subject["prediction"] = (subject["p_positive"] >= threshold).astype(int)
    return subject
