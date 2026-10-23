"""Metrics with an explicit AD-positive label convention."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    prediction,
    p_positive=None,
    include_mcc: bool = True,
    positive_label: str = "AD",
    negative_label: str = "CN",
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result: dict[str, object] = {
        "n": len(y_true),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity_positive": _ratio(tp, tp + fn),
        "specificity_negative": _ratio(tn, tn + fp),
        "f1_positive": float(f1_score(y_true, prediction, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "confusion_matrix_labels": [negative_label, positive_label],
        "confusion_matrix": cm.tolist(),
    }
    # Named aliases preserve week-1 output compatibility while making CN-FTD explicit.
    result[f"sensitivity_{positive_label}"] = result["sensitivity_positive"]
    result[f"specificity_{negative_label}"] = result["specificity_negative"]
    result[f"f1_{positive_label}"] = result["f1_positive"]
    if include_mcc:
        result["mcc"] = float(matthews_corrcoef(y_true, prediction))
    if p_positive is not None and len(np.unique(y_true)) == 2:
        p_positive = np.asarray(p_positive, dtype=float)
        result["roc_auc"] = float(roc_auc_score(y_true, p_positive))
        result["pr_auc"] = float(average_precision_score(y_true, p_positive))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def subject_bootstrap_intervals(
    subject_predictions,
    resamples: int = 10_000,
    seed: int = 2026,
    positive_label: str = "AD",
    negative_label: str = "CN",
) -> dict[str, dict[str, float | None]]:
    """Percentile 95% intervals from subject-level resampling."""
    frame = subject_predictions.reset_index(drop=True)
    required = {"y_true", "prediction", "p_positive"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Subject predictions lack {sorted(missing)}")
    if len(frame) < 2 or resamples < 1:
        raise ValueError("Bootstrap needs at least two subjects and one resample")
    point = classification_metrics(
        frame.y_true, frame.prediction, frame.p_positive,
        positive_label=positive_label, negative_label=negative_label,
    )
    metric_names = ("balanced_accuracy", "accuracy", "sensitivity_positive", "specificity_negative", "macro_f1", "roc_auc", "pr_auc", "mcc")
    values = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        sample = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
        metrics = classification_metrics(
            sample.y_true, sample.prediction, sample.p_positive,
            positive_label=positive_label, negative_label=negative_label,
        )
        for name in metric_names:
            value = metrics.get(name)
            if value is not None and np.isfinite(value):
                values[name].append(float(value))
    intervals = {}
    for name in metric_names:
        samples = np.asarray(values[name], dtype=float)
        intervals[name] = {
            "estimate": None if point.get(name) is None else float(point[name]),
            "lower_95": float(np.quantile(samples, 0.025)) if len(samples) else None,
            "upper_95": float(np.quantile(samples, 0.975)) if len(samples) else None,
            "valid_resamples": len(samples),
        }
    return intervals
