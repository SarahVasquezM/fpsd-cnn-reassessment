"""Metrics with an explicit AD-positive label convention."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
)


def classification_metrics(y_true, prediction, p_ad=None, include_mcc: bool = True) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result: dict[str, object] = {
        "n": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity_AD": _ratio(tp, tp + fn),
        "specificity_CN": _ratio(tn, tn + fp),
        "f1_AD": float(f1_score(y_true, prediction, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "confusion_matrix_labels": ["CN", "AD"],
        "confusion_matrix": cm.tolist(),
    }
    if include_mcc:
        result["mcc"] = float(matthews_corrcoef(y_true, prediction))
    if p_ad is not None and len(np.unique(y_true)) == 2:
        p_ad = np.asarray(p_ad, dtype=float)
        result["roc_auc"] = float(roc_auc_score(y_true, p_ad))
        result["pr_auc"] = float(average_precision_score(y_true, p_ad))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan
