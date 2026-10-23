import pandas as pd
import pytest

from fpsd_reassessment.aggregation import aggregate_subjects
from fpsd_reassessment.metrics import classification_metrics
from fpsd_reassessment.training import subject_class_window_weights


def test_subject_mean_probability_and_metrics():
    windows = pd.DataFrame({
        "subject_id": ["sub-001", "sub-001", "sub-037", "sub-037"],
        "y_true": [1, 1, 0, 0],
        "p_AD": [0.9, 0.7, 0.1, 0.3],
    })
    subjects = aggregate_subjects(windows)
    assert subjects.set_index("subject_id").loc["sub-001", "p_AD"] == 0.8
    metrics = classification_metrics(subjects.y_true, subjects.prediction, subjects.p_AD)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["sensitivity_AD"] == 1.0
    assert metrics["specificity_CN"] == 1.0


def test_cn_ftd_probability_alias_and_metrics():
    windows = pd.DataFrame({
        "subject_id": ["sub-037", "sub-037", "sub-066", "sub-066"],
        "y_true": [0, 0, 1, 1],
        "p_positive": [0.1, 0.2, 0.8, 0.9],
    })
    subjects = aggregate_subjects(windows, positive_label="FTD")
    assert "p_FTD" in subjects
    metrics = classification_metrics(
        subjects.y_true, subjects.prediction, subjects.p_positive,
        positive_label="FTD", negative_label="CN",
    )
    assert metrics["sensitivity_FTD"] == 1.0
    assert metrics["specificity_CN"] == 1.0


def test_subject_class_weights_equalize_class_and_subject_mass():
    metadata = pd.DataFrame({
        "subject_id": ["cn1"] * 2 + ["cn2"] * 4 + ["ad1"] * 3,
        "y_true": [0] * 6 + [1] * 3,
    })
    weights = subject_class_window_weights(metadata, metadata.index.to_numpy())
    weighted = metadata.assign(weight=weights)
    class_mass = weighted.groupby("y_true").weight.sum()
    subject_mass = weighted.groupby(["y_true", "subject_id"]).weight.sum()
    assert class_mass.loc[0] == pytest.approx(class_mass.loc[1])
    assert subject_mass.loc[(0, "cn1")] == pytest.approx(subject_mass.loc[(0, "cn2")])
