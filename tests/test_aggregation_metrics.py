import pandas as pd

from fpsd_reassessment.aggregation import aggregate_subjects
from fpsd_reassessment.metrics import classification_metrics


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
