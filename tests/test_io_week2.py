import pandas as pd
import pytest

from fpsd_reassessment.io import read_participants


def test_read_all_preprocessed_diagnoses(tmp_path):
    frame = pd.DataFrame({
        "participant_id": [f"sub-{number:03d}" for number in range(1, 89)],
        "Group": ["A"] * 36 + ["C"] * 29 + ["F"] * 23,
    })
    path = tmp_path / "participants.tsv"
    frame.to_csv(path, sep="\t", index=False)
    participants = read_participants(path, diagnoses=("AD", "CN", "FTD"))
    assert participants.diagnosis.value_counts().to_dict() == {"AD": 36, "CN": 29, "FTD": 23}


def test_read_participants_rejects_incomplete_cohort(tmp_path):
    path = tmp_path / "participants.tsv"
    pd.DataFrame({"participant_id": ["sub-001"], "Group": ["A"]}).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Expected"):
        read_participants(path, diagnoses=("AD", "CN", "FTD"))
