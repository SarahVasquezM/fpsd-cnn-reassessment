from pathlib import Path

import pandas as pd
import pytest

from fpsd_reassessment.splits import (
    load_split,
    make_grouped_folds,
    make_loso_folds,
    stratified_validation_subjects,
    task_subject_table,
    validate_split,
)

ROOT = Path(__file__).parents[1]


def test_published_split_counts_and_no_overlap():
    split = load_split(ROOT / "splits/original_holdout.json")
    train = set(split["train"]["AD"] + split["train"]["CN"])
    test = set(split["test"]["AD"] + split["test"]["CN"])
    assert len(train) == 43
    assert len(test) == 22
    assert not train & test


def test_internal_validation_is_subject_disjoint_and_deterministic():
    split = load_split(ROOT / "splits/original_holdout.json")
    first = stratified_validation_subjects(split)
    second = stratified_validation_subjects(split)
    assert first == second
    assert not set(first[0]) & set(first[1])


def test_leakage_fails():
    split = load_split(ROOT / "splits/original_holdout.json")
    split["test"]["AD"].append(split["train"]["AD"][0])
    with pytest.raises(ValueError, match="leakage"):
        validate_split(split)


def test_published_cn_ftd_split_counts_and_no_overlap():
    split = load_split(ROOT / "splits/published_holdout_cn_ftd.json")
    train = set(split["train"]["CN"] + split["train"]["FTD"])
    test = set(split["test"]["CN"] + split["test"]["FTD"])
    assert len(train) == 35
    assert len(test) == 17
    assert not train & test


def test_grouped_and_loso_cover_each_subject_once():
    metadata = pd.DataFrame({
        "subject_id": [f"sub-{number:03d}" for number in range(1, 37)] + [f"sub-{number:03d}" for number in range(37, 66)],
        "diagnosis": ["AD"] * 36 + ["CN"] * 29,
    })
    subjects = task_subject_table(metadata, "ad_cn")
    grouped = make_grouped_folds(subjects)
    loso = make_loso_folds(subjects)
    assert len(grouped) == 5
    assert len(loso) == 65
    assert sorted(subject for fold in grouped for subject in fold["test_subjects"]) == sorted(subjects.subject_id)
    assert sorted(subject for fold in loso for subject in fold["test_subjects"]) == sorted(subjects.subject_id)
