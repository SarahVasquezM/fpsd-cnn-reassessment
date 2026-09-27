import json
from pathlib import Path

import pytest

from fpsd_reassessment.splits import load_split, stratified_validation_subjects, validate_split


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
