"""Subject-level split validation and deterministic resampling protocols."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

TASK_LABELS = {
    "ad_cn": {"negative": "CN", "positive": "AD"},
    "cn_ftd": {"negative": "CN", "positive": "FTD"},
}


def load_split(path: str | Path) -> dict:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split(split)
    return split


def split_classes(split: dict) -> tuple[str, str]:
    train_keys = set(split.get("train", {}))
    test_keys = set(split.get("test", {}))
    if train_keys != test_keys or len(train_keys) != 2:
        raise ValueError(f"Expected the same two classes in train/test, observed {train_keys} and {test_keys}")
    negative = split.get("negative_label", "CN")
    positive_candidates = train_keys - {negative}
    if negative not in train_keys or len(positive_candidates) != 1:
        raise ValueError(f"Cannot infer positive/negative labels from {sorted(train_keys)}")
    return negative, next(iter(positive_candidates))


def _subjects(part: dict[str, list[str]], classes: tuple[str, str]) -> list[str]:
    return [subject for diagnosis in classes for subject in part[diagnosis]]


def validate_split(split: dict) -> None:
    classes = split_classes(split)
    train_list = _subjects(split["train"], classes)
    test_list = _subjects(split["test"], classes)
    if len(train_list) != len(set(train_list)) or len(test_list) != len(set(test_list)):
        raise ValueError("A subject is duplicated within a partition")
    train, test = set(train_list), set(test_list)
    overlap = train & test
    if overlap:
        raise ValueError(f"Subject leakage: {sorted(overlap)}")
    counts = split.get("counts", {})
    actual = {
        **{f"train_{diagnosis}": len(split["train"][diagnosis]) for diagnosis in classes},
        **{f"test_{diagnosis}": len(split["test"][diagnosis]) for diagnosis in classes},
        "total": len(train | test),
    }
    expected = {key: counts[key] for key in actual if key in counts}
    if expected and any(actual[key] != value for key, value in expected.items()):
        raise ValueError(f"Published split counts changed: expected={expected}, actual={actual}")


def subject_sets(split: dict) -> tuple[set[str], set[str]]:
    validate_split(split)
    classes = split_classes(split)
    return set(_subjects(split["train"], classes)), set(_subjects(split["test"], classes))


def task_labels(task: str) -> tuple[str, str]:
    try:
        labels = TASK_LABELS[task]
    except KeyError as error:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_LABELS)}") from error
    return labels["negative"], labels["positive"]


def task_subject_table(metadata: pd.DataFrame, task: str) -> pd.DataFrame:
    negative, positive = task_labels(task)
    required = {"subject_id", "diagnosis"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata lacks {sorted(missing)}")
    rows = metadata.loc[metadata["diagnosis"].isin([negative, positive]), ["subject_id", "diagnosis"]]
    cardinality = rows.groupby("subject_id")["diagnosis"].nunique()
    if (cardinality != 1).any():
        raise ValueError("A subject has conflicting diagnoses")
    subjects = rows.drop_duplicates().sort_values("subject_id").reset_index(drop=True)
    subjects["y_true"] = (subjects["diagnosis"] == positive).astype(int)
    return subjects


def make_grouped_folds(subjects: pd.DataFrame, n_splits: int = 5, seed: int = 2026) -> list[dict]:
    if not {"subject_id", "diagnosis", "y_true"}.issubset(subjects.columns):
        raise ValueError("Subject table must contain subject_id, diagnosis and y_true")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(subjects.subject_id, subjects.y_true)):
        folds.append(_fold_record(fold_index, subjects.iloc[train_index], subjects.iloc[test_index]))
    validate_outer_folds(folds, set(subjects.subject_id), require_single_test_appearance=True)
    return folds


def make_loso_folds(subjects: pd.DataFrame) -> list[dict]:
    ordered = subjects.sort_values("subject_id").reset_index(drop=True)
    folds = []
    for fold_index in range(len(ordered)):
        folds.append(_fold_record(fold_index, ordered.drop(index=fold_index), ordered.iloc[[fold_index]]))
    validate_outer_folds(folds, set(ordered.subject_id), require_single_test_appearance=True)
    return folds


def _fold_record(fold_index: int, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "fold": int(fold_index),
        "train_subjects": sorted(train.subject_id.tolist()),
        "test_subjects": sorted(test.subject_id.tolist()),
        "train_counts": train.diagnosis.value_counts().sort_index().to_dict(),
        "test_counts": test.diagnosis.value_counts().sort_index().to_dict(),
    }


def validate_outer_folds(
    folds: list[dict], all_subjects: set[str], require_single_test_appearance: bool = False,
) -> None:
    test_appearances: list[str] = []
    for fold in folds:
        train, test = set(fold["train_subjects"]), set(fold["test_subjects"])
        if train & test:
            raise ValueError(f"Subject leakage in fold {fold['fold']}")
        if train | test != all_subjects:
            raise ValueError(f"Fold {fold['fold']} does not cover the complete task cohort")
        test_appearances.extend(fold["test_subjects"])
    if require_single_test_appearance:
        counts = pd.Series(test_appearances).value_counts().to_dict()
        if set(counts) != all_subjects or set(counts.values()) != {1}:
            raise ValueError("Outer folds must test every subject exactly once")


def stratified_subject_partition(
    subjects: list[str] | np.ndarray,
    labels: list[int] | np.ndarray,
    fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[list[str], list[str]]:
    subject_array = np.asarray(subjects)
    label_array = np.asarray(labels, dtype=int)
    train, validation = train_test_split(
        subject_array, test_size=fraction, random_state=seed, stratify=label_array,
    )
    if set(train) & set(validation):
        raise AssertionError("Internal subject split overlaps")
    return sorted(train.tolist()), sorted(validation.tolist())


def stratified_validation_subjects(
    split: dict, fraction: float = 0.2, seed: int = 2026,
) -> tuple[list[str], list[str]]:
    validate_split(split)
    negative, positive = split_classes(split)
    subjects = split["train"][negative] + split["train"][positive]
    labels = [0] * len(split["train"][negative]) + [1] * len(split["train"][positive])
    return stratified_subject_partition(subjects, labels, fraction=fraction, seed=seed)
