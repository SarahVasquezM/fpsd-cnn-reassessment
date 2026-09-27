"""Subject-level split validation and internal validation construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_split(path: str | Path) -> dict:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split(split)
    return split


def _subjects(part: dict[str, list[str]]) -> list[str]:
    return [subject for group in ("AD", "CN") for subject in part[group]]


def validate_split(split: dict) -> None:
    train, test = set(_subjects(split["train"])), set(_subjects(split["test"]))
    overlap = train & test
    if overlap:
        raise ValueError(f"Subject leakage: {sorted(overlap)}")
    if len(train) != 43 or len(test) != 22 or len(train | test) != 65:
        raise ValueError(f"Unexpected split counts: train={len(train)}, test={len(test)}")
    expected = {"train_AD": 25, "train_CN": 18, "test_AD": 11, "test_CN": 11}
    actual = {
        "train_AD": len(split["train"]["AD"]), "train_CN": len(split["train"]["CN"]),
        "test_AD": len(split["test"]["AD"]), "test_CN": len(split["test"]["CN"]),
    }
    if actual != expected:
        raise ValueError(f"Published split counts changed: {actual}")


def subject_sets(split: dict) -> tuple[set[str], set[str]]:
    validate_split(split)
    return set(_subjects(split["train"])), set(_subjects(split["test"]))


def stratified_validation_subjects(split: dict, fraction: float = 0.2, seed: int = 2026) -> tuple[list[str], list[str]]:
    validate_split(split)
    subjects = np.array(_subjects(split["train"]))
    labels = np.array([1] * len(split["train"]["AD"]) + [0] * len(split["train"]["CN"]))
    train, validation = train_test_split(subjects, test_size=fraction, random_state=seed, stratify=labels)
    if set(train) & set(validation):
        raise AssertionError("Internal subject split overlaps")
    return sorted(train.tolist()), sorted(validation.tolist())
