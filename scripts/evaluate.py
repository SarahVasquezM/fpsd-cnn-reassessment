#!/usr/bin/env python
"""Aggregate completed independent seed results without changing the experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    subject_records, window_records = [], []
    for subject_path in sorted(args.results.glob("*/seed_*/subject_metrics.json")):
        protocol, seed_dir = subject_path.parent.parent.name, subject_path.parent.name
        seed = int(seed_dir.removeprefix("seed_"))
        subject_records.append({"protocol": protocol, "seed": seed, **json.loads(subject_path.read_text(encoding="utf-8"))})
        window_path = subject_path.with_name("window_metrics.json")
        window_records.append({"protocol": protocol, "seed": seed, **json.loads(window_path.read_text(encoding="utf-8"))})
    subject_frame = pd.DataFrame(subject_records)
    window_frame = pd.DataFrame(window_records)
    if subject_frame.empty:
        raise SystemExit("No completed subject_metrics.json files found")
    subject_frame.to_csv(args.results / "metrics_by_seed_subject.csv", index=False)
    window_frame.to_csv(args.results / "metrics_by_seed_window.csv", index=False)
    for unit, frame in (("subject", subject_frame), ("window", window_frame)):
        excluded = {"protocol", "seed", "confusion_matrix", "confusion_matrix_labels"}
        numeric = [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]
        aggregate = frame.groupby("protocol")[numeric].agg(["mean", "std"])
        aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
        aggregate.reset_index().to_csv(args.results / f"metrics_aggregate_{unit}.csv", index=False)


if __name__ == "__main__":
    main()
