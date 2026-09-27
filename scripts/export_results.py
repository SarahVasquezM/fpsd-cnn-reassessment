#!/usr/bin/env python
"""Create full and GitHub-safe result archives."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from fpsd_reassessment.reproducibility import write_hashes


def copy_if_exists(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    write_hashes(args.results)
    shutil.make_archive(str(args.results.parent / "week1_results_bundle"), "zip", args.results)
    with tempfile.TemporaryDirectory() as temp_name:
        root = Path(temp_name)
        for source, relative in [
            (args.results / "metrics_by_seed_subject.csv", Path("results/metrics/metrics_by_seed_subject.csv")),
            (args.results / "metrics_aggregate_subject.csv", Path("results/metrics/metrics_aggregate_subject.csv")),
            (args.results / "metrics_by_seed_window.csv", Path("results/metrics/metrics_by_seed_window.csv")),
            (args.results / "metrics_aggregate_window.csv", Path("results/metrics/metrics_aggregate_window.csv")),
            (args.results / "figures", Path("reports/figures")),
            (args.repo / "reports/week1_baseline.md", Path("reports/week1_baseline.md")),
            (args.repo / "logs/decisions.md", Path("logs/decisions.md")),
            (args.results / "environment.json", Path("results/metrics/environment.json")),
            (args.results / "benchmark.json", Path("results/metrics/benchmark.json")),
            (args.results / "github_commit.txt", Path("results/metrics/github_commit.txt")),
        ]:
            copy_if_exists(source, root / relative)
        write_hashes(root)
        shutil.make_archive(str(args.results.parent / "github_results_bundle"), "zip", root)


if __name__ == "__main__":
    main()
