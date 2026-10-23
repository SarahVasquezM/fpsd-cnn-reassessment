#!/usr/bin/env python
"""Create full and Git-friendly week-2 result archives."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

HEAVY_SUFFIXES = {".pth", ".pt", ".ckpt", ".parquet"}
REDUCED_EXCLUDED_NAMES = {"history.csv"}


def _archive(source: Path, target: Path, reduced: bool) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path == target:
                continue
            relative = path.relative_to(source)
            if reduced and (path.suffix in HEAVY_SUFFIXES or path.name in REDUCED_EXCLUDED_NAMES):
                continue
            archive.write(path, relative.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/kaggle/working/week2_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full = args.output_dir / "week2_results_bundle.zip"
    reduced = args.output_dir / "week2_github_results_bundle.zip"
    _archive(args.source, full, reduced=False)
    _archive(args.source, reduced, reduced=True)
    print(f"Full bundle: {full}")
    print(f"Reduced bundle: {reduced}")


if __name__ == "__main__":
    main()
