#!/usr/bin/env python
"""Build a Kaggle-ready source archive with pyproject.toml at the ZIP root."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from fpsd_reassessment.reproducibility import sha256_file, source_tree_sha256

INCLUDE_DIRECTORIES = ("configs", "docs", "notebooks", "reports/audit", "scripts", "splits", "src", "tests")
INCLUDE_ROOT_FILES = (
    ".gitattributes", ".gitignore", "CITATION.cff", "README.md", "pyproject.toml",
    "requirements-kaggle.txt", "reports/week1_baseline.md", "reports/week2_grouped_validation.md",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "graphify-out", ".venv"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pth", ".pt", ".ckpt", ".npz", ".npy", ".parquet", ".zip"}


def _files(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for relative in INCLUDE_ROOT_FILES:
        path = root / relative
        if path.is_file():
            paths.add(path)
    for directory in INCLUDE_DIRECTORIES:
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or set(path.relative_to(root).parts) & EXCLUDED_PARTS:
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            paths.add(path)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = _files(root)
    if not files:
        raise ValueError(f"No source files found in {root}")
    manifest_lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files]
    tree_hash = source_tree_sha256(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("SOURCE_TREE_SHA256.txt", tree_hash + "\n")
        archive.writestr("CODE_SHA256SUMS.txt", "".join(manifest_lines))
    archive_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Archive: {args.output}")
    print(f"Files: {len(files)}")
    print(f"Source tree SHA-256: {tree_hash}")
    print(f"Archive SHA-256: {archive_hash}")


if __name__ == "__main__":
    main()
