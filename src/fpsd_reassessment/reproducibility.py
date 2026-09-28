"""Hashes, environment capture, and deterministic controls."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: str | Path, exclude_name: str = "SHA256SUMS.txt") -> dict[str, str]:
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file() and path.name != exclude_name
    }


def write_hashes(root: str | Path) -> Path:
    root = Path(root)
    target = root / "SHA256SUMS.txt"
    target.write_text("".join(f"{digest}  {name}\n" for name, digest in hash_tree(root).items()), encoding="utf-8")
    return target


def verify_hashes(root: str | Path) -> None:
    root = Path(root)
    lines = (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"Hash mismatch for {relative}: {actual} != {expected}")


def git_commit(root: str | Path = ".") -> str:
    root = Path(root)
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        source_commit = root / "SOURCE_COMMIT"
        if source_commit.exists():
            value = source_commit.read_text(encoding="utf-8").strip().lower()
            if not re.fullmatch(r"[0-9a-f]{40}", value):
                raise ValueError(f"Invalid packaged source commit: {value!r}")
            return value
        return "UNCOMMITTED"


def environment_record() -> dict[str, object]:
    import importlib.metadata
    import psutil

    packages = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "mne", "torch", "pyarrow", "openneuro-py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "disk_free_bytes": shutil.disk_usage("/").free,
        "packages": packages,
        "kaggle_kernel_run_type": os.getenv("KAGGLE_KERNEL_RUN_TYPE"),
        "kaggle_url_base": os.getenv("KAGGLE_URL_BASE"),
    }
    try:
        import torch
        record["torch_cuda_available"] = torch.cuda.is_available()
        record["torch_cuda_version"] = torch.version.cuda
        record["cudnn_version"] = torch.backends.cudnn.version()
        record["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return record


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
