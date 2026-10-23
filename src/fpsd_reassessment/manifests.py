"""Compatibility helpers for persisted experiment manifests."""

from __future__ import annotations


def fold_index(manifest: dict) -> int:
    """Return the numeric fold index from current or legacy manifests."""
    fold = manifest["fold"]
    if isinstance(fold, dict):
        fold = fold["fold"]
    return int(fold)
