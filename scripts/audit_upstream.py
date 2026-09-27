#!/usr/bin/env python
"""Verify the frozen upstream evidence without modifying it."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from fpsd_reassessment.reproducibility import sha256_file


EXPECTED = {
    "1-data_processing/1-psd.py": "110de88e8dc1bfc63d9e0bab87737774ffe9bd87668a29b5846c2821a0e43399",
    "1-data_processing/2-fpsd-processing.py": "977edbf5e8544a3e2ed46e8990652a1b356fdcae7725c84985b40e99e5866356",
    "1-data_processing/3-Create-list.py": "225a69420ff033d4c01e28b64aadf9e128ee3a31167e3f575b244b4db4b47ede",
    "2-model/fpsd-cnn.py": "56530212c934baae660d16e7174927ee8d7c27857fdb90ada8cf83e5881b78fa",
    "model/AD-CN.pth": "b37a8197b8e59f077adbb0a52688d4438f1d97de332a626cc2ae09ffafd16817",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/audit/runtime_audit.csv"))
    args = parser.parse_args()
    rows = []
    for relative, expected in EXPECTED.items():
        actual = sha256_file(args.upstream / relative)
        rows.append({"check": f"sha256:{relative}", "status": "pass" if actual == expected else "fail", "evidence": actual})
    psd = (args.upstream / "1-data_processing/1-psd.py").read_text(encoding="utf-8")
    model = (args.upstream / "2-model/fpsd-cnn.py").read_text(encoding="utf-8")
    processing = (args.upstream / "1-data_processing/2-fpsd-processing.py").read_text(encoding="utf-8")
    creation = (args.upstream / "1-data_processing/3-Create-list.py").read_text(encoding="utf-8")
    checks = [
        ("extractor_six_bands", len(re.findall(r'"(?:delta|theta|alpha|low-beta|high-beta|gamma)"', psd)) == 6, "19×6=114"),
        ("next_stage_requires_95", "data.shape[1] != 95" in processing, "literal 95 guard"),
        ("source_conv1_128", "out_channels=128" in model, "source differs from article/checkpoint"),
        ("test_used_for_selection", "train(model, train_loader, test_loader" in model, "test_loader passed as val_loader"),
        ("drop_last_true", model.count("drop_last=True") >= 2, "train and test loaders"),
        ("csv_has_no_header", "writeheader" not in creation, "csv.writer rows only"),
        ("training_has_no_to_device", ".to(device)" not in model, "training remains on CPU"),
        ("unsorted_files", "os.listdir" in creation and "sorted(" not in creation, "filesystem order controls labels/rows"),
        ("seeds_absent", "manual_seed" not in model and "random.seed" not in model, "no deterministic seeds"),
    ]
    rows.extend({"check": name, "status": "pass" if passed else "fail", "evidence": evidence} for name, passed, evidence in checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "evidence"])
        writer.writeheader()
        writer.writerows(rows)
    if any(row["status"] == "fail" for row in rows):
        raise SystemExit("One or more upstream audit checks failed; inspect the CSV")


if __name__ == "__main__":
    main()
