#!/usr/bin/env python
"""Create the paper95 feature bundle from preprocessed EEG derivatives."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fpsd_reassessment.features import (
    WelchConfig,
    feature_index_rows,
    flatten_band_major,
    fpsd_from_windows,
    segment_nonoverlapping,
)
from fpsd_reassessment.io import (
    find_preprocessed_set,
    load_eeglab,
    read_participants,
    save_feature_bundle,
)
from fpsd_reassessment.qc import subject_qc
from fpsd_reassessment.reproducibility import (
    environment_record,
    sha256_file,
    write_hashes,
    write_json,
)

DATASET = "ds004504"
TAG = "1.0.9"
OPENNEURO_COMMIT = "63b2ec152bc3cd4487d131124f10f767ebe543bc"


SUBJECT_RANGES = {"AD": range(1, 37), "CN": range(37, 66), "FTD": range(66, 89)}


def download(dataset_dir: Path, diagnoses: tuple[str, ...]) -> None:
    includes = ["participants.tsv", "participants.json", "dataset_description.json", "CHANGES"]
    subject_numbers = sorted({number for diagnosis in diagnoses for number in SUBJECT_RANGES[diagnosis]})
    includes.extend(
        f"derivatives/sub-{number:03d}/eeg/sub-{number:03d}_task-eyesclosed_eeg.set"
        for number in subject_numbers
    )
    command = ["openneuro-py", "download", f"--dataset={DATASET}", f"--tag={TAG}", f"--target-dir={dataset_dir}"]
    for relative in includes:
        command.append(f"--include={relative}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("/kaggle/working/ds004504"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/fpsd_features"))
    parser.add_argument("--config", type=Path, default=Path("configs/features/paper95.yaml"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--diagnoses", nargs="+", choices=sorted(SUBJECT_RANGES), default=["AD", "CN"],
        help="Diagnoses to include. Week 2 uses: --diagnoses AD CN FTD",
    )
    args = parser.parse_args()
    diagnoses = tuple(dict.fromkeys(args.diagnoses))
    if not args.skip_download:
        download(args.dataset_dir, diagnoses)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    participants = read_participants(args.dataset_dir / "participants.tsv", diagnoses=diagnoses)
    tensors, metadata_rows, qc_rows, sources = [], [], [], []
    ordered_participants = participants.sort_values("participant_id")
    for subject_index, row in enumerate(ordered_participants.itertuples(index=False), start=1):
        print(f"[{subject_index}/{len(ordered_participants)}] Procesando {row.participant_id}", flush=True)
        source = find_preprocessed_set(args.dataset_dir, row.participant_id)
        data, fs, channels, _ = load_eeglab(source)
        source_relative = source.relative_to(args.dataset_dir).as_posix()
        source_hash = sha256_file(source)
        if fs != 500.0:
            raise ValueError(f"{row.participant_id}: expected 500 Hz, observed {fs}")
        windows = segment_nonoverlapping(data, fs, 5.0)
        tensor, _ = fpsd_from_windows(windows, WelchConfig(fs=fs))
        if not tensors:
            repeated, _ = fpsd_from_windows(windows, WelchConfig(fs=fs))
            np.testing.assert_array_equal(tensor, repeated, err_msg="Feature extraction is not deterministic")
        flat = flatten_band_major(tensor)
        tensors.append(tensor)
        # Both disease groups are positive relative to CN. Week 2 training filters
        # one binary task at a time and derives its labels again from diagnosis.
        label = 0 if row.diagnosis == "CN" else 1
        for window_id in range(len(tensor)):
            metadata_rows.append({
                "subject_id": row.participant_id, "window_id": window_id,
                "start_seconds": window_id * 5.0, "stop_seconds": (window_id + 1) * 5.0,
                "diagnosis": row.diagnosis, "y_true": label,
                "source_file": source_relative,
                "source_sha256": source_hash,
            })
        qc = subject_qc(row.participant_id, source, data, fs, channels)
        qc["source_file"] = source_relative
        qc_rows.append({**qc, "diagnosis": row.diagnosis})
        sources.append({"path": source_relative, "sha256": source_hash, "bytes": source.stat().st_size})
        if flat.shape != (len(tensor), 95):
            raise AssertionError(f"{row.participant_id}: bad flat shape {flat.shape}")
    all_tensor = np.concatenate(tensors, axis=0)
    metadata = pd.DataFrame(metadata_rows)
    save_feature_bundle(output, all_tensor, metadata)
    pd.DataFrame(qc_rows).to_csv(output / "subjects_qc.csv", index=False)
    pd.DataFrame(feature_index_rows()).to_csv(output / "feature_index.tsv", sep="\t", index=False)
    shutil.copy2(args.config, output / "feature_config.yaml")
    write_json(output / "environment.json", environment_record())
    write_json(output / "dataset_manifest.json", {
        "dataset": DATASET, "version": TAG, "openneuro_git_commit": OPENNEURO_COMMIT,
        "participants_sha256": sha256_file(args.dataset_dir / "participants.tsv"),
        "subject_counts": {
            **{diagnosis: int((participants["diagnosis"] == diagnosis).sum()) for diagnosis in diagnoses},
            "total": len(participants),
        },
        "n_windows": len(all_tensor), "tensor_shape": list(all_tensor.shape),
        "sources": sources,
    })
    (output / "README.md").write_text(
        "# paper95 feature bundle\n\nGenerated from preprocessed OpenNeuro ds004504 v1.0.9 derivatives. "
        "The master array is `X[window, channel, band]`; metadata preserves subject and window identity.\n",
        encoding="utf-8",
    )
    write_hashes(output)
    archive_name = "fpsd_features_all_bundle" if set(diagnoses) == set(SUBJECT_RANGES) else "fpsd_features_bundle"
    archive = shutil.make_archive(str(output.parent / archive_name), "zip", output)
    print(json.dumps({"bundle": archive, "shape": list(all_tensor.shape), "subjects": len(participants)}, indent=2))


if __name__ == "__main__":
    main()
