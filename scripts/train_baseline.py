#!/usr/bin/env python
"""Benchmark and run forensic plus valid baselines on Kaggle GPU."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from fpsd_reassessment.io import load_feature_bundle
from fpsd_reassessment.models import validate_architecture
from fpsd_reassessment.reproducibility import environment_record, git_commit, verify_hashes, write_hashes, write_json
from fpsd_reassessment.splits import load_split
from fpsd_reassessment.training import save_train_result, select_device, train_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/baseline_ad_cn.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/week1_results"))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--allow-cpu-smoke", action="store_true", help="Only for tests; never report as a full run")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    verify_hashes(args.features)
    device = select_device(require_gpu=not args.allow_cpu_smoke)
    count = validate_architecture()
    print(f"device={device}; gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else None}; parameters={count}")
    features, metadata = load_feature_bundle(args.features)
    split = load_split(config["split_file"])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output / "effective_experiment_config.yaml")
    shutil.copy2(config["feature_config"], output / "effective_feature_config.yaml")
    shutil.copy2(config["model_config"], output / "effective_model_config.yaml")
    shutil.copy2(Path("reports/week1_baseline.md"), output / "preliminary_report.md")
    write_json(output / "environment.json", environment_record())
    commit = git_commit(".")
    (output / "github_commit.txt").write_text(commit + "\n", encoding="utf-8")

    benchmark = None
    if not args.skip_benchmark:
        before = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        result = train_one(
            features, metadata, split, "valid_subject_disjoint", 2026, epochs=10,
            require_gpu=not args.allow_cpu_smoke, evaluate_test=False,
        )
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        estimate_seconds = result.elapsed_seconds / 10 * 100 * 2 * 3
        benchmark = {
            "benchmark_epochs": 10, "elapsed_seconds": result.elapsed_seconds,
            "seconds_per_epoch": result.elapsed_seconds / 10,
            "estimated_seconds_two_protocols_three_seeds": estimate_seconds,
            "estimated_hours": estimate_seconds / 3600,
            "gpu_peak_memory_bytes": max(0, peak - before),
            "fits_10h_with_10pct_margin": estimate_seconds <= 9 * 3600,
        }
        write_json(output / "benchmark.json", benchmark)
    seeds = args.seed or config["training"]["seeds"]
    if benchmark and not benchmark["fits_10h_with_10pct_margin"] and args.seed is None:
        seeds = [config["training"]["seeds"][0]]
        print("Estimate exceeds the safe session budget; this version will run seed 2026 only. Rerun with --seed 2027 and --seed 2028 in separate versions.")
    summary_rows = []
    for protocol in ("forensic_test_selected", "valid_subject_disjoint"):
        for seed in seeds:
            result = train_one(
                features, metadata, split, protocol, int(seed),
                epochs=int(config["training"]["max_epochs"]),
                batch_size=int(config["training"]["batch_size"]),
                learning_rate=float(config["training"]["learning_rate"]),
                require_gpu=not args.allow_cpu_smoke,
            )
            save_train_result(result, output, protocol, int(seed), commit)
            summary_rows.append({"protocol": protocol, "seed": seed, **result.subject_metrics, "elapsed_seconds": result.elapsed_seconds})
    pd.DataFrame(summary_rows).to_csv(output / "metrics_by_seed_subject.csv", index=False)
    write_hashes(output)
    print(json.dumps({"completed_seeds": seeds, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
