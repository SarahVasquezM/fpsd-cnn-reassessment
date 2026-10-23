#!/usr/bin/env python
"""Run published holdout, grouped 5-fold, or true LOSO evaluation on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from fpsd_reassessment.aggregation import aggregate_subjects
from fpsd_reassessment.io import load_feature_bundle
from fpsd_reassessment.manifests import fold_index
from fpsd_reassessment.metrics import (
    classification_metrics,
    subject_bootstrap_intervals,
)
from fpsd_reassessment.models import validate_architecture
from fpsd_reassessment.reproducibility import (
    environment_record,
    git_commit,
    sha256_file,
    source_tree_sha256,
    verify_hashes,
    write_hashes,
    write_json,
)
from fpsd_reassessment.splits import (
    load_split,
    make_grouped_folds,
    make_loso_folds,
    split_classes,
    stratified_subject_partition,
    task_labels,
    task_subject_table,
)
from fpsd_reassessment.training import select_device, train_subject_split


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/week2_subject_evaluation.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/week2_results"))
    parser.add_argument("--task", choices=["ad_cn", "cn_ftd"], required=True)
    parser.add_argument("--evaluation", choices=["published_holdout", "grouped5", "loso"], required=True)
    parser.add_argument("--balance", choices=["none", "subject_class"], default="none")
    parser.add_argument("--protocol", choices=["auto", "forensic", "valid"], default="auto")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--fold-start", type=int, default=0, help="Inclusive outer-fold index")
    parser.add_argument("--fold-stop", type=int, help="Exclusive outer-fold index")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-folds", type=int, default=5)
    parser.add_argument("--allow-cpu-smoke", action="store_true", help="Tests only; never report as a scientific run")
    return parser.parse_args()


def _task_data(features: np.ndarray, metadata: pd.DataFrame, task: str, expected: dict) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    negative, positive = task_labels(task)
    mask = metadata["diagnosis"].isin([negative, positive]).to_numpy()
    task_metadata = metadata.loc[mask].copy().reset_index(drop=True)
    task_features = features[mask]
    task_metadata["y_true"] = (task_metadata["diagnosis"] == positive).astype(int)
    subjects = task_subject_table(task_metadata, task)
    actual = subjects.diagnosis.value_counts().to_dict()
    expected_classes = {key: int(value) for key, value in expected.items() if key != "total"}
    if actual != expected_classes or len(subjects) != int(expected["total"]):
        raise ValueError(f"Unexpected task cohort for {task}: expected={expected}, actual={actual}")
    return task_features, task_metadata, subjects


def _outer_folds(task: str, evaluation: str, subjects: pd.DataFrame, config: dict) -> list[dict]:
    if evaluation == "grouped5":
        return make_grouped_folds(
            subjects,
            n_splits=int(config["evaluation"]["grouped_folds"]),
            seed=int(config["evaluation"]["grouped_seed"]),
        )
    if evaluation == "loso":
        return make_loso_folds(subjects)
    split = load_split(config["published_splits"][task])
    negative, positive = split_classes(split)
    expected_negative, expected_positive = task_labels(task)
    if (negative, positive) != (expected_negative, expected_positive):
        raise ValueError(f"Published split labels do not match task {task}")
    return [{
        "fold": 0,
        "train_subjects": sorted(split["train"][negative] + split["train"][positive]),
        "test_subjects": sorted(split["test"][negative] + split["test"][positive]),
        "train_counts": {key: len(value) for key, value in split["train"].items()},
        "test_counts": {key: len(value) for key, value in split["test"].items()},
        "published_split_name": split["name"],
    }]


def _protocols(evaluation: str, requested: str) -> list[str]:
    if requested == "auto":
        return ["forensic", "valid"] if evaluation == "published_holdout" else ["valid"]
    if requested == "forensic" and evaluation != "published_holdout":
        raise ValueError("Forensic checkpoint selection is only defined for the published holdout")
    return [requested]


def _inner_partition(fold: dict, subjects: pd.DataFrame, protocol: str, config: dict) -> tuple[set[str], set[str], set[str]]:
    outer_train = set(fold["train_subjects"])
    outer_test = set(fold["test_subjects"])
    if protocol == "forensic":
        return outer_train, outer_test, outer_test
    inner = subjects[subjects.subject_id.isin(outer_train)]
    fit, validation = stratified_subject_partition(
        inner.subject_id.tolist(), inner.y_true.tolist(),
        fraction=float(config["training"]["validation_fraction_subjects"]),
        seed=int(config["training"]["validation_seed"]),
    )
    return set(fit), set(validation), outer_test


def _signature(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _save_result(result, root: Path, manifest: dict, commit: str) -> None:
    root.mkdir(parents=True, exist_ok=False)
    result.history.to_csv(root / "history.csv", index=False)
    predictions = result.predictions.copy()
    for key in ("task", "evaluation", "balance", "protocol", "seed", "fold"):
        predictions[key] = manifest[key]
    predictions["github_commit"] = commit
    predictions.to_parquet(root / "window_predictions.parquet", index=False)
    subjects = aggregate_subjects(predictions, positive_label=result.positive_label)
    for key in ("task", "evaluation", "balance", "protocol", "seed", "fold"):
        subjects[key] = manifest[key]
    subjects.to_csv(root / "subject_predictions.csv", index=False)
    write_json(root / "window_metrics.json", result.window_metrics)
    write_json(root / "subject_metrics.json", result.subject_metrics)
    write_json(root / "split_manifest.json", manifest)
    write_json(root / "runtime.json", {"elapsed_seconds": result.elapsed_seconds, "best_epoch": result.best_epoch})
    torch.save(result.state_dict, root / "best_model.pth")
    write_json(root / "completed.json", {"signature": manifest["signature"], "completed": True})


def _existing_complete(root: Path, signature: str, resume: bool) -> bool:
    marker = root / "completed.json"
    if not root.exists():
        return False
    if not resume:
        raise FileExistsError(f"Output already exists; use --resume after verifying it: {root}")
    if not marker.is_file():
        raise RuntimeError(f"Incomplete fold exists and is not safe to overwrite: {root}")
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    if recorded.get("signature") != signature:
        raise RuntimeError(f"Resume signature mismatch in {root}")
    required = ["window_predictions.parquet", "subject_predictions.csv", "window_metrics.json", "subject_metrics.json"]
    if not all((root / name).is_file() for name in required):
        raise RuntimeError(f"Completed marker has missing artifacts in {root}")
    return True


def _aggregate_completed(root: Path, task: str, evaluation: str, balance: str, negative: str, positive: str, resamples: int) -> None:
    prediction_files = sorted(root.glob("*/seed_*/fold_*/window_predictions.parquet"))
    if not prediction_files:
        return
    windows = pd.concat([pd.read_parquet(path) for path in prediction_files], ignore_index=True)
    summary_rows = []
    fold_rows = []
    article_rows = []
    for prediction_path in prediction_files:
        fold_root = prediction_path.parent
        window_metrics = json.loads((fold_root / "window_metrics.json").read_text(encoding="utf-8"))
        subject_metrics = json.loads((fold_root / "subject_metrics.json").read_text(encoding="utf-8"))
        runtime = json.loads((fold_root / "runtime.json").read_text(encoding="utf-8"))
        manifest = json.loads((fold_root / "split_manifest.json").read_text(encoding="utf-8"))
        fold_rows.append({
            "task": task, "evaluation": evaluation, "balance": balance,
            "protocol": manifest["protocol"], "seed": manifest["seed"], "fold": fold_index(manifest),
            **{f"window_{key}": value for key, value in window_metrics.items() if not isinstance(value, list)},
            **{f"subject_{key}": value for key, value in subject_metrics.items() if not isinstance(value, list)},
            **runtime,
        })
    for (protocol, seed), frame in windows.groupby(["protocol", "seed"], sort=True):
        aggregate_root = root / "aggregate" / protocol / f"seed_{int(seed)}"
        aggregate_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(aggregate_root / "oof_window_predictions.parquet", index=False)
        subjects = aggregate_subjects(frame, positive_label=positive)
        subjects.to_csv(aggregate_root / "oof_subject_predictions.csv", index=False)
        window_metrics = classification_metrics(
            frame.y_true, frame.prediction, frame.p_positive,
            positive_label=positive, negative_label=negative,
        )
        subject_metrics = classification_metrics(
            subjects.y_true, subjects.prediction, subjects.p_positive,
            positive_label=positive, negative_label=negative,
        )
        intervals = subject_bootstrap_intervals(
            subjects, resamples=resamples, seed=int(seed),
            positive_label=positive, negative_label=negative,
        )
        write_json(aggregate_root / "window_metrics.json", window_metrics)
        write_json(aggregate_root / "subject_metrics.json", subject_metrics)
        write_json(aggregate_root / "subject_bootstrap_95.json", intervals)
        summary_rows.append({
            "task": task, "evaluation": evaluation, "balance": balance,
            "protocol": protocol, "seed": int(seed),
            **{f"window_{key}": value for key, value in window_metrics.items() if not isinstance(value, list)},
            **{f"subject_{key}": value for key, value in subject_metrics.items() if not isinstance(value, list)},
            "n_test_subjects": int(subjects.subject_id.nunique()),
        })
        if evaluation == "published_holdout":
            article_rows.append({
                "task": task, "source": "reassessment", "protocol": protocol,
                "seed": int(seed), "unit": "window", "accuracy": window_metrics["accuracy"],
                "directly_comparable_to_article": protocol == "forensic",
                "note": "Same fixed published holdout; forensic selection reproduces test-inspected checkpointing.",
            })
    pd.DataFrame(summary_rows).to_csv(root / "metrics_by_seed.csv", index=False)
    pd.DataFrame(fold_rows).sort_values(["protocol", "seed", "fold"]).to_csv(root / "metrics_by_fold.csv", index=False)
    if article_rows:
        pd.DataFrame(article_rows).to_csv(root / "article_comparison_runs.csv", index=False)


def main() -> None:
    args = _arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    verify_hashes(args.features)
    device = select_device(require_gpu=not args.allow_cpu_smoke)
    print(f"device={device}; gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else None}; parameters={validate_architecture()}")
    features, metadata = load_feature_bundle(args.features)
    features, metadata, subjects = _task_data(features, metadata, args.task, config["task_counts"][args.task])
    negative, positive = task_labels(args.task)
    folds = _outer_folds(args.task, args.evaluation, subjects, config)
    fold_stop = len(folds) if args.fold_stop is None else args.fold_stop
    if not (0 <= args.fold_start < fold_stop <= len(folds)):
        raise ValueError(f"Invalid fold range [{args.fold_start}, {fold_stop}) for {len(folds)} folds")
    selected_folds = folds[args.fold_start:fold_stop]
    if args.benchmark_only:
        selected_folds = selected_folds[: args.benchmark_folds]
    seeds = args.seed or ([2026] if args.evaluation == "loso" else config["training"]["seeds"])
    protocols = _protocols(args.evaluation, args.protocol)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output / "effective_week2_config.yaml")
    shutil.copy2(config["feature_config"], output / "effective_feature_config.yaml")
    shutil.copy2(config["model_config"], output / "effective_model_config.yaml")
    write_json(output / "environment.json", environment_record())
    commit = git_commit(".")
    (output / "github_commit.txt").write_text(commit + "\n", encoding="utf-8")
    tree_hash = source_tree_sha256(".")
    declared_tree_hash = Path("SOURCE_TREE_SHA256.txt")
    if declared_tree_hash.is_file():
        expected_tree_hash = declared_tree_hash.read_text(encoding="utf-8").strip()
        if tree_hash != expected_tree_hash:
            raise ValueError(f"Source tree hash mismatch: {tree_hash} != {expected_tree_hash}")
    (output / "source_tree_sha256.txt").write_text(tree_hash + "\n", encoding="utf-8")
    manifest_root = output / "manifests"
    manifest_root.mkdir(exist_ok=True)
    write_json(manifest_root / f"{args.task}_{args.evaluation}_folds.json", {
        "task": args.task, "evaluation": args.evaluation, "folds": folds,
    })
    feature_manifest_hash = sha256_file(args.features / "SHA256SUMS.txt")

    if args.benchmark_only:
        durations = []
        benchmark_epochs = int(config["kaggle"]["benchmark_epochs"])
        for fold in selected_folds:
            fit, validation, test = _inner_partition(fold, subjects, "valid", config)
            result = train_subject_split(
                features, metadata, fit, validation, test, 2026,
                epochs=benchmark_epochs,
                batch_size=int(config["training"]["batch_size"]),
                learning_rate=float(config["training"]["learning_rate"]),
                require_gpu=not args.allow_cpu_smoke,
                evaluate_test=False,
                balance=args.balance,
                negative_label=negative,
                positive_label=positive,
            )
            durations.append(result.elapsed_seconds)
        seconds_per_fold = float(np.mean(durations) / benchmark_epochs * int(config["training"]["max_epochs"]))
        estimate = seconds_per_fold * len(folds)
        benchmark = {
            "task": args.task, "evaluation": args.evaluation, "balance": args.balance,
            "benchmark_folds": len(durations), "benchmark_epochs": benchmark_epochs,
            "fold_seconds": durations, "estimated_seconds_full_seed": estimate,
            "estimated_hours_full_seed": estimate / 3600,
            "fits_9h_with_margin": estimate <= float(config["kaggle"]["safe_budget_hours"]) * 3600,
        }
        write_json(output / f"benchmark_{args.task}_{args.evaluation}_{args.balance}.json", benchmark)
        write_hashes(output)
        print(json.dumps(benchmark, indent=2))
        return

    run_root = output / args.task / args.evaluation / args.balance
    completed = 0
    for protocol in protocols:
        for seed in seeds:
            for fold in selected_folds:
                fit, validation, test = _inner_partition(fold, subjects, protocol, config)
                signature_payload = {
                    "task": args.task, "evaluation": args.evaluation, "balance": args.balance,
                    "protocol": protocol, "seed": int(seed), "fold": fold,
                    "fit_subjects": sorted(fit), "validation_subjects": sorted(validation),
                    "test_subjects": sorted(test), "feature_manifest_sha256": feature_manifest_hash,
                    "config": config, "source_tree_sha256": tree_hash,
                }
                signature = _signature(signature_payload)
                fold_root = run_root / protocol / f"seed_{int(seed)}" / f"fold_{fold['fold']:03d}"
                if _existing_complete(fold_root, signature, args.resume):
                    print(f"resume: fold already complete: {fold_root}")
                    continue
                result = train_subject_split(
                    features, metadata, fit, validation, test, int(seed),
                    epochs=int(config["training"]["max_epochs"]),
                    batch_size=int(config["training"]["batch_size"]),
                    learning_rate=float(config["training"]["learning_rate"]),
                    require_gpu=not args.allow_cpu_smoke,
                    selection_metric="accuracy" if protocol == "forensic" else "balanced_accuracy",
                    balance=args.balance,
                    negative_label=negative,
                    positive_label=positive,
                )
                manifest = {
                    **signature_payload,
                    "signature": signature,
                    "outer_train_count": len(fold["train_subjects"]),
                    "fit_count": len(fit), "validation_count": len(validation), "test_count": len(test),
                    "forensic_test_selected": protocol == "forensic",
                }
                _save_result(result, fold_root, manifest, commit)
                completed += 1
                print(f"completed {fold_root}")
    _aggregate_completed(
        run_root, args.task, args.evaluation, args.balance, negative, positive,
        int(config["evaluation"]["bootstrap_resamples"]),
    )
    article_rows = [
        {"task": "ad_cn", "source": "article", "protocol": "published_preprocessed", "seed": None, "unit": "window", "accuracy": config["published_preprocessed_window_accuracy"]["ad_cn"], "directly_comparable_to_article": True, "note": "Li et al. (2026), preprocessed AD-CN."},
        {"task": "cn_ftd", "source": "article", "protocol": "published_preprocessed", "seed": None, "unit": "window", "accuracy": config["published_preprocessed_window_accuracy"]["cn_ftd"], "directly_comparable_to_article": True, "note": "Li et al. (2026), preprocessed CN-FTD."},
    ]
    run_comparison = run_root / "article_comparison_runs.csv"
    if run_comparison.is_file():
        article_rows.extend(pd.read_csv(run_comparison).to_dict(orient="records"))
    pd.DataFrame(article_rows).to_csv(output / "article_comparison.csv", index=False)
    write_hashes(output)
    print(json.dumps({"completed_new_folds": completed, "output": str(output), "fold_range": [args.fold_start, fold_stop]}, indent=2))


if __name__ == "__main__":
    main()
