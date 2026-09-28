"""Deterministic GPU-aware training for forensic and valid protocols."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .aggregation import aggregate_subjects
from .metrics import classification_metrics
from .models import FPSDCNN, validate_architecture
from .reproducibility import seed_everything, write_json
from .splits import stratified_validation_subjects


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, metadata: pd.DataFrame, indices: np.ndarray) -> None:
        self.features = torch.as_tensor(features[indices], dtype=torch.float32)
        self.labels = torch.as_tensor(metadata.iloc[indices]["y_true"].to_numpy(), dtype=torch.long)
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], int(self.indices[index])


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        generator=generator, num_workers=0, pin_memory=torch.cuda.is_available(),
    )


def select_device(require_gpu: bool = True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_gpu and device.type != "cuda":
        raise RuntimeError("A Kaggle GPU is required for full training; enable an accelerator and rerun")
    return device


def _indices(metadata: pd.DataFrame, subjects: set[str]) -> np.ndarray:
    return np.flatnonzero(metadata["subject_id"].isin(subjects).to_numpy())


def _evaluate(model: nn.Module, loader: DataLoader, metadata: pd.DataFrame, device: torch.device) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y, original_index in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += float(criterion(logits, y).item())
            probability = torch.softmax(logits, dim=1)[:, 1]
            for offset, row_index in enumerate(original_index.numpy()):
                source = metadata.iloc[int(row_index)]
                rows.append({
                    "subject_id": source.subject_id,
                    "window_id": source.window_id,
                    "y_true": int(y[offset].item()),
                    "logit_CN": float(logits[offset, 0].item()),
                    "logit_AD": float(logits[offset, 1].item()),
                    "p_AD": float(probability[offset].item()),
                    "prediction": int(probability[offset].item() >= 0.5),
                })
    frame = pd.DataFrame(rows)
    metrics = classification_metrics(frame.y_true, frame.prediction, frame.p_AD)
    metrics["loss"] = total_loss / len(frame)
    return metrics, frame


@dataclass
class TrainResult:
    elapsed_seconds: float
    best_epoch: int
    history: pd.DataFrame
    predictions: pd.DataFrame
    window_metrics: dict
    subject_metrics: dict
    state_dict: dict


def train_one(
    features: np.ndarray,
    metadata: pd.DataFrame,
    split: dict,
    protocol: str,
    seed: int,
    epochs: int,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    require_gpu: bool = True,
    evaluate_test: bool = True,
) -> TrainResult:
    if protocol not in {"forensic_test_selected", "valid_subject_disjoint"}:
        raise ValueError(protocol)
    seed_everything(seed)
    device = select_device(require_gpu=require_gpu)
    train_subjects = set(split["train"]["AD"] + split["train"]["CN"])
    test_subjects = set(split["test"]["AD"] + split["test"]["CN"])
    if protocol == "valid_subject_disjoint":
        fit, validation = stratified_validation_subjects(split, fraction=0.2, seed=2026)
        fit_subjects, validation_subjects = set(fit), set(validation)
    else:
        fit_subjects, validation_subjects = train_subjects, test_subjects
    if fit_subjects & validation_subjects or train_subjects & test_subjects:
        raise RuntimeError("Subject leakage detected")

    fit_data = WindowDataset(features, metadata, _indices(metadata, fit_subjects))
    validation_data = WindowDataset(features, metadata, _indices(metadata, validation_subjects))
    test_data = WindowDataset(features, metadata, _indices(metadata, test_subjects))
    fit_loader = _loader(fit_data, batch_size, True, seed)
    validation_loader = _loader(validation_data, batch_size, False, seed)
    test_loader = _loader(test_data, batch_size, False, seed)

    model = FPSDCNN().to(device)
    validate_architecture(model)
    if next(model.parameters()).device.type != device.type:
        raise RuntimeError("Model was not moved to the selected device")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    best_score, best_epoch, best_state = -np.inf, 0, None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        epoch_started = time.perf_counter()
        for x, y, _ in fit_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if x.device != device or y.device != device:
                raise RuntimeError("A batch was not moved to the selected device")
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            seen += len(y)
        validation_metrics, _ = _evaluate(model, validation_loader, metadata, device)
        selection_metric = "accuracy" if protocol == "forensic_test_selected" else "balanced_accuracy"
        score = float(validation_metrics[selection_metric])
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "selection_score": score,
            "selection_metric": selection_metric,
            "selection_loss": float(validation_metrics["loss"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    evaluation_loader = test_loader if evaluate_test else validation_loader
    window_metrics, predictions = _evaluate(model, evaluation_loader, metadata, device)
    subjects = aggregate_subjects(predictions)
    subject_metrics = classification_metrics(subjects.y_true, subjects.prediction, subjects.p_AD)
    predictions["protocol"] = protocol
    predictions["seed"] = seed
    return TrainResult(
        elapsed_seconds=time.perf_counter() - started,
        best_epoch=best_epoch,
        history=pd.DataFrame(history),
        predictions=predictions,
        window_metrics=window_metrics,
        subject_metrics=subject_metrics,
        state_dict=best_state,
    )


def save_train_result(result: TrainResult, output_dir: str | Path, protocol: str, seed: int, commit: str) -> None:
    root = Path(output_dir) / protocol / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(root / "history.csv", index=False)
    predictions = result.predictions.copy()
    predictions["github_commit"] = commit
    predictions["split"] = "test"
    predictions["configuration"] = protocol
    predictions.to_parquet(root / "window_predictions.parquet", index=False)
    aggregate_subjects(predictions).to_csv(root / "subject_predictions.csv", index=False)
    write_json(root / "window_metrics.json", result.window_metrics)
    write_json(root / "subject_metrics.json", result.subject_metrics)
    write_json(root / "runtime.json", {"elapsed_seconds": result.elapsed_seconds, "best_epoch": result.best_epoch})
    torch.save(result.state_dict, root / "best_model.pth")
    _save_figures(result, Path(output_dir) / "figures", protocol, seed)


def _save_figures(result: TrainResult, figure_dir: Path, protocol: str, seed: int) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{protocol}_seed_{seed}"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(result.history["epoch"], result.history["train_loss"], label="train")
    axes[0].plot(result.history["epoch"], result.history["selection_loss"], label="selection")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(result.history["epoch"], result.history["train_accuracy"], label="train accuracy")
    axes[1].plot(result.history["epoch"], result.history["selection_score"], label="selection score")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Training and selection")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figure_dir / f"{prefix}_curves.png", dpi=160)
    plt.close(figure)
    matrix = np.asarray(result.subject_metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(4, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["CN", "AD"], yticklabels=["CN", "AD"], xlabel="Predicción", ylabel="Real", title=f"{protocol} · seed {seed}")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(figure_dir / f"{prefix}_subject_confusion.png", dpi=160)
    plt.close(figure)
