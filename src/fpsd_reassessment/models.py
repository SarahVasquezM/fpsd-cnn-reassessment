"""The checkpoint-compatible FPSD-CNN architecture."""

from __future__ import annotations

import torch
from torch import nn


class StarReLU(nn.Module):
    def __init__(self, scale: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([bias], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.relu(x).square() + self.bias


class FPSDCNN(nn.Module):
    """512→32 reconstruction; one shared StarReLU matches the published source pattern."""

    expected_parameters = 141_732

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 512, kernel_size=3)
        self.conv2 = nn.Conv1d(512, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(32 * 22, 128)
        self.fc2 = nn.Linear(128, 2)
        self.activation = StarReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1:] != (1, 95):
            raise ValueError(f"Expected [batch, 1, 95], got {tuple(x.shape)}")
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        if x.shape[1:] != (32, 22):
            raise RuntimeError(f"Unexpected convolution output {tuple(x.shape)}")
        x = x.flatten(1)
        x = self.activation(self.fc1(x))
        return self.fc2(x)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def validate_architecture(model: nn.Module | None = None) -> int:
    model = FPSDCNN() if model is None else model
    count = parameter_count(model)
    if count != FPSDCNN.expected_parameters:
        raise RuntimeError(f"Parameter count {count:,} != expected {FPSDCNN.expected_parameters:,}")
    reference = next(model.parameters())
    with torch.no_grad():
        output = model(torch.zeros(2, 95, device=reference.device, dtype=reference.dtype))
    if tuple(output.shape) != (2, 2):
        raise RuntimeError(f"Unexpected output shape {tuple(output.shape)}")
    return count
