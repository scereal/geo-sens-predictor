"""PyTorch dataset wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RasterizedAirfoilDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset over preprocessed arrays.

    inputs: [N,C_in,H,W], targets: [N,C_out,H,W]
    """

    def __init__(self, npz_path: str | Path):
        arr = np.load(npz_path)
        self.inputs = arr["inputs"].astype(np.float32)
        self.targets = arr["targets"].astype(np.float32)
        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError("inputs/targets sample count mismatch")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.inputs[idx])
        y = torch.from_numpy(self.targets[idx])
        return x, y
