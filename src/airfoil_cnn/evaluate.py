"""Model evaluation helpers and plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import AppConfig
from .datasets import RasterizedAirfoilDataset
from .model import SimpleUNet
from .plots import plot_per_channel_error, plot_prediction_vs_target
from .utils import resolve_device


def run_evaluation(cfg: AppConfig) -> dict[str, float]:
    ds = RasterizedAirfoilDataset(Path(cfg.paths.processed_dir) / "dataset_processed.npz")
    x0, y0 = ds[0]
    model = SimpleUNet(x0.shape[0], y0.shape[0])
    ckpt = torch.load(cfg.training.checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    device = resolve_device(cfg.training.device)
    model.to(device)
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in ds:
            xb = x.unsqueeze(0).to(device)
            pred = model(xb).squeeze(0).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())

    pred_arr = np.stack(all_pred)
    true_arr = np.stack(all_true)
    mse = ((pred_arr - true_arr) ** 2).mean()
    channel_mse = ((pred_arr - true_arr) ** 2).mean(axis=(0, 2, 3))

    Path(cfg.paths.figures_dir).mkdir(parents=True, exist_ok=True)
    plot_per_channel_error(channel_mse, Path(cfg.paths.figures_dir) / "per_channel_error.png")
    plot_prediction_vs_target(pred_arr[0], true_arr[0], Path(cfg.paths.figures_dir) / "sample_prediction_vs_target.png")

    out_path = Path(cfg.paths.models_dir) / "evaluation_metrics.npz"
    np.savez_compressed(out_path, mse=mse, channel_mse=channel_mse)
    return {"mse": float(mse)}
