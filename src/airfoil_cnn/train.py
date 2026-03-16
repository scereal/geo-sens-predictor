"""Training entry points."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from .config import AppConfig
from .datasets import RasterizedAirfoilDataset
from .model import SimpleUNet
from .plots import plot_losses
from .utils import resolve_device, set_seed, write_epoch_log


def run_training(cfg: AppConfig) -> dict[str, list[float]]:
    set_seed(cfg.seed)
    ds = RasterizedAirfoilDataset(Path(cfg.paths.processed_dir) / "dataset_processed.npz")
    n_total = len(ds)
    n_val = max(1, int(round(cfg.training.val_fraction * n_total)))
    n_train = n_total - n_val
    if n_train < 1:
        raise ValueError("Training split is empty")

    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    sample_x, sample_y = ds[0]
    model = SimpleUNet(in_channels=sample_x.shape[0], out_channels=sample_y.shape[0])
    device = resolve_device(cfg.training.device)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    logs: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_acc = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_acc += float(loss.item()) * xb.shape[0]
        train_epoch = train_acc / n_train

        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_acc += float(loss.item()) * xb.shape[0]
        val_epoch = val_acc / n_val

        train_losses.append(train_epoch)
        val_losses.append(val_epoch)
        logs.append({"epoch": float(epoch), "train_loss": train_epoch, "val_loss": val_epoch})

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        Path(cfg.training.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, cfg.training.checkpoint_path)
        if val_epoch < best_val:
            best_val = val_epoch
            torch.save(ckpt, Path(cfg.paths.models_dir) / "cnn_best.pt")

    write_epoch_log(cfg.training.log_csv_path, logs)
    plot_losses(train_losses, val_losses, cfg.paths.figures_dir)

    Path(cfg.paths.figures_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(cfg.paths.figures_dir) / "loss_history.npz",
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train MSE", linewidth=2)
    plt.plot(epochs, val_losses, label="Val MSE", linewidth=2, color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE vs pyGeo sensitivity")
    plt.title("CNN Training: Geometric Sensitivity Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(cfg.paths.figures_dir) / "loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {"train_losses": train_losses, "val_losses": val_losses}


def quick_smoke_train(inputs: np.ndarray, targets: np.ndarray, epochs: int = 1) -> float:
    """Tiny in-memory train helper used by tests."""
    device = torch.device("cpu")
    x = torch.from_numpy(inputs)
    y = torch.from_numpy(targets)
    ds = list(zip(x, y, strict=True))
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    model = SimpleUNet(inputs.shape[1], targets.shape[1], base=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loss_val = 0.0
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
    return loss_val
