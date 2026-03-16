"""Plot helpers for training and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_losses(train_losses: list[float], val_losses: list[float], fig_dir: str | Path) -> None:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="train")
    ax.set_title("Training loss vs pyGeo target")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "train_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, val_losses, label="val", color="tab:orange")
    ax.set_title("Validation loss vs pyGeo target")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "val_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.set_title("CNN error against pyGeo sensitivity target")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_per_channel_error(channel_mse: np.ndarray, fig_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(channel_mse)), channel_mse)
    ax.set_title("Per-channel MSE (prediction vs pyGeo target)")
    ax.set_xlabel("channel index")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_prediction_vs_target(pred: np.ndarray, target: np.ndarray, fig_path: str | Path, n_show: int = 4) -> None:
    n_show = min(n_show, pred.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    for i in range(n_show):
        im0 = axes[0, i].imshow(target[i], origin="lower", cmap="coolwarm")
        axes[0, i].set_title(f"target ch{i}")
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046)

        im1 = axes[1, i].imshow(pred[i], origin="lower", cmap="coolwarm")
        axes[1, i].set_title(f"pred ch{i}")
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046)
    fig.suptitle("Held-out sample: pyGeo target vs CNN prediction")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
