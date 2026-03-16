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

def plot_sample_predictions(pred_arr: np.ndarray, true_arr: np.ndarray, fig_dir: str | Path, n_samples: int = 3) -> None:
    """Save side-by-side target/pred/error images for a few samples.

    Args:
        pred_arr: [N, C, H, W]
        true_arr: [N, C, H, W]
        fig_dir: output directory
        n_samples: number of samples to save
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_plot = min(n_samples, pred_arr.shape[0])

    for i in range(n_plot):
        target_img = true_arr[i].mean(axis=0)
        pred_img = pred_arr[i].mean(axis=0)
        err_img = pred_img - target_img
        vmax = float(np.max(np.abs(err_img))) + 1e-12

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(target_img, origin="lower", cmap="viridis")
        axes[0].set_title("Target")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(pred_img, origin="lower", cmap="viridis")
        axes[1].set_title("Pred")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(err_img, origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax)
        axes[2].set_title("Error")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        fig.suptitle(f"Prediction diagnostics for sample {i}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"prediction_sample_{i}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
