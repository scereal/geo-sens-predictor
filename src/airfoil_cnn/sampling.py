"""Sampling utilities for design vectors."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc


def latin_hypercube_sample(bounds: dict[str, tuple[float, float]], n_samples: int, seed: int) -> tuple[np.ndarray, list[str]]:
    names = list(bounds.keys())
    n_dv = len(names)
    sampler = qmc.LatinHypercube(d=n_dv, seed=seed)
    u = sampler.random(n=n_samples)
    l = np.array([bounds[n][0] for n in names], dtype=float)
    h = np.array([bounds[n][1] for n in names], dtype=float)
    p = qmc.scale(u, l, h)
    return p, names


def save_dv_samples(path: str | Path, p: np.ndarray, names: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=p, names=np.array(names, dtype="U64"))


def plot_sampling_coverage(path: str | Path, p: np.ndarray, names: list[str]) -> None:
    n_dv = p.shape[1]
    fig, axes = plt.subplots(1, n_dv, figsize=(3 * n_dv, 3))
    if n_dv == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.hist(p[:, i], bins=12, alpha=0.8)
        ax.set_title(names[i])
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
