"""Rasterization of pointwise geometry and sensitivities to grid tensors."""

from __future__ import annotations

import numpy as np


def _grid_xy(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    xv = np.linspace(0.0, 1.0, grid_size)
    yv = np.linspace(-0.5, 0.5, grid_size)
    return np.meshgrid(xv, yv, indexing="xy")


def rasterize_sample(X: np.ndarray, S: np.ndarray, grid_size: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Map sample to CNN-ready arrays.

    Args:
        X: [n_points,2]
        S: [n_points,2,n_dv]
    Returns:
        inp: [1,H,W] occupancy-like smooth mask
        tgt: [2*n_dv,H,W] sensitivity channels
    """
    n_points = X.shape[0]
    n_dv = S.shape[2]
    gx, gy = _grid_xy(grid_size)

    # gaussian splat weights [n_points,H,W]
    dx = gx[None, :, :] - X[:, 0][:, None, None]
    dy = gy[None, :, :] - X[:, 1][:, None, None]
    d2 = dx**2 + dy**2
    w = np.exp(-0.5 * d2 / (sigma**2))

    occ = w.max(axis=0)
    occ = occ[None, :, :]

    denom = w.sum(axis=0) + 1e-8
    tgt = np.zeros((2 * n_dv, grid_size, grid_size), dtype=np.float32)
    for j in range(n_dv):
        sx = (w * S[:, 0, j][:, None, None]).sum(axis=0) / denom
        sy = (w * S[:, 1, j][:, None, None]).sum(axis=0) / denom
        tgt[2 * j] = sx
        tgt[2 * j + 1] = sy

    return occ.astype(np.float32), tgt.astype(np.float32)
