from __future__ import annotations

import numpy as np

from airfoil_cnn.rasterize import rasterize_sample


def test_rasterized_shapes() -> None:
    n_points = 64
    n_dv = 6
    x = np.linspace(0.0, 1.0, n_points)
    y = 0.08 * np.sin(np.pi * x)
    X = np.column_stack([x, y])
    S = np.zeros((n_points, 2, n_dv), dtype=float)
    S[:, 1, :] = 1.0
    inp, tgt = rasterize_sample(X, S, grid_size=64, sigma=0.03)
    assert inp.shape == (1, 64, 64)
    assert tgt.shape == (12, 64, 64)
