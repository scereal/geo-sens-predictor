from __future__ import annotations

import numpy as np

from airfoil_cnn.design_variables import DV_NAMES, apply_modes
from airfoil_cnn.geometry_baseline import generate_naca4
from airfoil_cnn.rasterize import rasterize_sample
from airfoil_cnn.train import quick_smoke_train


def test_raw_shapes_and_smoke_train() -> None:
    n_points = 128
    n_dv = len(DV_NAMES)
    base = generate_naca4("NACA0012", n_points=n_points)

    rng = np.random.default_rng(1)
    all_inp, all_tgt = [], []
    for _ in range(4):
        p = rng.normal(scale=0.01, size=(n_dv,))
        X, S = apply_modes(base, p, DV_NAMES)
        assert p.shape == (n_dv,)
        assert X.shape == (n_points, 2)
        assert S.shape == (n_points, 2, n_dv)
        inp, tgt = rasterize_sample(X, S, grid_size=64, sigma=0.03)
        all_inp.append(inp)
        all_tgt.append(tgt)

    inputs = np.stack(all_inp).astype(np.float32)
    targets = np.stack(all_tgt).astype(np.float32)
    final_loss = quick_smoke_train(inputs, targets, epochs=1)
    assert np.isfinite(final_loss)
