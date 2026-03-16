from __future__ import annotations

import numpy as np

from airfoil_cnn.sampling import latin_hypercube_sample


def test_lhs_shapes_and_bounds() -> None:
    bounds = {"a": (-1.0, 1.0), "b": (2.0, 4.0), "c": (0.0, 0.5)}
    p, names = latin_hypercube_sample(bounds, n_samples=32, seed=10)
    assert p.shape == (32, 3)
    assert names == ["a", "b", "c"]
    lows = np.array([bounds[n][0] for n in names])
    highs = np.array([bounds[n][1] for n in names])
    assert np.all(p >= lows - 1e-12)
    assert np.all(p <= highs + 1e-12)
