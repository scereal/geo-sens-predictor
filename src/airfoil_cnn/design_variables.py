"""Interpretable smooth deformation modes for 2D airfoil points."""

from __future__ import annotations

import numpy as np

DV_NAMES = [
    "max_camber_mode",
    "camber_location_mode",
    "max_thickness_mode",
    "thickness_location_mode",
    "leading_edge_radius_mode",
    "trailing_edge_reflex_mode",
]


def mode_matrix(points_2d: np.ndarray, dv_names: list[str]) -> np.ndarray:
    """Return [n_points, n_dv] mode values affecting y displacement."""
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    sign = np.where(y >= 0.0, 1.0, -1.0)

    modes: list[np.ndarray] = []
    for name in dv_names:
        if name == "max_camber_mode":
            m = np.sin(np.pi * x)
        elif name == "camber_location_mode":
            m = np.sin(2 * np.pi * x) * (1 - x)
        elif name == "max_thickness_mode":
            m = sign * np.sin(np.pi * x)
        elif name == "thickness_location_mode":
            m = sign * np.sin(2 * np.pi * x)
        elif name == "leading_edge_radius_mode":
            m = sign * np.exp(-((x - 0.05) ** 2) / 0.002)
        elif name == "trailing_edge_reflex_mode":
            m = np.exp(-((x - 0.95) ** 2) / 0.004)
        else:
            raise ValueError(f"Unknown DV {name}")
        modes.append(m)
    return np.stack(modes, axis=-1)


def apply_modes(points_2d: np.ndarray, p: np.ndarray, dv_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Apply modes to produce deformed points and analytic sensitivity.

    Returns:
        X: [n_points, 2]
        S: [n_points, 2, n_dv] where S[:,1,:] = dY/dp and S[:,0,:]=0.
    """
    m = mode_matrix(points_2d, dv_names)
    dy = m @ p
    X = points_2d.copy()
    X[:, 1] = points_2d[:, 1] + dy

    n_points, n_dv = m.shape
    S = np.zeros((n_points, 2, n_dv), dtype=float)
    S[:, 1, :] = m
    return X, S
