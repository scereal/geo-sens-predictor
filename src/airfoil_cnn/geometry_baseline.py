"""Baseline analytical airfoil generators."""

from __future__ import annotations

import numpy as np


def _naca4_params(name: str) -> tuple[float, float, float]:
    name = name.upper().replace("NACA", "")
    if len(name) != 4 or not name.isdigit():
        raise ValueError(f"Expected NACA4 name, got {name}")
    m = int(name[0]) / 100.0
    p = int(name[1]) / 10.0
    t = int(name[2:]) / 100.0
    return m, p, t


def generate_naca4(name: str, n_points: int = 256) -> np.ndarray:
    """Return ordered 2D contour points [n_points, 2] from TE upper to TE lower."""
    m, p, t = _naca4_params(name)
    if n_points % 2 != 0:
        raise ValueError("n_points must be even")
    n_half = n_points // 2
    beta = np.linspace(0.0, np.pi, n_half)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    if m == 0.0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / (p**2) * (2 * p * x - x**2),
            m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x**2),
        )
        dyc_dx = np.where(
            x < p,
            2 * m / (p**2) * (p - x),
            2 * m / ((1 - p) ** 2) * (p - x),
        )

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_contour = np.concatenate([xu[::-1], xl[1:]])
    y_contour = np.concatenate([yu[::-1], yl[1:]])
    coords = np.column_stack([x_contour, y_contour])
    # match exact n_points (TE duplicated if needed)
    if coords.shape[0] < n_points:
        coords = np.vstack([coords, coords[-1:]])
    return coords[:n_points]


def embed_2d_in_3d(points_2d: np.ndarray) -> np.ndarray:
    """Embed [n_points, 2] into [n_points, 3] with z=0."""
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must be [n_points,2]")
    z = np.zeros((points_2d.shape[0], 1), dtype=points_2d.dtype)
    return np.hstack([points_2d, z])
