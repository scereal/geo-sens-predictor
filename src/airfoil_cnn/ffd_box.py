"""Create a simple Plot3D FFD box around airfoil points."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def create_ffd_box(points_3d: np.ndarray, dims: tuple[int, int, int], margins: tuple[float, float, float]) -> np.ndarray:
    """Create structured FFD control lattice [ni,nj,nk,3]."""
    if points_3d.shape[1] != 3:
        raise ValueError("points_3d must be [n_points,3]")
    ni, nj, nk = dims
    mx, my, mz = margins
    mins = points_3d.min(axis=0) - np.array([mx, my, mz])
    maxs = points_3d.max(axis=0) + np.array([mx, my, mz])

    xs = np.linspace(mins[0], maxs[0], ni)
    ys = np.linspace(mins[1], maxs[1], nj)
    zs = np.linspace(mins[2], maxs[2], nk)

    lattice = np.zeros((ni, nj, nk, 3), dtype=float)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                lattice[i, j, k] = [x, y, z]
    return lattice


def write_plot3d_ffd(path: str | Path, lattice: np.ndarray) -> None:
    """Write one-block Plot3D style xyz file for pyGeo DVGeometry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ni, nj, nk, _ = lattice.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("1\n")
        f.write(f"{ni} {nj} {nk}\n")
        for c in range(3):
            for k in range(nk):
                for j in range(nj):
                    vals = [f"{lattice[i, j, k, c]:.12e}" for i in range(ni)]
                    f.write(" ".join(vals) + "\n")


def verify_inside_ffd(points_3d: np.ndarray, lattice: np.ndarray) -> bool:
    mins = lattice.reshape(-1, 3).min(axis=0)
    maxs = lattice.reshape(-1, 3).max(axis=0)
    inside = np.all(points_3d >= mins - 1e-10) and np.all(points_3d <= maxs + 1e-10)
    return bool(inside)
