"""End-to-end dataset generation from baseline geometry to processed tensors."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import AppConfig
from .design_variables import DV_NAMES
from .ffd_box import create_ffd_box, verify_inside_ffd, write_plot3d_ffd
from .geometry_baseline import embed_2d_in_3d, generate_naca4
from .pygeo_wrapper import PyGeoAirfoilWrapper
from .rasterize import rasterize_sample
from .sampling import latin_hypercube_sample, plot_sampling_coverage, save_dv_samples


def _ensure_dirs(cfg: AppConfig) -> None:
    for d in [cfg.paths.raw_dir, cfg.paths.processed_dir, cfg.paths.figures_dir, cfg.paths.models_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _plot_airfoil(path: Path, X: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(X[:, 0], X[:, 1], "-k")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_dataset(cfg: AppConfig) -> None:
    """Generate raw and processed datasets.

    Raw each sample:
      p [n_dv], X [n_points,2], S [n_points,2,n_dv]
    Processed arrays:
      inputs [N,C_in,H,W], targets [N,C_out,H,W]
    """
    _ensure_dirs(cfg)

    base_2d = generate_naca4(cfg.airfoil.name, cfg.airfoil.n_points)
    base_3d = embed_2d_in_3d(base_2d)

    lattice = create_ffd_box(base_3d, cfg.ffd.dims, (cfg.ffd.margin_x, cfg.ffd.margin_y, cfg.ffd.margin_z))
    if not verify_inside_ffd(base_3d, lattice):
        raise RuntimeError("Baseline points are not inside generated FFD box.")
    write_plot3d_ffd(cfg.ffd.ffd_path, lattice)

    p_samples, dv_names = latin_hypercube_sample(cfg.sampling.dv_bounds, cfg.sampling.n_samples, cfg.sampling.random_seed)
    if dv_names != DV_NAMES:
        # Keep fixed semantic order
        idx = [dv_names.index(n) for n in DV_NAMES]
        p_samples = p_samples[:, idx]
        dv_names = DV_NAMES.copy()

    save_dv_samples(Path(cfg.paths.raw_dir) / "design_vectors.npz", p_samples, dv_names)
    plot_sampling_coverage(Path(cfg.paths.figures_dir) / "sampling_histograms.png", p_samples, dv_names)

    wrapper = PyGeoAirfoilWrapper(base_2d, cfg.ffd.ffd_path, dv_names, pygeo_required=cfg.pygeo.required)

    all_p, all_X, all_S = [], [], []
    inputs, targets = [], []

    for k, p in enumerate(p_samples):
        res = wrapper.deform_and_sens(p)
        assert res.X.shape == (cfg.airfoil.n_points, 2)
        assert res.S.shape == (cfg.airfoil.n_points, 2, len(dv_names))

        all_p.append(res.p)
        all_X.append(res.X)
        all_S.append(res.S)

        sample_file = Path(cfg.paths.raw_dir) / f"sample_{k+1:06d}.npz"
        np.savez_compressed(
            sample_file,
            p=res.p,
            X=res.X,
            S=res.S,
            airfoil_name=cfg.airfoil.name,
            n_points=cfg.airfoil.n_points,
            n_dv=len(dv_names),
            sample_index=k,
            backend=res.backend,
        )

        inp, tgt = rasterize_sample(res.X, res.S, cfg.raster.grid_size, cfg.raster.sigma)
        inputs.append(inp)
        targets.append(tgt)

        if k < 3:
            _plot_airfoil(Path(cfg.paths.figures_dir) / f"deformed_sample_{k+1:02d}.png", res.X, f"Sample {k+1}")

    P = np.stack(all_p)
    X = np.stack(all_X)
    S = np.stack(all_S)
    np.savez_compressed(Path(cfg.paths.raw_dir) / "dataset_raw.npz", p=P, X=X, S=S, dv_names=np.array(dv_names, dtype="U64"))

    in_arr = np.stack(inputs).astype(np.float32)
    tgt_arr = np.stack(targets).astype(np.float32)
    np.savez_compressed(Path(cfg.paths.processed_dir) / "dataset_processed.npz", inputs=in_arr, targets=tgt_arr, p=P)

    meta = {
        "airfoil_name": cfg.airfoil.name,
        "n_samples": int(P.shape[0]),
        "n_points": int(cfg.airfoil.n_points),
        "n_dv": len(dv_names),
        "dv_names": dv_names,
        "dv_bounds": {k: list(v) for k, v in cfg.sampling.dv_bounds.items()},
        "grid_size": cfg.raster.grid_size,
        "input_shape": list(in_arr.shape),
        "target_shape": list(tgt_arr.shape),
    }
    with (Path(cfg.paths.raw_dir) / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
