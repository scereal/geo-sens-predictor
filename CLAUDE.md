# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`geo-sens-predictor` is a research scaffold for 2D airfoil geometric sensitivity learning. It generates synthetic datasets via Free-Form Deformation (FFD), trains a CNN to predict aerodynamic sensitivity fields, and evaluates the results. The three main scripts must be run sequentially: generate → train → evaluate.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Commands

```bash
# Tests
pytest -q                          # Run all tests
pytest tests/test_model_forward.py # Single test file
pytest tests/test_sampling.py::test_latin_hypercube_sample  # Single test

# Full pipeline (production config)
python scripts/generate_dataset.py --config configs/default.yaml
python scripts/train_cnn.py        --config configs/default.yaml
python scripts/evaluate_model.py   --config configs/default.yaml

# Quick debug pipeline (Makefile wrappers)
make gen    # 16 samples, 128 points, CPU, no pyGeo required
make train
make eval
make test
```

## Architecture

### Pipeline Flow

```
generate_dataset.py → train_cnn.py → evaluate_model.py
       ↓                    ↓                ↓
 data/raw/*.npz    data/models/*.pt    data/figures/*.png
 data/processed/
```

### Configuration (`config.py`)

All behavior is driven by YAML configs loaded into a hierarchy of dataclasses:
`AppConfig → AirfoilConfig, FFDConfig, PyGeoConfig, SamplingConfig, RasterConfig, TrainingConfig, PathsConfig`

- `configs/default.yaml` — production (256 samples, 256 pts, 30 epochs, auto device)
- `configs/small_debug.yaml` — fast iteration (16 samples, 128 pts, 3 epochs, CPU, pyGeo fallback enabled)

### Geometry & Deformation

1. `geometry_baseline.py` — generates NACA airfoil coordinates, embeds in 3D for pyGeo
2. `ffd_box.py` — creates FFD lattice around the airfoil, verifies point containment
3. `design_variables.py` — defines 6 interpretable deformation modes (camber, thickness, LE radius, etc.); `apply_modes(points_2d, p, dv_names) → (X, S)` is the analytic fallback
4. `pygeo_wrapper.py` — wraps pyGeo's `DVGeometry` if available; falls back to analytic modes when `pygeo.required: false`. Returns `DeformationResult` with `X [n_pts,2]`, `S [n_pts,2,n_dv]`

### Dataset Generation (`dataset_builder.py`)

- Latin Hypercube sampling via `sampling.py` → design vectors `[N, 6]`
- Per-sample: deform airfoil → rasterize to tensors → save `.npz`
- `rasterize.py` Gaussian-splats surface points onto a grid: input `[1,H,W]` (occupancy), target `[2*n_dv, H,W]` (sensitivity channels ordered as `[dX/dp1_x, dX/dp1_y, dX/dp2_x, ...]`)
- Saves `data/raw/sample_XXXXXX.npz` per sample, plus `dataset_processed.npz` with stacked `[N,1,64,64]` inputs and `[N,12,64,64]` targets

### Model & Training (`model.py`, `train.py`)

- `SimpleUNet(in_channels, out_channels)` — 2-level encoder-decoder with skip connections, base 16 channels. Intentionally minimal; README notes this is the baseline to improve upon.
- `RasterizedAirfoilDataset` — `torch.utils.data.Dataset` wrapping `dataset_processed.npz`
- `run_training()` — Adam optimizer, MSE loss, saves `cnn_latest.pt` + `cnn_best.pt`, writes `train_log.csv`
- `quick_smoke_train()` — in-memory helper used by integration tests

### pyGeo Dependency

pyGeo is an external aerodynamic geometry library not bundled here. The codebase degrades gracefully:
- If `pygeo.required: true` and pyGeo is missing → hard error
- If `pygeo.required: false` → falls back to analytic `apply_modes()` in `design_variables.py`

The debug config always sets `required: false`, so tests and local development work without pyGeo installed.
