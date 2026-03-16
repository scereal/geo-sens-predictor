# geo-sens-predictor

Minimal, complete research scaffold for **2D airfoil geometric sensitivity learning** with a pyGeo-driven target and a CNN baseline.

## Purpose
This repository implements an end-to-end pipeline for:

1. generating a deformed 2D airfoil dataset via an FFD setup (`pyGeo` preferred),
2. storing samples as \(\{(p^{(k)}, X^{(k)}, S^{(k)})\}_{k=1}^K\),
3. rasterizing to image-like tensors,
4. training a simple CNN to predict sensitivity fields,
5. evaluating prediction error **against the pyGeo geometric sensitivity target**,
6. plotting training/validation error over epochs.

> Note: this is a **2D study in a degenerate 3D setup** (airfoil points embedded at `z=0`) to match pyGeo's 3D-oriented workflow.

## Repository structure

```text
README.md
pyproject.toml
.gitignore
configs/
  default.yaml
  small_debug.yaml
data/
  raw/
  processed/
  figures/
  models/
src/
  airfoil_cnn/
    __init__.py
    config.py
    geometry_baseline.py
    ffd_box.py
    pygeo_wrapper.py
    design_variables.py
    sampling.py
    dataset_builder.py
    rasterize.py
    datasets.py
    model.py
    train.py
    evaluate.py
    plots.py
    utils.py
scripts/
  generate_dataset.py
  train_cnn.py
  evaluate_model.py
tests/
  test_sampling.py
  test_dataset_shapes.py
  test_rasterize.py
  test_model_forward.py
```

## Environment setup

### Python
- Python 3.11+

### Install package
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### Install pyGeo
`pyGeo` installation depends on your platform and MDO stack. Typical route:
- use conda-forge / mdolab instructions,
- ensure `from pygeo import DVGeometry` works.

If pyGeo is unavailable, the debug config supports an **analytic fallback** (`pygeo.required: false`) so the pipeline remains executable for smoke tests.

## Data definitions
Each raw sample `k` stores:
- `p(k)`: design vector, shape `[n_dv]`
- `X(k)`: deformed coordinates, shape `[n_points, 2]`
- `S(k)`: geometric sensitivity tensor, shape `[n_points, 2, n_dv]`

Raw outputs:
- `data/raw/sample_000001.npz`, ...
- `data/raw/dataset_raw.npz`
- `data/raw/design_vectors.npz`
- `data/raw/dataset_metadata.json`

Processed CNN data:
- `data/processed/dataset_processed.npz` with
  - `inputs`: `[N, C_in, H, W]`
  - `targets`: `[N, C_out, H, W]`, where `C_out = 2*n_dv` and channel order is
    `[dX/dp1_x, dX/dp1_y, dX/dp2_x, dX/dp2_y, ...]`

## CLI run order

1) Generate dataset:
```bash
python scripts/generate_dataset.py --config configs/default.yaml
```

2) Train CNN:
```bash
python scripts/train_cnn.py --config configs/default.yaml
```

3) Evaluate + plots:
```bash
python scripts/evaluate_model.py --config configs/default.yaml
```

For quick CPU smoke runs:
```bash
python scripts/generate_dataset.py --config configs/small_debug.yaml
python scripts/train_cnn.py --config configs/small_debug.yaml
python scripts/evaluate_model.py --config configs/small_debug.yaml
```

## Figures produced
Saved under `data/figures/`:
- `train_loss.png`
- `val_loss.png`
- `loss_curves.png`
- `per_channel_error.png`
- `sample_prediction_vs_target.png`
- sampling/deformation sanity figures

These losses are computed directly against the generated sensitivity target field (pyGeo backend if enabled).

## Limitations
- Default pyGeo path currently uses finite-difference Jacobian through the wrapper for portability; replace with native pyGeo Jacobian calls if your installation/API supports it.
- Rasterization uses simple Gaussian splatting; it is transparent but coarse.
- The CNN is intentionally minimal and not tuned for best accuracy.

## Suggested next steps
- Point-cloud models (PointNet-like) or graph neural networks over airfoil contours.
- Adaptive/active sampling in the design space.
- Uncertainty-aware training and refinement loops.
- Replace raster target interpolation with higher-fidelity projection.
