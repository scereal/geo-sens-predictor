"""Configuration objects and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AirfoilConfig:
    name: str
    n_points: int


@dataclass(slots=True)
class FFDConfig:
    margin_x: float
    margin_y: float
    margin_z: float
    dims: tuple[int, int, int]
    ffd_path: str


@dataclass(slots=True)
class PyGeoConfig:
    required: bool = True


@dataclass(slots=True)
class SamplingConfig:
    n_samples: int
    random_seed: int
    dv_bounds: dict[str, tuple[float, float]]


@dataclass(slots=True)
class RasterConfig:
    grid_size: int
    sigma: float


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    lr: float
    val_fraction: float
    num_workers: int
    checkpoint_path: str
    log_csv_path: str
    device: str


@dataclass(slots=True)
class PathsConfig:
    raw_dir: str
    processed_dir: str
    figures_dir: str
    models_dir: str


@dataclass(slots=True)
class AppConfig:
    seed: int
    airfoil: AirfoilConfig
    ffd: FFDConfig
    pygeo: PyGeoConfig
    sampling: SamplingConfig
    raster: RasterConfig
    training: TrainingConfig
    paths: PathsConfig


def _tuple_bounds(d: dict[str, list[float] | tuple[float, float]]) -> dict[str, tuple[float, float]]:
    return {k: (float(v[0]), float(v[1])) for k, v in d.items()}


def load_config(path: str | Path) -> AppConfig:
    """Load application config from YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return AppConfig(
        seed=int(raw["seed"]),
        airfoil=AirfoilConfig(**raw["airfoil"]),
        ffd=FFDConfig(
            margin_x=float(raw["ffd"]["margin_x"]),
            margin_y=float(raw["ffd"]["margin_y"]),
            margin_z=float(raw["ffd"]["margin_z"]),
            dims=tuple(raw["ffd"]["dims"]),
            ffd_path=str(raw["ffd"]["ffd_path"]),
        ),
        pygeo=PyGeoConfig(**raw["pygeo"]),
        sampling=SamplingConfig(
            n_samples=int(raw["sampling"]["n_samples"]),
            random_seed=int(raw["sampling"]["random_seed"]),
            dv_bounds=_tuple_bounds(raw["sampling"]["dv_bounds"]),
        ),
        raster=RasterConfig(**raw["raster"]),
        training=TrainingConfig(**raw["training"]),
        paths=PathsConfig(**raw["paths"]),
    )
