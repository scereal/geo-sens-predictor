#!/usr/bin/env python3
"""CLI: train CNN model."""

from __future__ import annotations

import argparse

from airfoil_cnn.config import load_config
from airfoil_cnn.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = run_training(cfg)
    print(f"Training complete. Final val loss={out['val_losses'][-1]:.6e}")


if __name__ == "__main__":
    main()
