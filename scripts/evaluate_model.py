#!/usr/bin/env python3
"""CLI: evaluate model and save figures."""

from __future__ import annotations

import argparse

from airfoil_cnn.config import load_config
from airfoil_cnn.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = run_evaluation(cfg)
    print(f"Evaluation complete. MSE={out['mse']:.6e}")


if __name__ == "__main__":
    main()
