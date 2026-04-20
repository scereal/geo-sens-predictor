#!/usr/bin/env python3
"""CLI: generate raw and processed dataset."""

from __future__ import annotations

import argparse

from airfoil_cnn.config import load_config
from airfoil_cnn.dataset_builder import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    generate_dataset(cfg)
    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
