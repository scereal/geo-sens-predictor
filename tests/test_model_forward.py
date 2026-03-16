from __future__ import annotations

import torch

from airfoil_cnn.model import SimpleUNet


def test_model_forward_shape() -> None:
    model = SimpleUNet(in_channels=1, out_channels=12)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 12, 64, 64)
