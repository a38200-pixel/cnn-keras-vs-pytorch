"""Shared deterministic image representation for Experiment 01."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_aligned_rgb(path: str | Path, image_size: int) -> np.ndarray:
    """Return one RGB image as HWC float32 in [0, 1] using Pillow bilinear resize."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    return array / np.float32(255.0)


def compare_framework_inputs(paths: list[str], image_size: int) -> dict[str, object]:
    """Compare NHWC NumPy/Keras input with NCHW PyTorch input after transpose."""
    import torch

    keras_nhwc = np.stack([load_aligned_rgb(path, image_size) for path in paths])
    pytorch_nchw = torch.from_numpy(keras_nhwc.copy()).permute(0, 3, 1, 2).contiguous()
    pytorch_nhwc = pytorch_nchw.permute(0, 2, 3, 1).cpu().numpy()
    difference = keras_nhwc - pytorch_nhwc
    return {
        "samples": len(paths),
        "keras_shape": tuple(keras_nhwc.shape),
        "pytorch_shape": tuple(pytorch_nchw.shape),
        "dtype": str(keras_nhwc.dtype),
        "minimum": float(keras_nhwc.min()),
        "maximum": float(keras_nhwc.max()),
        "mae": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "mse": float(np.mean(np.square(difference))),
        "allclose": bool(np.allclose(keras_nhwc, pytorch_nhwc, rtol=0.0, atol=1e-7)),
    }
