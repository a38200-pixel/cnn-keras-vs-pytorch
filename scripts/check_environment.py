"""Validate the fixed Apple Silicon environment without training a model."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    failures: list[str] = []

    import keras
    import matplotlib
    import numpy as np
    import pandas
    import PIL
    import sklearn
    import tensorflow as tf
    import torch
    import torchvision

    versions = {
        "Python": sys.version.split()[0],
        "TensorFlow": tf.__version__,
        "Keras": keras.__version__,
        "tf.keras": tf.keras.__version__,
        "tensorflow-metal": importlib.metadata.version("tensorflow-metal"),
        "PyTorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "NumPy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "Pillow": PIL.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
    }
    print("Executable:", sys.executable)
    print("Architecture:", platform.machine())
    for name, version in versions.items():
        print(f"{name}: {version}")

    if platform.machine() != "arm64": failures.append("Architecture is not arm64")
    if tf.__version__ != "2.18.1": failures.append("TensorFlow must be 2.18.1")

    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    print("TensorFlow physical GPUs:", physical)
    print("TensorFlow logical GPUs:", logical)
    if not logical:
        failures.append("TensorFlow GPU was not detected")
    else:
        with tf.device("/GPU:0"):
            tf_result = tf.matmul(tf.random.normal((64, 64)), tf.random.normal((64, 64)))
        tf_finite = bool(tf.reduce_all(tf.math.is_finite(tf_result)).numpy())
        print("TensorFlow matmul:", tf_result.shape, tf_result.device, "finite=", tf_finite)
        if "GPU:0" not in tf_result.device or not tf_finite:
            failures.append("TensorFlow GPU matmul failed validation")

    print("PyTorch MPS built:", torch.backends.mps.is_built())
    print("PyTorch MPS available:", torch.backends.mps.is_available())
    print("PYTORCH_ENABLE_MPS_FALLBACK:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"))
    if not torch.backends.mps.is_available():
        failures.append("PyTorch MPS is not available")
    else:
        mps = torch.device("mps")
        torch_result = torch.randn(64, 64, device=mps) @ torch.randn(64, 64, device=mps)
        torch_finite = bool(torch.isfinite(torch_result).all().item())
        print("PyTorch matmul:", tuple(torch_result.shape), torch_result.device, "finite=", torch_finite)
        if torch_result.device.type != "mps" or not torch_finite:
            failures.append("PyTorch MPS matmul failed validation")

    from common.dataset_utils import CLASSES, build_split_manifest, is_physically_split, split_counts
    counts = split_counts(build_split_manifest())
    print("Dataset physical split:", is_physically_split())
    print("Dataset counts:", counts)
    print("Classes:", CLASSES)
    if not is_physically_split() or counts != {"train": 10251, "val": 2194, "test": 2203}:
        failures.append("Dataset validation failed")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    print("pip check:", (pip_check.stdout or pip_check.stderr).strip())
    if pip_check.returncode != 0: failures.append("pip dependency validation failed")

    if failures:
        print("Environment validation: FAIL")
        for failure in failures: print(" -", failure)
        raise SystemExit(1)
    print("Environment validation: PASS")


if __name__ == "__main__":
    main()
