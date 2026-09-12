"""Explicit device selection and reproducibility-oriented environment logging."""
from __future__ import annotations

import platform
import sys


def select_torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensorflow_device_name() -> str:
    import tensorflow as tf

    return "tensorflow-metal GPU:0" if tf.config.list_physical_devices("GPU") else "cpu"


def print_tensorflow_environment(experiment: str, seed: int) -> None:
    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    print({
        "experiment": experiment,
        "framework": "TensorFlow / Keras",
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "architecture": platform.machine(),
        "device": tensorflow_device_name(),
        "physical_gpus": physical,
        "logical_gpus": logical,
        "seed": seed,
    })
    if not physical:
        print("[WARNING] TensorFlow GPU was not detected.")


def print_torch_environment(experiment: str, seed: int):
    import torch

    device = select_torch_device()
    print({
        "experiment": experiment,
        "framework": "PyTorch",
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "architecture": platform.machine(),
        "device": str(device),
        "seed": seed,
    })
    return device
