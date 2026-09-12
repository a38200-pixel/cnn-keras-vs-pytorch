"""Framework seed helpers. Deterministic kernels can reduce speed."""
from __future__ import annotations

import os
import random


def seed_python_numpy(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def seed_tensorflow(seed: int) -> None:
    seed_python_numpy(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def seed_pytorch(seed: int) -> None:
    seed_python_numpy(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
