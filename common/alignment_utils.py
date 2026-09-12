"""Transparent helpers used only by the named alignment experiments."""
from __future__ import annotations

import hashlib
import math


def epoch_permutation(length: int, seed: int, epoch: int) -> list[int]:
    """Framework-neutral order; hash sorting avoids RNG algorithm differences."""
    return sorted(range(length), key=lambda i: hashlib.sha256(f"{seed}:{epoch}:{i}".encode()).digest())


def augmentation_decision(path: str, seed: int, epoch: int = 0) -> tuple[bool, float]:
    digest = hashlib.sha256(f"{seed}:{epoch}:{path}".encode()).digest()
    flip = digest[0] < 128
    unit = int.from_bytes(digest[1:9], "big") / (2**64 - 1)
    return flip, -5.0 + 10.0 * unit


def common_numpy_weights(seed: int, shapes: list[tuple[int, ...]]) -> list["object"]:
    import numpy as np
    rng = np.random.default_rng(seed)
    result = []
    for shape in shapes:
        fan_in = math.prod(shape[:-1]) if len(shape) > 1 else shape[0]
        limit = math.sqrt(6.0 / fan_in)
        result.append(rng.uniform(-limit, limit, shape).astype("float32"))
    return result
