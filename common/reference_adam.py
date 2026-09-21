"""Framework-independent Adam arithmetic for attribution diagnostics.

The bias-corrected form matches PyTorch's epsilon placement. Keras uses
alpha * m / (sqrt(v) + epsilon), where m/v are uncorrected moments.
"""

from __future__ import annotations

import numpy as np

from common.controlled_config import BETA1, BETA2, EPSILON, LEARNING_RATE


def adam_step(
    gradient: np.ndarray,
    previous_m: np.ndarray | None = None,
    previous_v: np.ndarray | None = None,
    *,
    step: int = 1,
    learning_rate: float = LEARNING_RATE,
    beta1: float = BETA1,
    beta2: float = BETA2,
    epsilon: float = EPSILON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (m, v, delta) using float64 reference arithmetic."""
    if step < 1:
        raise ValueError("Adam step must be >= 1")
    g = np.asarray(gradient, dtype=np.float64)
    m0 = np.zeros_like(g) if previous_m is None else np.asarray(previous_m, dtype=np.float64)
    v0 = np.zeros_like(g) if previous_v is None else np.asarray(previous_v, dtype=np.float64)
    if g.shape != m0.shape or g.shape != v0.shape:
        raise ValueError("Gradient/moment shape mismatch")
    m = beta1 * m0 + (1 - beta1) * g
    v = beta2 * v0 + (1 - beta2) * g * g
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    delta = -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    if not all(np.isfinite(value).all() for value in (m, v, delta)):
        raise RuntimeError("NaN/Inf in reference Adam")
    return m, v, delta
