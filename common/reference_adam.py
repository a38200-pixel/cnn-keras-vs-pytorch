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


class CommonAdam:
    """Stateful float32 Adam shared by Keras and PyTorch controlled runs.

    Gradients and parameters use canonical NumPy layouts. Both framework
    branches call this exact implementation with the same operation order.
    """

    def __init__(
        self, parameters: dict[str, np.ndarray], *, learning_rate: float = LEARNING_RATE,
        beta1: float = BETA1, beta2: float = BETA2, epsilon: float = EPSILON,
    ) -> None:
        self.learning_rate = np.float32(learning_rate)
        self.beta1 = np.float32(beta1)
        self.beta2 = np.float32(beta2)
        self.epsilon = np.float32(epsilon)
        self.step = 0
        self.m = {name: np.zeros_like(value, dtype=np.float32) for name, value in parameters.items()}
        self.v = {name: np.zeros_like(value, dtype=np.float32) for name, value in parameters.items()}

    def update(
        self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if set(parameters) != set(self.m) or set(gradients) != set(self.m):
            raise RuntimeError("Common Adam parameter/gradient mapping mismatch")
        self.step += 1
        one = np.float32(1.0)
        beta1_power = np.power(self.beta1, np.int32(self.step), dtype=np.float32)
        beta2_power = np.power(self.beta2, np.int32(self.step), dtype=np.float32)
        correction1 = np.float32(one - beta1_power)
        correction2 = np.float32(one - beta2_power)
        updated, deltas = {}, {}
        for name in parameters:
            parameter = np.asarray(parameters[name], dtype=np.float32)
            gradient = np.asarray(gradients[name], dtype=np.float32)
            if parameter.shape != gradient.shape or gradient.shape != self.m[name].shape:
                raise RuntimeError(f"Common Adam shape mismatch: {name}")
            self.m[name] = np.add(
                np.multiply(self.beta1, self.m[name], dtype=np.float32),
                np.multiply(one - self.beta1, gradient, dtype=np.float32),
                dtype=np.float32,
            )
            self.v[name] = np.add(
                np.multiply(self.beta2, self.v[name], dtype=np.float32),
                np.multiply(one - self.beta2, np.square(gradient, dtype=np.float32), dtype=np.float32),
                dtype=np.float32,
            )
            m_hat = np.divide(self.m[name], correction1, dtype=np.float32)
            v_hat = np.divide(self.v[name], correction2, dtype=np.float32)
            denominator = np.add(np.sqrt(v_hat, dtype=np.float32), self.epsilon, dtype=np.float32)
            delta = np.negative(
                np.divide(np.multiply(self.learning_rate, m_hat, dtype=np.float32), denominator, dtype=np.float32),
                dtype=np.float32,
            )
            value = np.add(parameter, delta, dtype=np.float32)
            if not np.isfinite(value).all() or not np.isfinite(self.m[name]).all() or not np.isfinite(self.v[name]).all():
                raise RuntimeError(f"NaN/Inf in Common Adam: {name}")
            updated[name], deltas[name] = value, delta
        return updated, deltas

    def export_state(self) -> dict[str, np.ndarray]:
        state = {}
        for name in self.m:
            state[f"{name}/m"] = self.m[name].copy()
            state[f"{name}/v"] = self.v[name].copy()
        return state
