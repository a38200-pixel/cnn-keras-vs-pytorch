"""Canonical state helpers for checkpoint-local re-synchronization probes."""

from __future__ import annotations

import numpy as np

from common.common_adam_control import TRAINABLE_NAMES, trainable_values
from common.reference_adam import CommonAdam
from common.trace_utils import compare, state_hash

BN_GAMMA_NAMES = [f"bn{index}/gamma" for index in range(1, 5)]
BN_BETA_NAMES = [f"bn{index}/beta" for index in range(1, 5)]
BN_MEAN_NAMES = [f"bn{index}/mean" for index in range(1, 5)]
BN_VARIANCE_NAMES = [f"bn{index}/variance" for index in range(1, 5)]


def clone_state(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in values.items()}


def concatenate(values: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    missing = set(names) - set(values)
    if missing:
        raise RuntimeError(f"Canonical state is missing keys: {sorted(missing)}")
    return np.concatenate([np.asarray(values[name]).ravel() for name in names])


def compare_group(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], names: list[str],
) -> dict[str, object]:
    return compare(concatenate(left, names), concatenate(right, names))


def states_exact(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], names: list[str] | None = None,
) -> bool:
    selected = sorted(left) if names is None else list(names)
    return (
        set(left) == set(right) if names is None else all(name in left and name in right for name in selected)
    ) and all(np.array_equal(left[name], right[name]) for name in selected)


def adam_from_checkpoint(
    model_state: dict[str, np.ndarray], optimizer_state: dict[str, np.ndarray], step: int,
) -> CommonAdam:
    """Restore CommonAdam from the canonical NPZ state without modifying its source."""
    adam = CommonAdam(trainable_values(model_state))
    if step == 0:
        if optimizer_state:
            raise RuntimeError("Initial optimizer state must be empty")
        return adam
    expected = {
        f"{name}/{moment}" for name in TRAINABLE_NAMES for moment in ("m", "v")
    }
    if set(optimizer_state) != expected:
        raise RuntimeError("Canonical CommonAdam checkpoint mapping mismatch")
    for name in TRAINABLE_NAMES:
        adam.m[name] = np.asarray(optimizer_state[f"{name}/m"], dtype=np.float32).copy()
        adam.v[name] = np.asarray(optimizer_state[f"{name}/v"], dtype=np.float32).copy()
        if adam.m[name].shape != model_state[name].shape or adam.v[name].shape != model_state[name].shape:
            raise RuntimeError(f"CommonAdam checkpoint shape mismatch: {name}")
    adam.step = int(step)
    return adam


def adam_m_state(adam: CommonAdam) -> dict[str, np.ndarray]:
    return clone_state(adam.m)


def adam_v_state(adam: CommonAdam) -> dict[str, np.ndarray]:
    return clone_state(adam.v)


def state_fingerprint(values: dict[str, np.ndarray]) -> str:
    return state_hash(clone_state(values))
