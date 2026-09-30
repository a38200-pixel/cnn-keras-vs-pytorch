"""Canonical V4 state: Phase 2 body W0 plus a deterministic 10-class head."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from common.controlled_initialization import load_initial_weights


@dataclass(frozen=True)
class ParameterSpec:
    semantic_name: str
    shape: tuple[int, ...]
    trainable: bool
    keras_name: str
    pytorch_name: str


def _specs() -> list[ParameterSpec]:
    channels = (3, 32, 64, 128, 256)
    specs = []
    for index in range(1, 5):
        specs.append(ParameterSpec(
            f"conv{index}/kernel", (3, 3, channels[index - 1], channels[index]), True,
            f"conv{index}/kernel", f"conv{index}.weight",
        ))
        for field, trainable, keras_field, torch_field in (
            ("gamma", True, "gamma", "weight"),
            ("beta", True, "beta", "bias"),
            ("mean", False, "moving_mean", "running_mean"),
            ("variance", False, "moving_variance", "running_var"),
        ):
            specs.append(ParameterSpec(
                f"bn{index}/{field}", (channels[index],), trainable,
                f"bn{index}/{keras_field}", f"bn{index}.{torch_field}",
            ))
    specs.extend([
        ParameterSpec("fc128/kernel", (256, 128), True, "fc128/kernel", "fc128.weight"),
        ParameterSpec("fc128/bias", (128,), True, "fc128/bias", "fc128.bias"),
        ParameterSpec("classifier/kernel", (128, 10), True, "classifier/kernel", "classifier.weight"),
        ParameterSpec("classifier/bias", (10,), True, "classifier/bias", "classifier.bias"),
    ])
    return specs


SPECS = _specs()
SPEC_BY_NAME = {spec.semantic_name: spec for spec in SPECS}
TRAINABLE_NAMES = [spec.semantic_name for spec in SPECS if spec.trainable]
BODY_NAMES = [name for name in SPEC_BY_NAME if not name.startswith("classifier/")]
BN_MEAN_NAMES = [name for name in SPEC_BY_NAME if name.endswith("/mean")]
BN_VARIANCE_NAMES = [name for name in SPEC_BY_NAME if name.endswith("/variance")]
TRAINABLE_PARAMETER_COUNT = int(sum(np.prod(SPEC_BY_NAME[name].shape) for name in TRAINABLE_NAMES))
NONTRAINABLE_STATE_COUNT = int(sum(np.prod(SPEC_BY_NAME[name].shape) for name in BN_MEAN_NAMES + BN_VARIANCE_NAMES))


def create_initial_state(seed: int) -> dict[str, np.ndarray]:
    phase2 = load_initial_weights(seed)
    state = {name: np.asarray(phase2[name], dtype=np.float32).copy() for name in BODY_NAMES}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1A17, 0xC1FA10]))
    shape = (128, 10)
    limit = np.sqrt(6.0 / sum(shape))
    state["classifier/kernel"] = rng.uniform(-limit, limit, shape).astype(np.float32)
    state["classifier/bias"] = np.zeros((10,), dtype=np.float32)
    if set(state) != set(SPEC_BY_NAME):
        raise RuntimeError("V4 canonical state coverage mismatch")
    return state


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def trainable_values(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(state[name], dtype=np.float32).copy() for name in TRAINABLE_NAMES}


def load_keras_state(model, state: dict[str, np.ndarray]) -> None:
    bindings = model.state_bindings()
    if set(bindings) != set(SPEC_BY_NAME):
        raise RuntimeError("Keras V4 semantic state mapping mismatch")
    for name, variable in bindings.items():
        variable.assign(state[name])


def export_keras_state(model) -> dict[str, np.ndarray]:
    return {name: np.asarray(value.numpy(), dtype=np.float32).copy() for name, value in model.state_bindings().items()}


def load_torch_state(model, state: dict[str, np.ndarray]) -> None:
    import torch

    bindings = model.state_bindings()
    if set(bindings) != set(SPEC_BY_NAME):
        raise RuntimeError("PyTorch V4 semantic state mapping mismatch")
    with torch.no_grad():
        for name, tensor in bindings.items():
            value = state[name]
            if name.endswith("/kernel"):
                value = value.transpose(3, 2, 0, 1) if value.ndim == 4 else value.T
            tensor.copy_(torch.from_numpy(np.ascontiguousarray(value)).to(tensor.device))


def export_torch_state(model) -> dict[str, np.ndarray]:
    result = {}
    for name, tensor in model.state_bindings().items():
        value = tensor.detach().cpu().numpy()
        if name.endswith("/kernel"):
            value = value.transpose(2, 3, 1, 0) if value.ndim == 4 else value.T
        result[name] = np.asarray(value, dtype=np.float32).copy()
    return result


def keras_gradients(model, gradients) -> dict[str, np.ndarray]:
    by_ref = {id(variable): name for name, variable in model.state_bindings().items() if name in TRAINABLE_NAMES}
    result = {}
    for variable, gradient in zip(model.trainable_variables, gradients):
        name = by_ref.get(id(variable))
        if name is None or gradient is None:
            raise RuntimeError(f"Keras V4 gradient mapping failed: {variable.path}")
        result[name] = np.asarray(gradient.numpy(), dtype=np.float32).copy()
    if set(result) != set(TRAINABLE_NAMES):
        raise RuntimeError("Keras V4 gradient coverage mismatch")
    return result


def torch_gradients(model) -> dict[str, np.ndarray]:
    result = {}
    for name, tensor in model.state_bindings().items():
        if name not in TRAINABLE_NAMES:
            continue
        if tensor.grad is None:
            raise RuntimeError(f"Missing PyTorch V4 gradient: {name}")
        value = tensor.grad.detach().cpu().numpy()
        if name.endswith("/kernel"):
            value = value.transpose(2, 3, 1, 0) if value.ndim == 4 else value.T
        result[name] = np.asarray(value, dtype=np.float32).copy()
    return result
