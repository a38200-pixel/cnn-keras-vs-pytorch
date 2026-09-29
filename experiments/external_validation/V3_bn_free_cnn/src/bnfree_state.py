"""Phase 2 shared-W0 bridge for the V3 BN-free custom CNN."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from common.controlled_initialization import load_initial_weights


@dataclass(frozen=True)
class ParameterSpec:
    semantic_name: str
    shape: tuple[int, ...]
    keras_name: str
    pytorch_name: str


SPECS = [
    ParameterSpec("conv1/kernel", (3, 3, 3, 32), "conv1/kernel", "conv1.weight"),
    ParameterSpec("conv2/kernel", (3, 3, 32, 64), "conv2/kernel", "conv2.weight"),
    ParameterSpec("conv3/kernel", (3, 3, 64, 128), "conv3/kernel", "conv3.weight"),
    ParameterSpec("conv4/kernel", (3, 3, 128, 256), "conv4/kernel", "conv4.weight"),
    ParameterSpec("fc128/kernel", (256, 128), "fc128/kernel", "fc128.weight"),
    ParameterSpec("fc128/bias", (128,), "fc128/bias", "fc128.bias"),
    ParameterSpec("logits/kernel", (128, 8), "logits/kernel", "logits.weight"),
    ParameterSpec("logits/bias", (8,), "logits/bias", "logits.bias"),
]
SPEC_BY_NAME = {spec.semantic_name: spec for spec in SPECS}
TRAINABLE_NAMES = [spec.semantic_name for spec in SPECS]
TRAINABLE_PARAMETER_COUNT = int(sum(np.prod(spec.shape) for spec in SPECS))


def phase2_shared_state(seed: int) -> dict[str, np.ndarray]:
    phase2 = load_initial_weights(seed)
    missing = set(TRAINABLE_NAMES) - set(phase2)
    if missing:
        raise RuntimeError(f"Phase 2 W0 lacks V3 parameters: {sorted(missing)}")
    result = {
        name: np.asarray(phase2[name], dtype=np.float32).copy()
        for name in TRAINABLE_NAMES
    }
    for spec in SPECS:
        if result[spec.semantic_name].shape != spec.shape:
            raise RuntimeError(f"V3 canonical shape mismatch: {spec.semantic_name}")
    return result


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def trainable_values(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(state[name], dtype=np.float32).copy() for name in TRAINABLE_NAMES}


def load_keras_state(model, state: dict[str, np.ndarray]) -> None:
    bindings = model.state_bindings()
    if set(bindings) != set(TRAINABLE_NAMES):
        raise RuntimeError("Keras V3 semantic parameter mapping mismatch")
    for name, variable in bindings.items():
        variable.assign(state[name])


def export_keras_state(model) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(variable.numpy(), dtype=np.float32).copy()
        for name, variable in model.state_bindings().items()
    }


def load_torch_state(model, state: dict[str, np.ndarray]) -> None:
    import torch

    bindings = model.state_bindings()
    if set(bindings) != set(TRAINABLE_NAMES):
        raise RuntimeError("PyTorch V3 semantic parameter mapping mismatch")
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
    by_ref = {id(variable): name for name, variable in model.state_bindings().items()}
    result = {}
    for variable, gradient in zip(model.trainable_variables, gradients):
        name = by_ref.get(id(variable))
        if name is None or gradient is None:
            raise RuntimeError(f"Keras V3 gradient mapping failed: {variable.path}")
        result[name] = np.asarray(gradient.numpy(), dtype=np.float32).copy()
    if set(result) != set(TRAINABLE_NAMES):
        raise RuntimeError("Keras V3 gradient coverage mismatch")
    return result


def torch_gradients(model) -> dict[str, np.ndarray]:
    result = {}
    for name, tensor in model.state_bindings().items():
        if tensor.grad is None:
            raise RuntimeError(f"Missing PyTorch V3 gradient: {name}")
        value = tensor.grad.detach().cpu().numpy()
        if name.endswith("/kernel"):
            value = value.transpose(2, 3, 1, 0) if value.ndim == 4 else value.T
        result[name] = np.asarray(value, dtype=np.float32).copy()
    return result
