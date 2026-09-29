"""Canonical ResNet18 specification, initialization, and framework bridges."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from v2_config import NUM_CLASSES


@dataclass(frozen=True)
class ParameterSpec:
    semantic_name: str
    shape: tuple[int, ...]
    rule: str
    trainable: bool
    keras_name: str
    pytorch_name: str


STAGE_CHANNELS = (64, 128, 256, 512)
BLOCKS_PER_STAGE = 2


def _bn_specs(prefix: str, channels: int, keras_prefix: str, torch_prefix: str):
    for field, rule, trainable, keras_suffix, torch_suffix in (
        ("gamma", "ones", True, "gamma", "weight"),
        ("beta", "zeros", True, "beta", "bias"),
        ("mean", "zeros", False, "moving_mean", "running_mean"),
        ("variance", "ones", False, "moving_variance", "running_var"),
    ):
        yield ParameterSpec(
            f"{prefix}.{field}", (channels,), rule, trainable,
            f"{keras_prefix}/{keras_suffix}", f"{torch_prefix}.{torch_suffix}",
        )


def parameter_specs() -> list[ParameterSpec]:
    specs = [ParameterSpec(
        "stem.conv.kernel", (7, 7, 3, 64), "glorot_uniform", True,
        "stem_conv/kernel", "stem_conv.weight",
    )]
    specs.extend(_bn_specs("stem.bn", 64, "stem_bn", "stem_bn"))
    in_channels = 64
    for stage_index, channels in enumerate(STAGE_CHANNELS, 1):
        for block_index in range(1, BLOCKS_PER_STAGE + 1):
            prefix = f"stage{stage_index}.block{block_index}"
            module = f"stage{stage_index}_block{block_index}"
            stride = 2 if stage_index > 1 and block_index == 1 else 1
            specs.append(ParameterSpec(
                f"{prefix}.conv1.kernel", (3, 3, in_channels, channels),
                "glorot_uniform", True, f"{module}/conv1/kernel", f"{module}.conv1.weight",
            ))
            specs.extend(_bn_specs(f"{prefix}.bn1", channels, f"{module}/bn1", f"{module}.bn1"))
            specs.append(ParameterSpec(
                f"{prefix}.conv2.kernel", (3, 3, channels, channels),
                "glorot_uniform", True, f"{module}/conv2/kernel", f"{module}.conv2.weight",
            ))
            specs.extend(_bn_specs(f"{prefix}.bn2", channels, f"{module}/bn2", f"{module}.bn2"))
            if stride != 1 or in_channels != channels:
                specs.append(ParameterSpec(
                    f"{prefix}.shortcut.conv.kernel", (1, 1, in_channels, channels),
                    "glorot_uniform", True,
                    f"{module}/shortcut_conv/kernel", f"{module}.shortcut_conv.weight",
                ))
                specs.extend(_bn_specs(
                    f"{prefix}.shortcut.bn", channels,
                    f"{module}/shortcut_bn", f"{module}.shortcut_bn",
                ))
            in_channels = channels
    specs.extend([
        ParameterSpec(
            "classifier.kernel", (512, NUM_CLASSES), "glorot_uniform", True,
            "classifier/kernel", "classifier.weight",
        ),
        ParameterSpec(
            "classifier.bias", (NUM_CLASSES,), "zeros", True,
            "classifier/bias", "classifier.bias",
        ),
    ])
    return specs


SPECS = parameter_specs()
SPEC_BY_NAME = {spec.semantic_name: spec for spec in SPECS}
TRAINABLE_NAMES = [spec.semantic_name for spec in SPECS if spec.trainable]
BN_GAMMA_NAMES = [name for name in SPEC_BY_NAME if name.endswith(".gamma")]
BN_BETA_NAMES = [name for name in SPEC_BY_NAME if name.endswith(".beta")]
BN_MEAN_NAMES = [name for name in SPEC_BY_NAME if name.endswith(".mean")]
BN_VARIANCE_NAMES = [name for name in SPEC_BY_NAME if name.endswith(".variance")]


def create_initial_state(seed: int) -> dict[str, np.ndarray]:
    """Use the Phase 2 policy/RNG method with V2-specific ResNet18 shapes."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1A17]))
    values: dict[str, np.ndarray] = {}
    for spec in SPECS:
        if spec.rule == "glorot_uniform":
            shape = spec.shape
            if len(shape) == 4:
                fan_in = shape[0] * shape[1] * shape[2]
                fan_out = shape[0] * shape[1] * shape[3]
            else:
                fan_in, fan_out = shape
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            value = rng.uniform(-limit, limit, shape).astype(np.float32)
        elif spec.rule == "ones":
            value = np.ones(spec.shape, dtype=np.float32)
        else:
            value = np.zeros(spec.shape, dtype=np.float32)
        values[spec.semantic_name] = value
    return values


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def trainable_values(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(state[name], dtype=np.float32).copy() for name in TRAINABLE_NAMES}


def count_parameters(names: list[str]) -> int:
    return int(sum(np.prod(SPEC_BY_NAME[name].shape) for name in names))


TRAINABLE_PARAMETER_COUNT = count_parameters(TRAINABLE_NAMES)
NON_TRAINABLE_STATE_COUNT = count_parameters(BN_MEAN_NAMES + BN_VARIANCE_NAMES)


def load_keras_state(model, state: dict[str, np.ndarray]) -> None:
    bindings = model.state_bindings()
    if set(bindings) != set(SPEC_BY_NAME):
        raise RuntimeError("Keras ResNet18 semantic state mapping mismatch")
    for name, variable in bindings.items():
        variable.assign(state[name])


def export_keras_state(model) -> dict[str, np.ndarray]:
    return {name: np.asarray(variable.numpy(), dtype=np.float32).copy()
            for name, variable in model.state_bindings().items()}


def load_torch_state(model, state: dict[str, np.ndarray]) -> None:
    import torch

    bindings = model.state_bindings()
    if set(bindings) != set(SPEC_BY_NAME):
        raise RuntimeError("PyTorch ResNet18 semantic state mapping mismatch")
    with torch.no_grad():
        for name, tensor in bindings.items():
            value = state[name]
            if name.endswith(".kernel"):
                value = value.transpose(3, 2, 0, 1) if value.ndim == 4 else value.T
            tensor.copy_(torch.from_numpy(np.ascontiguousarray(value)).to(tensor.device))


def export_torch_state(model) -> dict[str, np.ndarray]:
    result = {}
    for name, tensor in model.state_bindings().items():
        value = tensor.detach().cpu().numpy()
        if name.endswith(".kernel"):
            value = value.transpose(2, 3, 1, 0) if value.ndim == 4 else value.T
        result[name] = np.asarray(value, dtype=np.float32).copy()
    return result


def load_keras_trainable(model, values: dict[str, np.ndarray]) -> None:
    bindings = model.state_bindings()
    for name in TRAINABLE_NAMES:
        bindings[name].assign(values[name])


def load_torch_trainable(model, values: dict[str, np.ndarray]) -> None:
    import torch

    bindings = model.state_bindings()
    with torch.no_grad():
        for name in TRAINABLE_NAMES:
            value = values[name]
            if name.endswith(".kernel"):
                value = value.transpose(3, 2, 0, 1) if value.ndim == 4 else value.T
            bindings[name].copy_(torch.from_numpy(np.ascontiguousarray(value)).to(bindings[name].device))


def keras_gradients(model, gradients) -> dict[str, np.ndarray]:
    by_ref = {id(variable): name for name, variable in model.state_bindings().items()
              if name in TRAINABLE_NAMES}
    result = {}
    for variable, gradient in zip(model.trainable_variables, gradients):
        name = by_ref.get(id(variable))
        if name is None or gradient is None:
            raise RuntimeError(f"Keras ResNet18 gradient mapping failed: {variable.path}")
        result[name] = np.asarray(gradient.numpy(), dtype=np.float32).copy()
    if set(result) != set(TRAINABLE_NAMES):
        raise RuntimeError("Keras ResNet18 gradient coverage mismatch")
    return result


def torch_gradients(model) -> dict[str, np.ndarray]:
    result = {}
    for name, tensor in model.state_bindings().items():
        if name not in TRAINABLE_NAMES:
            continue
        if tensor.grad is None:
            raise RuntimeError(f"Missing PyTorch ResNet18 gradient: {name}")
        value = tensor.grad.detach().cpu().numpy()
        if name.endswith(".kernel"):
            value = value.transpose(2, 3, 1, 0) if value.ndim == 4 else value.T
        result[name] = np.asarray(value, dtype=np.float32).copy()
    return result
