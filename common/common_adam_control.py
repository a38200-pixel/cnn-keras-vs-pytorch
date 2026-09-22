"""Canonical trainable parameter/gradient bridge for CommonAdam."""

from __future__ import annotations

import numpy as np

TRAINABLE_NAMES = [
    *(name for index in range(1, 5) for name in (
        f"conv{index}/kernel", f"bn{index}/gamma", f"bn{index}/beta",
    )),
    "fc128/kernel", "fc128/bias", "logits/kernel", "logits/bias",
]


def trainable_values(full_state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(full_state[name], dtype=np.float32).copy() for name in TRAINABLE_NAMES}


def keras_gradients(model, gradients) -> dict[str, np.ndarray]:
    lookup = {variable.path: gradient for variable, gradient in zip(model.trainable_variables, gradients)}
    if set(lookup) != set(TRAINABLE_NAMES) or any(value is None for value in lookup.values()):
        raise RuntimeError(f"Keras canonical gradient mapping mismatch: {sorted(lookup)}")
    return {name: np.asarray(lookup[name].numpy(), dtype=np.float32) for name in TRAINABLE_NAMES}


def torch_gradients(model) -> dict[str, np.ndarray]:
    result = {}
    for index in range(1, 5):
        conv = getattr(model, f"conv{index}")
        result[f"conv{index}/kernel"] = conv.weight.grad.detach().cpu().numpy().transpose(2, 3, 1, 0).astype(np.float32, copy=True)
        bn = getattr(model, f"bn{index}")
        result[f"bn{index}/gamma"] = bn.weight.grad.detach().cpu().numpy().astype(np.float32, copy=True)
        result[f"bn{index}/beta"] = bn.bias.grad.detach().cpu().numpy().astype(np.float32, copy=True)
    for name in ("fc128", "logits"):
        layer = getattr(model, name)
        result[f"{name}/kernel"] = layer.weight.grad.detach().cpu().numpy().T.astype(np.float32, copy=True)
        result[f"{name}/bias"] = layer.bias.grad.detach().cpu().numpy().astype(np.float32, copy=True)
    if set(result) != set(TRAINABLE_NAMES):
        raise RuntimeError("PyTorch canonical gradient mapping mismatch")
    return result


def load_keras_trainable(model, values: dict[str, np.ndarray]) -> None:
    for index in range(1, 5):
        model.get_layer(f"conv{index}").kernel.assign(values[f"conv{index}/kernel"])
        bn = model.get_layer(f"bn{index}")
        bn.gamma.assign(values[f"bn{index}/gamma"])
        bn.beta.assign(values[f"bn{index}/beta"])
    for name in ("fc128", "logits"):
        layer = model.get_layer(name)
        layer.kernel.assign(values[f"{name}/kernel"])
        layer.bias.assign(values[f"{name}/bias"])


def load_torch_trainable(model, values: dict[str, np.ndarray]) -> None:
    import torch

    with torch.no_grad():
        for index in range(1, 5):
            conv = getattr(model, f"conv{index}")
            conv.weight.copy_(torch.from_numpy(values[f"conv{index}/kernel"].transpose(3, 2, 0, 1).copy()).to(conv.weight.device))
            bn = getattr(model, f"bn{index}")
            bn.weight.copy_(torch.from_numpy(values[f"bn{index}/gamma"]).to(bn.weight.device))
            bn.bias.copy_(torch.from_numpy(values[f"bn{index}/beta"]).to(bn.bias.device))
        for name in ("fc128", "logits"):
            layer = getattr(model, name)
            layer.weight.copy_(torch.from_numpy(values[f"{name}/kernel"].T.copy()).to(layer.weight.device))
            layer.bias.copy_(torch.from_numpy(values[f"{name}/bias"]).to(layer.bias.device))
