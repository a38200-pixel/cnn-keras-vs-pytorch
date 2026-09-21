"""Version-checked canonical Adam moment export for Keras and PyTorch."""

from __future__ import annotations

import numpy as np


def _torch_canonical(name: str, value: np.ndarray) -> tuple[str, np.ndarray]:
    layer, field = name.split(".", 1)
    if layer.startswith("bn"):
        canonical = "gamma" if field == "weight" else "beta"
        return f"{layer}/{canonical}", value.copy()
    if field == "weight":
        canonical = "kernel"
        if layer.startswith("conv"):
            value = value.transpose(2, 3, 1, 0)
        else:
            value = value.T
    else:
        canonical = "bias"
    return f"{layer}/{canonical}", value.copy()


def export_keras_adam(model, optimizer) -> tuple[dict[str, np.ndarray], int]:
    trainable = list(model.trainable_variables)
    slots = list(optimizer.variables)
    if len(slots) != 2 + 2 * len(trainable):
        raise RuntimeError(f"Unexpected Keras Adam slot count: {len(slots)}")
    if slots[0].name != "iteration" or slots[1].name != "learning_rate":
        raise RuntimeError("Unexpected Keras Adam optimizer metadata order")
    result = {}
    for index, variable in enumerate(trainable):
        name = variable.path
        prefix = name.replace("/", "_")
        m, v = slots[2 + 2 * index], slots[3 + 2 * index]
        if m.name != f"{prefix}_momentum" or v.name != f"{prefix}_velocity":
            raise RuntimeError(f"Keras Adam slot mapping mismatch: {name}: {m.name}/{v.name}")
        if tuple(m.shape) != tuple(variable.shape) or tuple(v.shape) != tuple(variable.shape):
            raise RuntimeError(f"Keras Adam slot shape mismatch: {name}")
        result[f"{name}/m"] = m.numpy().copy()
        result[f"{name}/v"] = v.numpy().copy()
    return result, int(optimizer.iterations.numpy())


def export_torch_adam(model, optimizer) -> tuple[dict[str, np.ndarray], int]:
    result = {}
    steps = set()
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not state or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise RuntimeError(f"Missing PyTorch Adam state: {name}")
        step = state["step"]
        steps.add(int(step.item()) if hasattr(step, "item") else int(step))
        for native, suffix in (("exp_avg", "m"), ("exp_avg_sq", "v")):
            key, value = _torch_canonical(name, state[native].detach().cpu().numpy())
            result[f"{key}/{suffix}"] = value
    if len(steps) != 1:
        raise RuntimeError(f"Inconsistent PyTorch Adam step counters: {steps}")
    return result, steps.pop()
