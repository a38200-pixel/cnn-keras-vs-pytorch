"""Framework-independent W0 and canonical import/export for both CNNs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from common.controlled_config import BN_EPS, KERAS_BN_MOMENTUM, NUM_CLASSES, RESULTS, TORCH_BN_MOMENTUM


def weight_path(seed: int) -> Path:
    return RESULTS / "initialization" / f"seed{seed}_initial_weights.npz"


def _canonical_spec():
    channels = (3, 32, 64, 128, 256)
    for index in range(1, 5):
        yield f"conv{index}/kernel", (3, 3, channels[index - 1], channels[index]), "glorot_uniform"
        for field in ("gamma", "beta", "mean", "variance"):
            yield f"bn{index}/{field}", (channels[index],), field
    yield "fc128/kernel", (256, 128), "glorot_uniform"
    yield "fc128/bias", (128,), "zeros"
    yield "logits/kernel", (128, NUM_CLASSES), "glorot_uniform"
    yield "logits/bias", (NUM_CLASSES,), "zeros"


def create_initial_weights(seed: int, save: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1A17]))
    values: dict[str, np.ndarray] = {}
    metadata = []
    for name, shape, rule in _canonical_spec():
        fan_in = fan_out = None
        if rule == "glorot_uniform":
            if len(shape) == 4:
                fan_in, fan_out = shape[0] * shape[1] * shape[2], shape[0] * shape[1] * shape[3]
            else:
                fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            value = rng.uniform(-limit, limit, shape).astype(np.float32)
        elif rule in {"gamma", "variance"}:
            value = np.ones(shape, dtype=np.float32)
        else:
            value = np.zeros(shape, dtype=np.float32)
        values[name] = value
        metadata.append({
            "layer": name, "shape": list(shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "fan_in": fan_in, "fan_out": fan_out, "rule": rule,
        })
    if save:
        path = weight_path(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path) as archive:
                if set(archive.files) != set(values) or any(
                    not np.array_equal(archive[name], value) for name, value in values.items()
                ):
                    raise RuntimeError(f"Existing W0 differs from deterministic generation: {path}")
        else:
            np.savez_compressed(path, **values)
        manifest_path = path.parent / "initial_weight_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest[str(seed)] = metadata
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return values


def load_initial_weights(seed: int) -> dict[str, np.ndarray]:
    path = weight_path(seed)
    with np.load(path) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    expected = create_initial_weights(seed, save=False)
    if set(values) != set(expected) or any(not np.array_equal(values[name], expected[name]) for name in expected):
        raise RuntimeError(f"Canonical W0 validation failed: {path}")
    return values


def build_keras_model():
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input((128, 128, 3), dtype="float32", name="input")
    x = inputs
    for index, channels in enumerate((32, 64, 128, 256), 1):
        x = layers.Conv2D(channels, 3, padding="same", use_bias=False, name=f"conv{index}")(x)
        x = layers.BatchNormalization(epsilon=BN_EPS, momentum=KERAS_BN_MOMENTUM, name=f"bn{index}")(x)
        x = layers.ReLU(name=f"relu{index}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{index}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, name="fc128")(x)
    x = layers.ReLU(name="dense128_relu")(x)
    outputs = layers.Dense(NUM_CLASSES, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="controlled_final_rgb_cnn")


def build_torch_model():
    import torch
    from torch import nn

    class ControlledCNN(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (3, 32, 64, 128, 256)
            for index in range(1, 5):
                setattr(self, f"conv{index}", nn.Conv2d(channels[index - 1], channels[index], 3, padding=1, bias=False))
                setattr(self, f"bn{index}", nn.BatchNorm2d(channels[index], eps=BN_EPS, momentum=TORCH_BN_MOMENTUM))
                setattr(self, f"relu{index}", nn.ReLU())
                setattr(self, f"pool{index}", nn.MaxPool2d(2))
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc128 = nn.Linear(256, 128)
            self.dense128_relu = nn.ReLU()
            self.logits = nn.Linear(128, NUM_CLASSES)

        def forward(self, x):
            for index in range(1, 5):
                x = getattr(self, f"pool{index}")(
                    getattr(self, f"relu{index}")(
                        getattr(self, f"bn{index}")(
                            getattr(self, f"conv{index}")(x)
                        )
                    )
                )
            x = self.gap(x).flatten(1)
            return self.logits(self.dense128_relu(self.fc128(x)))

    return ControlledCNN()


def load_keras_weights(model, values: dict[str, np.ndarray]) -> None:
    for index in range(1, 5):
        model.get_layer(f"conv{index}").set_weights([values[f"conv{index}/kernel"]])
        model.get_layer(f"bn{index}").set_weights([
            values[f"bn{index}/{field}"] for field in ("gamma", "beta", "mean", "variance")
        ])
    for name in ("fc128", "logits"):
        model.get_layer(name).set_weights([values[f"{name}/kernel"], values[f"{name}/bias"]])


def load_torch_weights(model, values: dict[str, np.ndarray]) -> None:
    import torch

    with torch.no_grad():
        for index in range(1, 5):
            getattr(model, f"conv{index}").weight.copy_(
                torch.from_numpy(values[f"conv{index}/kernel"].transpose(3, 2, 0, 1).copy())
            )
            bn = getattr(model, f"bn{index}")
            for field, torch_field in (("gamma", "weight"), ("beta", "bias"), ("mean", "running_mean"), ("variance", "running_var")):
                getattr(bn, torch_field).copy_(torch.from_numpy(values[f"bn{index}/{field}"]))
            bn.num_batches_tracked.zero_()
        for name in ("fc128", "logits"):
            layer = getattr(model, name)
            layer.weight.copy_(torch.from_numpy(values[f"{name}/kernel"].T.copy()))
            layer.bias.copy_(torch.from_numpy(values[f"{name}/bias"]))


def export_keras_weights(model) -> dict[str, np.ndarray]:
    result = {}
    for index in range(1, 5):
        result[f"conv{index}/kernel"] = model.get_layer(f"conv{index}").get_weights()[0].copy()
        for field, value in zip(("gamma", "beta", "mean", "variance"), model.get_layer(f"bn{index}").get_weights()):
            result[f"bn{index}/{field}"] = value.copy()
    for name in ("fc128", "logits"):
        kernel, bias = model.get_layer(name).get_weights()
        result[f"{name}/kernel"], result[f"{name}/bias"] = kernel.copy(), bias.copy()
    return result


def export_torch_weights(model) -> dict[str, np.ndarray]:
    result = {}
    for index in range(1, 5):
        result[f"conv{index}/kernel"] = getattr(model, f"conv{index}").weight.detach().cpu().numpy().transpose(2, 3, 1, 0).copy()
        bn = getattr(model, f"bn{index}")
        for field, torch_field in (("gamma", "weight"), ("beta", "bias"), ("mean", "running_mean"), ("variance", "running_var")):
            result[f"bn{index}/{field}"] = getattr(bn, torch_field).detach().cpu().numpy().copy()
    for name in ("fc128", "logits"):
        layer = getattr(model, name)
        result[f"{name}/kernel"] = layer.weight.detach().cpu().numpy().T.copy()
        result[f"{name}/bias"] = layer.bias.detach().cpu().numpy().copy()
    return result


def write_initial_audit(seed: int, canonical, keras_values, torch_values) -> tuple[Path, float]:
    path = RESULTS / "preflight" / f"seed{seed}_initial_weight_audit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    max_diff = 0.0
    for name in canonical:
        a, b, reference = keras_values[name], torch_values[name], canonical[name]
        if a.shape != b.shape or a.dtype != b.dtype or a.shape != reference.shape:
            raise RuntimeError(f"Initial weight shape/dtype mismatch: {name}")
        diff = np.abs(a - b)
        maximum = float(max(diff.max(), np.abs(a - reference).max(), np.abs(b - reference).max()))
        max_diff = max(max_diff, maximum)
        rows.append({
            "seed": seed, "layer": name,
            "keras_hash": hashlib.sha256(a.tobytes()).hexdigest(),
            "pytorch_hash": hashlib.sha256(b.tobytes()).hexdigest(),
            "canonical_hash": hashlib.sha256(reference.tobytes()).hexdigest(),
            "max_abs_diff": maximum, "mean_abs_diff": float(diff.mean()),
            "mse": float(np.mean(np.square(a - b))),
            "match": bool(np.array_equal(a, b) and np.array_equal(a, reference)),
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, max_diff
