"""Compare aligned initial Keras/PyTorch activations for one shared batch."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.dataset_utils import build_split_manifest
from common.environment_utils import select_torch_device

SEEDS = [42, 123, 2026]
BATCH_SIZE = 2
RESULTS = Path(__file__).parent / "results"
LAYER_NAMES = ["Input", "Conv1", "BN1", "ReLU1", "Pool1", "Conv2", "BN2", "ReLU2", "Pool2",
               "Conv3", "BN3", "ReLU3", "Pool3", "Conv4", "BN4", "ReLU4", "Pool4", "GAP", "FC128", "ReLU", "Logits"]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader
    spec.loader.exec_module(module)
    return module


def shared_batch():
    from PIL import Image
    rows = build_split_manifest()["train"][:BATCH_SIZE]
    arrays = []
    for row in rows:
        image = Image.open(str(row["path"])).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(arrays), np.asarray([r["label"] for r in rows], dtype=np.int64)


def build_pair(seed: int):
    base = ROOT / "experiments" / "00_baseline_final_cnn_3seed"
    keras_script = load_script(f"keras_baseline_{seed}", base / "keras.py")
    torch_script = load_script(f"torch_baseline_{seed}", base / "pytorch.py")
    keras_script.ALIGNMENT = torch_script.ALIGNMENT = "diagnostic"
    keras_script.set_seed(seed); torch_script.set_seed(seed)
    return keras_script.build_model(seed), torch_script.build_model(seed).to(select_torch_device())


def keras_outputs(model, batch: np.ndarray, training: bool = False):
    import tensorflow as tf
    names = ["conv1", "bn1", "relu1", "pool1", "conv2", "bn2", "relu2", "pool2",
             "conv3", "bn3", "relu3", "pool3", "conv4", "bn4", "relu4", "pool4",
             "gap", "fc128", "fc_relu", "logits"]
    probe = tf.keras.Model(model.input, [model.get_layer(n).output for n in names])
    values = probe(batch, training=training)
    return {"Input": batch, **{display: value.numpy() for display, value in zip(LAYER_NAMES[1:], values)}}


def torch_outputs(model, batch: np.ndarray, training: bool = False):
    import torch
    model.train(training)
    values = {"Input": batch}
    module_names = []
    kinds = ("Conv", "BN", "ReLU", "Pool")
    for block in range(4): module_names.extend(f"{kind}{block+1}" for kind in kinds)
    hooks = []
    def capture(name):
        def hook(_module, _inputs, output):
            value = output.detach().cpu().numpy()
            values[name] = value.transpose(0, 2, 3, 1) if value.ndim == 4 else value
        return hook
    for name, module in zip(module_names, model.features): hooks.append(module.register_forward_hook(capture(name)))
    hooks.extend([model.gap.register_forward_hook(capture("GAP")), model.fc128.register_forward_hook(capture("FC128")),
                  model.relu.register_forward_hook(capture("ReLU")), model.logits.register_forward_hook(capture("Logits"))])
    device = next(model.parameters()).device
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
    with torch.no_grad(): model(tensor)
    for hook in hooks: hook.remove()
    if values["GAP"].ndim == 4: values["GAP"] = values["GAP"][:, 0, 0, :]
    return values


def metrics(keras_value: np.ndarray, torch_value: np.ndarray) -> dict[str, float]:
    left, right = keras_value.astype(np.float64).ravel(), torch_value.astype(np.float64).ravel()
    diff = left - right
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {"mae": float(np.mean(np.abs(diff))), "max_absolute_error": float(np.max(np.abs(diff))),
            "mse": float(np.mean(diff**2)), "cosine_similarity": float(np.dot(left, right) / denominator) if denominator else 1.0,
            "keras_mean": float(left.mean()), "pytorch_mean": float(right.mean()),
            "keras_std": float(left.std()), "pytorch_std": float(right.std())}


def compare_models(keras_model, torch_model, batch: np.ndarray, seed: int, stage: str) -> list[dict[str, object]]:
    left, right = keras_outputs(keras_model, batch), torch_outputs(torch_model, batch)
    rows = []
    for name in LAYER_NAMES:
        if left[name].shape != right[name].shape:
            raise ValueError(f"{name} shape mismatch: {left[name].shape} vs {right[name].shape}")
        rows.append({"seed": seed, "stage": stage, "layer": name, **metrics(left[name], right[name])})
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); batch, _ = shared_batch()
    for seed in SEEDS:
        keras_model, torch_model = build_pair(seed)
        rows = compare_models(keras_model, torch_model, batch, seed, "initial")
        (RESULTS / f"initial_seed{seed}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(seed, rows[-1])


if __name__ == "__main__": main()
