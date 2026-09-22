"""Common-Adam first-step validation; isolated from full training."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.common_adam_control import (
    TRAINABLE_NAMES, keras_gradients, load_keras_trainable, load_torch_trainable,
    torch_gradients, trainable_values,
)
from common.controlled_data import canonical_batch, canonical_rows, load_orders
from common.controlled_initialization import (
    build_keras_model, build_torch_model, export_keras_weights, export_torch_weights,
    load_initial_weights, load_keras_weights, load_torch_weights,
)
from common.environment_utils import select_torch_device
from common.reference_adam import CommonAdam
from common.trace_utils import assert_finite, compare, write_csv, write_json
from experiment_config import BATCH_SIZE, RESULTS, SEEDS, config_hash


def _global(rows: list[dict[str, object]]) -> float:
    numerator = sum(float(row["l2_norm_difference"]) ** 2 for row in rows) ** .5
    left = sum(float(row["keras_norm"]) ** 2 for row in rows) ** .5
    right = sum(float(row["pytorch_norm"]) ** 2 for row in rows) ** .5
    return numerator / max(left, right, 1e-12)


def _native_04_update(seed: int) -> float:
    path = ROOT / "experiments/04_common_initialization_controlled_training/results/first_step" / f"seed{seed}_update_distribution.csv"
    with path.open(encoding="utf-8") as handle:
        return _global(list(csv.DictReader(handle)))


def run_seed(seed: int) -> dict[str, object]:
    import tensorflow as tf
    import torch

    tf.keras.backend.clear_session(); tf.keras.backend.set_floatx("float32")
    device = select_torch_device()
    if not tf.config.list_physical_devices("GPU") or device.type != "mps":
        raise RuntimeError("TensorFlow Metal GPU and PyTorch MPS are required")
    rows = canonical_rows("train"); order = load_orders(seed, len(rows))[0]
    images, labels = canonical_batch(rows, order[:BATCH_SIZE], seed, 0, True)
    kx = tf.convert_to_tensor(images, dtype=tf.float32)
    px = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device)
    py = torch.from_numpy(labels.astype(np.int64)).to(device)
    if not np.array_equal(images, px.cpu().numpy().transpose(0, 2, 3, 1)):
        raise RuntimeError("Input mismatch")

    w0 = load_initial_weights(seed)
    km = build_keras_model(); pm = build_torch_model().to(device)
    load_keras_weights(km, w0); load_torch_weights(pm, w0)
    kw0, pw0 = export_keras_weights(km), export_torch_weights(pm)
    if any(not np.array_equal(kw0[name], pw0[name]) for name in w0):
        raise RuntimeError("W0 mismatch")

    trace_model = tf.keras.Model(km.input, [km.get_layer("conv1").output, km.output])
    with tf.GradientTape() as tape:
        kconv1, klogits = trace_model(kx, training=True)
        kloss = tf.keras.losses.sparse_categorical_crossentropy(labels, klogits, from_logits=True)
        kloss = tf.reduce_mean(kloss)
    kg = keras_gradients(km, tape.gradient(kloss, km.trainable_variables))
    captured = {}
    def capture_conv1(_module, _inputs, output):
        captured["conv1"] = output.detach().cpu().numpy().transpose(0, 2, 3, 1).copy()
    hook = pm.conv1.register_forward_hook(capture_conv1)
    pm.train(); pm.zero_grad(set_to_none=True)
    plogits = pm(px); ploss = torch.nn.functional.cross_entropy(plogits, py); ploss.backward(); hook.remove()
    pg = torch_gradients(pm)
    if not np.array_equal(kconv1.numpy(), captured["conv1"]):
        raise RuntimeError("Conv1 must be exact before Common Adam")
    assert_finite(kg, "Keras gradients"); assert_finite(pg, "PyTorch gradients")

    ka = CommonAdam(trainable_values(kw0)); pa = CommonAdam(trainable_values(pw0))
    kupdated, kd = ka.update(trainable_values(kw0), kg)
    pupdated, pd = pa.update(trainable_values(pw0), pg)
    if ka.step != pa.step or ka.step != 1:
        raise RuntimeError("Common Adam step counter mismatch")
    load_keras_trainable(km, kupdated); load_torch_trainable(pm, pupdated)
    kw1, pw1 = export_keras_weights(km), export_torch_weights(pm)

    gradient_rows = [{"seed": seed, "parameter": name, **compare(kg[name], pg[name])} for name in TRAINABLE_NAMES]
    update_rows = [{"seed": seed, "parameter": name, **compare(kd[name], pd[name])} for name in TRAINABLE_NAMES]
    m_rows = [{"seed": seed, "parameter": name, **compare(ka.m[name], pa.m[name])} for name in TRAINABLE_NAMES]
    v_rows = [{"seed": seed, "parameter": name, **compare(ka.v[name], pa.v[name])} for name in TRAINABLE_NAMES]
    weight_rows = [{"seed": seed, "parameter": name, **compare(kupdated[name], pupdated[name])} for name in TRAINABLE_NAMES]
    output = RESULTS / "first_step"
    for name, values in (("gradient", gradient_rows), ("update", update_rows), ("m", m_rows), ("v", v_rows), ("weight", weight_rows)):
        write_csv(output / f"seed{seed}_{name}_trace.csv", values)
    common_update = _global(update_rows); native_update = _native_04_update(seed)
    summary = {
        "seed": seed, "config_sha256": config_hash(), "w0_exact": True,
        "input_exact": True, "conv1_exact": True, "labels_exact": True,
        "common_adam_class": f"{CommonAdam.__module__}.{CommonAdam.__name__}",
        "same_common_adam_implementation": type(ka) is type(pa), "common_adam_step": 1,
        "keras_loss": float(kloss.numpy()), "pytorch_loss": float(ploss.detach().cpu()),
        "gradient_relative_l2": _global(gradient_rows),
        "update_relative_l2": common_update, "m_relative_l2": _global(m_rows),
        "v_relative_l2": _global(v_rows), "weight_relative_l2": _global(weight_rows),
        "experiment04_native_update_relative_l2": native_update,
        "common_vs_native_update_ratio": common_update / native_update,
        "nan_or_inf": False, "keras_device": kx.device, "pytorch_device": str(device),
    }
    write_json(output / f"seed{seed}_summary.json", summary)
    print(f"Seed {seed}: grad rel-L2={summary['gradient_relative_l2']:.6g} common update={common_update:.6g} native04={native_update:.6g}", flush=True)
    return summary


def main() -> None:
    summaries = [run_seed(seed) for seed in SEEDS]
    write_json(RESULTS / "first_step" / "diagnostic_3seed_summary.json", {
        "experiment": "05_gradient_optimizer_divergence", "config_sha256": config_hash(),
        "status": "VALID", "by_seed": {str(item["seed"]): item for item in summaries},
    })
    print("Experiment 05 first-step Common Adam validation completed; no full training was run.")


if __name__ == "__main__":
    main()
