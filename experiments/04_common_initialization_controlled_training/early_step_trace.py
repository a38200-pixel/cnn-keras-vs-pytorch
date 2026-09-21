"""Isolated 0–100 update trajectory; never starts the 30-epoch training loop."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cnn-controlled-mpl"))

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_checkpoint import save_canonical_checkpoint
from common.controlled_config import (
    BETA1, BETA2, DIAGNOSTIC_STEPS, EPSILON, LEARNING_RATE, RESULTS, SEEDS, config_hash,
)
from common.controlled_data import canonical_batch, canonical_rows, iter_batches, load_orders
from common.controlled_initialization import (
    build_keras_model, build_torch_model, export_keras_weights, export_torch_weights,
    load_initial_weights, load_keras_weights, load_torch_weights,
)
from common.controlled_optimizer import export_keras_adam, export_torch_adam
from common.environment_utils import select_torch_device
from common.trace_utils import assert_finite, compare, write_csv
from first_step_trace import keras_gradient_map, torch_gradient_map

PARAMETER_NAMES = [
    *(f"conv{i}/kernel" for i in range(1, 5)),
    *(f"bn{i}/{field}" for i in range(1, 5) for field in ("gamma", "beta")),
    "fc128/kernel", "fc128/bias", "logits/kernel", "logits/bias",
]


def _flat(values: dict[str, np.ndarray], names=PARAMETER_NAMES) -> np.ndarray:
    return np.concatenate([np.asarray(values[name]).ravel() for name in names])


def _state_flat(values: dict[str, np.ndarray], suffix: str) -> np.ndarray:
    return np.concatenate([values[f"{name}/{suffix}"].ravel() for name in PARAMETER_NAMES])


def _checkpoint(seed, step, keras_model, torch_model, keras_optimizer, torch_optimizer):
    kw, pw = export_keras_weights(keras_model), export_torch_weights(torch_model)
    if step:
        ks, ki = export_keras_adam(keras_model, keras_optimizer)
        ps, pi = export_torch_adam(torch_model, torch_optimizer)
        if ki != pi or ki != step or set(ks) != set(ps):
            raise RuntimeError(f"Adam mapping/step mismatch at seed={seed} step={step}")
    else:
        ks = ps = {}
    for framework, weights, slots in (("keras", kw, ks), ("pytorch", pw, ps)):
        save_canonical_checkpoint(
            framework, seed, f"step{step:03d}", weights, slots,
            epoch=0, optimizer_step=step, diagnostic=True,
        )
    return kw, pw, ks, ps


def run_seed(seed: int) -> None:
    import tensorflow as tf
    import torch

    tf.keras.backend.clear_session()
    device = select_torch_device()
    if not tf.config.list_physical_devices("GPU") or device.type != "mps":
        raise RuntimeError("Actual TensorFlow GPU and PyTorch MPS are required")
    rows = canonical_rows("train")
    order = load_orders(seed, len(rows))[0]
    batches = iter_batches(order)
    w0 = load_initial_weights(seed)
    km = build_keras_model()
    pm = build_torch_model().to(device)
    load_keras_weights(km, w0)
    load_torch_weights(pm, w0)
    ko = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=BETA1, beta_2=BETA2, epsilon=EPSILON,
        weight_decay=0.0, amsgrad=False, clipnorm=None, clipvalue=None,
        global_clipnorm=None, use_ema=False,
    )
    po = torch.optim.Adam(pm.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2),
                          eps=EPSILON, weight_decay=0.0, amsgrad=False, foreach=False)
    kc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    pc = torch.nn.CrossEntropyLoss(reduction="mean")
    summary_rows, layer_rows = [], []

    def capture(step, kloss="", ploss="", kacc="", pacc="", kg=None, pg=None, pre_k=None, pre_p=None):
        kw, pw, ks, ps = _checkpoint(seed, step, km, pm, ko, po)
        assert_finite(kw, f"early-step Keras weights {seed}/{step}")
        assert_finite(pw, f"early-step PyTorch weights {seed}/{step}")
        parameter_stats = compare(_flat(kw), _flat(pw))
        gradient_stats = compare(_flat(kg), _flat(pg)) if kg is not None else None
        update_stats = compare(_flat({key: kw[key] - pre_k[key] for key in PARAMETER_NAMES}),
                               _flat({key: pw[key] - pre_p[key] for key in PARAMETER_NAMES})) if step else None
        m_stats = compare(_state_flat(ks, "m"), _state_flat(ps, "m")) if step else None
        v_stats = compare(_state_flat(ks, "v"), _state_flat(ps, "v")) if step else None
        bn_mean = [f"bn{i}/mean" for i in range(1, 5)]
        bn_var = [f"bn{i}/variance" for i in range(1, 5)]
        summary_rows.append({
            "seed": seed, "step": step, "config_sha256": config_hash(),
            "keras_loss": kloss, "pytorch_loss": ploss,
            "loss_abs_diff": abs(kloss - ploss) if step else "",
            "keras_batch_accuracy": kacc, "pytorch_batch_accuracy": pacc,
            "global_weight_relative_l2": parameter_stats["relative_l2_error"],
            "global_gradient_relative_l2": gradient_stats["relative_l2_error"] if gradient_stats else "",
            "global_update_relative_l2": update_stats["relative_l2_error"] if update_stats else "",
            "global_m_relative_l2": m_stats["relative_l2_error"] if m_stats else "",
            "global_v_relative_l2": v_stats["relative_l2_error"] if v_stats else "",
            "bn_mean_relative_l2": compare(_flat(kw, bn_mean), _flat(pw, bn_mean))["relative_l2_error"],
            "bn_var_relative_l2": compare(_flat(kw, bn_var), _flat(pw, bn_var))["relative_l2_error"],
        })
        for name in PARAMETER_NAMES:
            ws = compare(kw[name], pw[name])
            gs = compare(kg[name], pg[name]) if kg is not None else None
            us = compare(kw[name] - pre_k[name], pw[name] - pre_p[name]) if step else None
            ms = compare(ks[f"{name}/m"], ps[f"{name}/m"]) if step else None
            vs = compare(ks[f"{name}/v"], ps[f"{name}/v"]) if step else None
            layer_rows.append({
                "seed": seed, "step": step, "layer": name,
                "weight_mean_abs_diff": ws["mean_abs_diff"],
                "weight_max_abs_diff": ws["max_abs_diff"],
                "weight_relative_l2": ws["relative_l2_error"],
                "weight_cosine_similarity": ws["cosine_similarity"],
                "gradient_relative_l2": gs["relative_l2_error"] if gs else "",
                "update_relative_l2": us["relative_l2_error"] if us else "",
                "m_relative_l2": ms["relative_l2_error"] if ms else "",
                "v_relative_l2": vs["relative_l2_error"] if vs else "",
            })
        print(f"Seed {seed} step {step:03d}: global W rel-L2={parameter_stats['relative_l2_error']:.6g}", flush=True)

    capture(0)
    pm.train()
    for step in range(1, max(DIAGNOSTIC_STEPS) + 1):
        images, labels = canonical_batch(rows, next(batches), seed, 0, True)
        if step in DIAGNOSTIC_STEPS:
            pre_k, pre_p = export_keras_weights(km), export_torch_weights(pm)
        kx = tf.convert_to_tensor(images, dtype=tf.float32)
        ky = tf.convert_to_tensor(labels, dtype=tf.int32)
        px = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device)
        py = torch.from_numpy(labels.astype(np.int64)).to(device)
        with tf.GradientTape() as tape:
            klogits = km(kx, training=True)
            kloss = kc(ky, klogits)
        kgrads = tape.gradient(kloss, km.trainable_variables)
        if any(gradient is None for gradient in kgrads):
            raise RuntimeError(f"Missing Keras gradient step={step}")
        po.zero_grad(set_to_none=True)
        plogits = pm(px)
        ploss = pc(plogits, py)
        ploss.backward()
        if not bool(tf.math.is_finite(kloss).numpy()) or not bool(torch.isfinite(ploss).item()):
            raise RuntimeError(f"NaN/Inf loss at seed={seed} step={step}")
        if step in DIAGNOSTIC_STEPS:
            kg, pg = keras_gradient_map(km, kgrads), torch_gradient_map(pm)
            assert_finite(kg, f"early-step Keras gradient {seed}/{step}")
            assert_finite(pg, f"early-step PyTorch gradient {seed}/{step}")
        ko.apply_gradients(zip(kgrads, km.trainable_variables))
        po.step()
        if step in DIAGNOSTIC_STEPS:
            capture(step, float(kloss.numpy()), float(ploss.detach().cpu()),
                    float(np.mean(klogits.numpy().argmax(1) == labels)),
                    float((plogits.argmax(1) == py).float().mean().item()), kg, pg, pre_k, pre_p)
    output = RESULTS / "early_steps"
    write_csv(output / f"seed{seed}_step_summary.csv", summary_rows)
    write_csv(output / f"seed{seed}_layer_trajectory.csv", layer_rows)
    print(f"Seed {seed}: isolated {max(DIAGNOSTIC_STEPS)}-step diagnostic completed.", flush=True)


def main() -> None:
    for seed in SEEDS:
        run_seed(seed)
    print("Early-step trajectory trace completed; no full training was run.")


if __name__ == "__main__":
    main()
