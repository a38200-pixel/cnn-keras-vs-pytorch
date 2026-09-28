"""Isolated 0–100 step trajectory with CommonAdam and Common BN."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.common_adam_control import (
    TRAINABLE_NAMES, keras_gradients, load_keras_trainable, load_torch_trainable,
    torch_gradients, trainable_values,
)
from common.controlled_batchnorm import build_keras_common_bn_model, build_torch_common_bn_model
from common.controlled_data import canonical_batch, canonical_rows, iter_batches, load_orders
from common.controlled_initialization import (
    export_keras_weights, export_torch_weights, load_initial_weights,
    load_keras_weights, load_torch_weights,
)
from common.environment_utils import select_torch_device
from common.reference_adam import CommonAdam
from common.trace_utils import compare, write_csv, write_json
from experiment_config import DIAGNOSTIC_STEPS, RESULTS, SEEDS, config_hash
from experiment_utils import save_checkpoint

PARENT_RESULTS = ROOT / "experiments/05_gradient_optimizer_divergence/results"


def _flat(values, names):
    return np.concatenate([values[name].ravel() for name in names])


def _parent_step100(seed: int) -> float:
    path = PARENT_RESULTS / "early_steps" / f"seed{seed}_step_summary.csv"
    with path.open(encoding="utf-8") as handle:
        row = next(item for item in csv.DictReader(handle) if int(item["step"]) == 100)
    return float(row["global_weight_relative_l2"])


def run_seed(seed: int) -> dict[str, float]:
    import tensorflow as tf
    import torch

    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx("float32")
    device = select_torch_device()
    if not tf.config.list_physical_devices("GPU") or device.type != "mps":
        raise RuntimeError("TensorFlow Metal GPU and PyTorch MPS are required")
    rows = canonical_rows("train")
    batches = iter_batches(load_orders(seed, len(rows))[0])
    w0 = load_initial_weights(seed)
    km = build_keras_common_bn_model()
    pm = build_torch_common_bn_model().to(device)
    load_keras_weights(km, w0)
    load_torch_weights(pm, w0)
    ka, pa = CommonAdam(trainable_values(w0)), CommonAdam(trainable_values(w0))
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    summaries, layers = [], []

    def capture(step, kloss="", ploss="", kg=None, pg=None, kd=None, pd=None):
        kw, pw = export_keras_weights(km), export_torch_weights(pm)
        save_checkpoint("keras", seed, f"step{step:03d}", kw, ka, 0, step, diagnostic=True)
        save_checkpoint("pytorch", seed, f"step{step:03d}", pw, pa, 0, step, diagnostic=True)
        weight = compare(_flat(kw, TRAINABLE_NAMES), _flat(pw, TRAINABLE_NAMES))
        gradient = compare(_flat(kg, TRAINABLE_NAMES), _flat(pg, TRAINABLE_NAMES)) if step else None
        update = compare(_flat(kd, TRAINABLE_NAMES), _flat(pd, TRAINABLE_NAMES)) if step else None
        kmom, pmom = ka.export_state(), pa.export_state()
        moment_names = [f"{name}/m" for name in TRAINABLE_NAMES]
        velocity_names = [f"{name}/v" for name in TRAINABLE_NAMES]
        m = compare(_flat(kmom, moment_names), _flat(pmom, moment_names)) if step else None
        v = compare(_flat(kmom, velocity_names), _flat(pmom, velocity_names)) if step else None
        row = {
            "seed": seed, "step": step, "config_sha256": config_hash(),
            "keras_loss": kloss, "pytorch_loss": ploss,
            "loss_abs_diff": abs(kloss - ploss) if step else "",
            "global_weight_relative_l2": weight["relative_l2_error"],
            "global_gradient_relative_l2": gradient["relative_l2_error"] if gradient else "",
            "global_update_relative_l2": update["relative_l2_error"] if update else "",
            "global_m_relative_l2": m["relative_l2_error"] if m else "",
            "global_v_relative_l2": v["relative_l2_error"] if v else "",
        }
        for index in range(1, 5):
            row[f"bn{index}_running_mean_relative_l2"] = compare(
                kw[f"bn{index}/mean"], pw[f"bn{index}/mean"]
            )["relative_l2_error"]
            row[f"bn{index}_running_variance_relative_l2"] = compare(
                kw[f"bn{index}/variance"], pw[f"bn{index}/variance"]
            )["relative_l2_error"]
        summaries.append(row)
        for name in TRAINABLE_NAMES:
            ws = compare(kw[name], pw[name])
            layers.append({
                "seed": seed, "step": step, "parameter": name,
                "weight_relative_l2": ws["relative_l2_error"],
                "weight_max_abs_diff": ws["max_abs_diff"],
                "gradient_relative_l2": compare(kg[name], pg[name])["relative_l2_error"] if step else "",
                "update_relative_l2": compare(kd[name], pd[name])["relative_l2_error"] if step else "",
                "m_relative_l2": compare(ka.m[name], pa.m[name])["relative_l2_error"] if step else "",
                "v_relative_l2": compare(ka.v[name], pa.v[name])["relative_l2_error"] if step else "",
            })
        print(f"Seed {seed} step {step:03d}: W rel-L2={weight['relative_l2_error']:.6g}", flush=True)

    capture(0)
    pm.train()
    for step in range(1, max(DIAGNOSTIC_STEPS) + 1):
        images, labels = canonical_batch(rows, next(batches), seed, 0, True)
        kx = tf.convert_to_tensor(images, dtype=tf.float32)
        ky = tf.convert_to_tensor(labels, dtype=tf.int32)
        px = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device)
        py = torch.from_numpy(labels.astype(np.int64)).to(device)
        with tf.GradientTape() as tape:
            klogits = km(kx, training=True)
            kloss = criterion(ky, klogits)
        kg = keras_gradients(km, tape.gradient(kloss, km.trainable_variables))
        pm.zero_grad(set_to_none=True)
        plogits = pm(px)
        ploss = torch.nn.functional.cross_entropy(plogits, py)
        ploss.backward()
        pg = torch_gradients(pm)
        kvalues = trainable_values(export_keras_weights(km))
        pvalues = trainable_values(export_torch_weights(pm))
        knew, kd = ka.update(kvalues, kg)
        pnew, pd = pa.update(pvalues, pg)
        load_keras_trainable(km, knew)
        load_torch_trainable(pm, pnew)
        if step in DIAGNOSTIC_STEPS:
            capture(step, float(kloss.numpy()), float(ploss.detach().cpu()), kg, pg, kd, pd)
    output = RESULTS / "early_steps"
    write_csv(output / f"seed{seed}_step_summary.csv", summaries)
    write_csv(output / f"seed{seed}_layer_trajectory.csv", layers)
    common_bn_step100 = float(summaries[-1]["global_weight_relative_l2"])
    native_bn_step100 = _parent_step100(seed)
    return {
        "experiment05_native_bn_step100_weight_relative_l2": native_bn_step100,
        "experiment06_common_bn_step100_weight_relative_l2": common_bn_step100,
        "common_bn_vs_native_bn_step100_ratio": common_bn_step100 / native_bn_step100,
    }


def main() -> None:
    comparison = {str(seed): run_seed(seed) for seed in SEEDS}
    write_json(RESULTS / "early_steps" / "status.json", {
        "status": "VALID", "steps": list(DIAGNOSTIC_STEPS),
        "config_sha256": config_hash(), "comparison_05_vs_06": comparison,
        "full_training_executed": False,
    })
    print("Experiment 06 early-step Common BN trace completed; no full training was run.")


if __name__ == "__main__":
    main()
