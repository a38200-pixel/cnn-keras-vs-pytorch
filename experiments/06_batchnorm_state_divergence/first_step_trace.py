"""First-step Common BN diagnostic; never starts full training."""

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
from common.controlled_batchnorm import (
    COMMON_BN_SEMANTICS_ID, build_keras_common_bn_model, build_torch_common_bn_model,
    keras_uses_native_batchnorm, torch_uses_native_batchnorm,
)
from common.controlled_data import canonical_batch, canonical_rows, load_orders
from common.controlled_initialization import (
    export_keras_weights, export_torch_weights, load_initial_weights,
    load_keras_weights, load_torch_weights,
)
from common.environment_utils import select_torch_device
from common.reference_adam import CommonAdam
from common.trace_utils import assert_finite, compare, write_csv, write_json
from experiment_config import BATCH_SIZE, RESULTS, SEEDS, config_hash

PARENT_RESULTS = ROOT / "experiments/05_gradient_optimizer_divergence/results"
NATIVE_FORWARD_RESULTS = ROOT / "experiments/04_common_initialization_controlled_training/results/first_step"


def _relative(left: dict[str, np.ndarray], right: dict[str, np.ndarray], names: list[str]) -> float:
    return float(compare(
        np.concatenate([left[name].ravel() for name in names]),
        np.concatenate([right[name].ravel() for name in names]),
    )["relative_l2_error"])


def _parent_metrics(seed: int) -> dict[str, float]:
    summary = json.loads((PARENT_RESULTS / "first_step" / f"seed{seed}_summary.json").read_text(encoding="utf-8"))
    with (NATIVE_FORWARD_RESULTS / f"seed{seed}_forward_distribution.csv").open(encoding="utf-8") as handle:
        native_forward = {row["layer"]: row for row in csv.DictReader(handle)}
    states = {}
    for framework in ("keras", "pytorch"):
        path = PARENT_RESULTS / "checkpoints" / f"seed{seed}" / framework / "after_first_step.npz"
        with np.load(path, allow_pickle=False) as archive:
            states[framework] = {name: archive[name].copy() for name in archive.files}
    mean_names = [f"bn{index}/mean" for index in range(1, 5)]
    variance_names = [f"bn{index}/variance" for index in range(1, 5)]
    return {
        "bn1_output_relative_l2": float(native_forward["bn1"]["relative_l2_error"]),
        "bn1_output_max_abs_diff": float(native_forward["bn1"]["max_abs_diff"]),
        "bn_running_mean_relative_l2": _relative(states["keras"], states["pytorch"], mean_names),
        "bn_running_variance_relative_l2": _relative(states["keras"], states["pytorch"], variance_names),
        "gradient_relative_l2": float(summary["gradient_relative_l2"]),
        "update_relative_l2": float(summary["update_relative_l2"]),
        "weight_relative_l2": float(summary["weight_relative_l2"]),
    }


def run_seed(seed: int) -> dict[str, object]:
    import tensorflow as tf
    import torch

    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx("float32")
    device = select_torch_device()
    if not tf.config.list_physical_devices("GPU") or device.type != "mps":
        raise RuntimeError("TensorFlow Metal GPU and PyTorch MPS are required")
    rows = canonical_rows("train")
    order = load_orders(seed, len(rows))[0]
    images, labels = canonical_batch(rows, order[:BATCH_SIZE], seed, 0, True)
    kx = tf.convert_to_tensor(images, dtype=tf.float32)
    px = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device)
    py = torch.from_numpy(labels.astype(np.int64)).to(device)
    if not np.array_equal(images, px.cpu().numpy().transpose(0, 2, 3, 1)):
        raise RuntimeError("Input mismatch")

    w0 = load_initial_weights(seed)
    km = build_keras_common_bn_model()
    pm = build_torch_common_bn_model().to(device)
    if keras_uses_native_batchnorm(km) or torch_uses_native_batchnorm(pm):
        raise RuntimeError("Native BatchNorm found in Experiment 06 diagnostic path")
    load_keras_weights(km, w0)
    load_torch_weights(pm, w0)
    kw0, pw0 = export_keras_weights(km), export_torch_weights(pm)
    if any(not np.array_equal(kw0[name], pw0[name]) for name in w0):
        raise RuntimeError("W0/Common BN initial state mismatch")

    keras_outputs = []
    keras_names = []
    for index in range(1, 5):
        keras_names.extend((f"conv{index}", f"bn{index}"))
        keras_outputs.extend((km.get_layer(f"conv{index}").output, km.get_layer(f"bn{index}").output))
    trace_model = tf.keras.Model(km.input, [*keras_outputs, km.output])
    with tf.GradientTape() as tape:
        traced = trace_model(kx, training=True)
        klogits = traced[-1]
        kloss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, klogits, from_logits=True))
    kgrads_raw = tape.gradient(kloss, km.trainable_variables)
    kg = keras_gradients(km, kgrads_raw)
    ktrace = dict(zip(keras_names, traced[:-1]))

    ptrace = {}
    hooks = []
    def capture(name):
        def hook(_module, _inputs, output):
            ptrace[name] = output
        return hook
    for index in range(1, 5):
        hooks.append(getattr(pm, f"conv{index}").register_forward_hook(capture(f"conv{index}")))
        hooks.append(getattr(pm, f"bn{index}").register_forward_hook(capture(f"bn{index}")))
    pm.train()
    pm.zero_grad(set_to_none=True)
    plogits = pm(px)
    ploss = torch.nn.functional.cross_entropy(plogits, py)
    ploss.backward()
    for hook in hooks:
        hook.remove()
    pg = torch_gradients(pm)
    assert_finite(kg, "Keras gradients")
    assert_finite(pg, "PyTorch gradients")
    gradient_flow_valid = (
        all(gradient is not None for gradient in kgrads_raw)
        and all(parameter.grad is not None for parameter in pm.parameters() if parameter.requires_grad)
    )
    if not gradient_flow_valid:
        raise RuntimeError("Gradient flow through Common BN is incomplete")

    forward_rows = []
    bn_rows = []
    first_nonzero = None
    for name in keras_names:
        ka = ktrace[name].numpy()
        pa = ptrace[name].detach().cpu().numpy().transpose(0, 2, 3, 1)
        stats = compare(ka, pa)
        forward_rows.append({"seed": seed, "layer": name, **stats})
        if first_nonzero is None and not np.array_equal(ka, pa):
            first_nonzero = name
    kw_after_forward, pw_after_forward = export_keras_weights(km), export_torch_weights(pm)
    for index in range(1, 5):
        kconv = ktrace[f"conv{index}"]
        pconv = ptrace[f"conv{index}"]
        kmean = tf.reduce_mean(kconv, axis=(0, 1, 2)).numpy()
        kcentered = tf.subtract(kconv, tf.reduce_mean(kconv, axis=(0, 1, 2)))
        kvar = tf.reduce_mean(tf.square(kcentered), axis=(0, 1, 2)).numpy()
        pmean_tensor = torch.mean(pconv, dim=(0, 2, 3))
        pcentered = torch.sub(pconv, pmean_tensor[None, :, None, None])
        pvar = torch.mean(torch.square(pcentered), dim=(0, 2, 3)).detach().cpu().numpy()
        pmean = pmean_tensor.detach().cpu().numpy()
        tensors = {
            "batch_mean": (kmean, pmean),
            "batch_variance": (kvar, pvar),
            "output": (
                ktrace[f"bn{index}"].numpy(),
                ptrace[f"bn{index}"].detach().cpu().numpy().transpose(0, 2, 3, 1),
            ),
            "running_mean": (kw_after_forward[f"bn{index}/mean"], pw_after_forward[f"bn{index}/mean"]),
            "running_variance": (kw_after_forward[f"bn{index}/variance"], pw_after_forward[f"bn{index}/variance"]),
        }
        for tensor_name, (left, right) in tensors.items():
            bn_rows.append({"seed": seed, "layer": f"bn{index}", "tensor": tensor_name, **compare(left, right)})

    ka = CommonAdam(trainable_values(kw0))
    pa = CommonAdam(trainable_values(pw0))
    kupdated, kd = ka.update(trainable_values(kw_after_forward), kg)
    pupdated, pd = pa.update(trainable_values(pw_after_forward), pg)
    load_keras_trainable(km, kupdated)
    load_torch_trainable(pm, pupdated)
    kw1, pw1 = export_keras_weights(km), export_torch_weights(pm)
    gradient_rows = [{"seed": seed, "parameter": name, **compare(kg[name], pg[name])} for name in TRAINABLE_NAMES]
    update_rows = [{"seed": seed, "parameter": name, **compare(kd[name], pd[name])} for name in TRAINABLE_NAMES]
    weight_rows = [{"seed": seed, "parameter": name, **compare(kupdated[name], pupdated[name])} for name in TRAINABLE_NAMES]
    m_rows = [{"seed": seed, "parameter": name, **compare(ka.m[name], pa.m[name])} for name in TRAINABLE_NAMES]
    v_rows = [{"seed": seed, "parameter": name, **compare(ka.v[name], pa.v[name])} for name in TRAINABLE_NAMES]
    output = RESULTS / "first_step"
    write_csv(output / f"seed{seed}_forward_trace.csv", forward_rows)
    write_csv(output / f"seed{seed}_bn_trace.csv", bn_rows)
    for name, values in (("gradient", gradient_rows), ("update", update_rows), ("weight", weight_rows), ("m", m_rows), ("v", v_rows)):
        write_csv(output / f"seed{seed}_{name}_trace.csv", values)

    mean_names = [f"bn{index}/mean" for index in range(1, 5)]
    variance_names = [f"bn{index}/variance" for index in range(1, 5)]
    parent = _parent_metrics(seed)
    bn1 = {(row["layer"], row["tensor"]): row for row in bn_rows}
    summary = {
        "seed": seed, "config_sha256": config_hash(), "w0_exact": True,
        "input_exact": True, "labels_exact": True, "bn_initial_state_exact": True,
        "conv1_exact": bool(np.array_equal(ktrace["conv1"].numpy(), ptrace["conv1"].detach().cpu().numpy().transpose(0, 2, 3, 1))),
        "first_nonzero_layer": first_nonzero or "none",
        "common_bn_semantics_id": COMMON_BN_SEMANTICS_ID,
        "same_common_bn_semantics": True,
        "native_batchnorm_absent": True,
        "common_adam_class": f"{CommonAdam.__module__}.{CommonAdam.__name__}",
        "same_common_adam_implementation": type(ka) is type(pa), "common_adam_step": ka.step,
        "gradient_flow_valid": gradient_flow_valid,
        "keras_loss": float(kloss.numpy()), "pytorch_loss": float(ploss.detach().cpu()),
        "bn1_batch_mean_relative_l2": float(bn1[("bn1", "batch_mean")]["relative_l2_error"]),
        "bn1_batch_variance_relative_l2": float(bn1[("bn1", "batch_variance")]["relative_l2_error"]),
        "bn1_output_relative_l2": float(bn1[("bn1", "output")]["relative_l2_error"]),
        "bn1_output_max_abs_diff": float(bn1[("bn1", "output")]["max_abs_diff"]),
        "bn_running_mean_relative_l2": _relative(kw1, pw1, mean_names),
        "bn_running_variance_relative_l2": _relative(kw1, pw1, variance_names),
        "gradient_relative_l2": _relative(kg, pg, TRAINABLE_NAMES),
        "update_relative_l2": _relative(kd, pd, TRAINABLE_NAMES),
        "weight_relative_l2": _relative(kupdated, pupdated, TRAINABLE_NAMES),
        "m_relative_l2": _relative(ka.m, pa.m, TRAINABLE_NAMES),
        "v_relative_l2": _relative(ka.v, pa.v, TRAINABLE_NAMES),
        "experiment05_native_bn": parent,
        "nan_or_inf": False, "keras_device": kx.device, "pytorch_device": str(device),
    }
    assert_finite(kw1, "Keras post-step state")
    assert_finite(pw1, "PyTorch post-step state")
    write_json(output / f"seed{seed}_summary.json", summary)
    print(
        f"Seed {seed}: BN1 out={summary['bn1_output_relative_l2']:.6g} "
        f"grad={summary['gradient_relative_l2']:.6g} update={summary['update_relative_l2']:.6g}",
        flush=True,
    )
    return summary


def main() -> None:
    summaries = [run_seed(seed) for seed in SEEDS]
    write_json(RESULTS / "first_step" / "diagnostic_3seed_summary.json", {
        "experiment": "06_batchnorm_state_divergence", "config_sha256": config_hash(),
        "status": "VALID", "by_seed": {str(item["seed"]): item for item in summaries},
        "full_training_executed": False,
    })
    print("Experiment 06 first-step Common BN diagnostic completed; no full training was run.")


if __name__ == "__main__":
    main()
