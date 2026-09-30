"""V4 Stage A: three synchronized CIFAR-10 initial GPU one-step traces."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from common.reference_adam import CommonAdam
from common.state_resynchronization import compare_group, states_exact
from common.trace_utils import assert_finite, compare, write_csv, write_json
from cifar10_canonical_state import (
    TRAINABLE_NAMES, create_initial_state, export_keras_state, export_torch_state,
    keras_gradients, load_keras_state, load_torch_state, torch_gradients, trainable_values,
)
from cifar10_keras import build_keras_cifar10
from cifar10_pytorch import build_torch_cifar10
from v4_config import REPEATABILITY_ATOL, REPEATABILITY_RTOL, RESULTS, RUN_FULL_TRAINING, SEEDS, config_hash
from v4_data import canonical_images, require_local_cifar10, stratified_split

ACTIVATION_STAGES = [
    "logits", "fc128.relu", "fc128.preactivation", "gap",
    *(name for index in range(4, 0, -1) for name in (
        f"pool{index}", f"relu{index}", f"bn{index}.output", f"conv{index}",
    )),
]


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value).copy()


def _canonical(value, pytorch=False):
    array = _numpy(value)
    return array.transpose(0, 2, 3, 1).copy() if pytorch and array.ndim == 4 else array


def _metric(a, b, activation=True):
    x, y = _canonical(a), _canonical(b, pytorch=activation)
    stats = compare(x, y)
    return {
        "shape": json.dumps(x.shape), "keras_dtype": str(x.dtype), "pytorch_dtype": str(y.dtype),
        "exact_equal": bool(np.array_equal(x, y)), "keras_norm": stats["keras_norm"],
        "pytorch_norm": stats["pytorch_norm"], "max_abs_diff": stats["max_abs_diff"],
        "mean_abs_diff": stats["mean_abs_diff"], "relative_l2": stats["relative_l2_error"],
        "cosine_similarity": stats["cosine_similarity"],
        "sign_mismatch_fraction": stats["sign_mismatch_ratio"],
    }


class Runner:
    def __init__(self, images, labels):
        import tensorflow as tf
        import torch
        self.tf, self.torch = tf, torch
        if not tf.config.list_physical_devices("GPU") or not torch.backends.mps.is_available():
            raise RuntimeError("V4 Stage A requires TensorFlow Metal GPU and PyTorch MPS")
        self.device = torch.device("mps")
        self.images, self.labels = images, labels.astype(np.int32)
        with tf.device("/GPU:0"):
            self.kx = tf.convert_to_tensor(images, tf.float32); self.ky = tf.convert_to_tensor(self.labels, tf.int32)
            self.kmodel = build_keras_cifar10()
        self.px = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(self.device)
        self.py = torch.from_numpy(self.labels.astype(np.int64)).to(self.device)
        self.pmodel = build_torch_cifar10().to(self.device)
        self.pmodel.gradient_trace_names = set(ACTIVATION_STAGES)

    def sync(self, state):
        load_keras_state(self.kmodel, state); load_torch_state(self.pmodel, state)
        ks, ps = export_keras_state(self.kmodel), export_torch_state(self.pmodel)
        ka, pa = CommonAdam(trainable_values(state)), CommonAdam(trainable_values(state))
        exact = states_exact(ks, ps) and states_exact(ka.m, pa.m) and states_exact(ka.v, pa.v) and ka.step == pa.step == 0
        if not exact:
            raise RuntimeError("V4 Stage A synchronization failed")
        return exact

    def keras_step(self, state, traced):
        tf = self.tf; load_keras_state(self.kmodel, state); adam = CommonAdam(trainable_values(state))
        with tf.device("/GPU:0"):
            with tf.GradientTape() as tape:
                logits = self.kmodel(self.kx, training=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(self.ky, logits, from_logits=True))
            trace = dict(self.kmodel.last_trace)
            if traced:
                targets = [trace[name] for name in ACTIVATION_STAGES] + list(self.kmodel.trainable_variables)
                gradients = tape.gradient(loss, targets)
                activation = dict(zip(ACTIVATION_STAGES, gradients[:len(ACTIVATION_STAGES)]))
                parameter_values = gradients[len(ACTIVATION_STAGES):]
            else:
                parameter_values = tape.gradient(loss, self.kmodel.trainable_variables); activation = None
            if any(value is None for value in parameter_values) or traced and any(value is None for value in activation.values()):
                raise RuntimeError("Missing Keras V4 gradient")
            parameter = keras_gradients(self.kmodel, parameter_values)
        before = export_keras_state(self.kmodel); updated, updates = adam.update(trainable_values(before), parameter)
        merged = dict(before); merged.update(updated); load_keras_state(self.kmodel, merged)
        result = {"logits": _numpy(logits), "loss": float(loss.numpy()), "parameter_gradients": parameter,
                  "updates": updates, "adam_m": adam.m, "adam_v": adam.v, "post": export_keras_state(self.kmodel)}
        if traced: result.update({"forward": trace, "activation": activation})
        return result

    def torch_step(self, state, traced):
        torch = self.torch; load_torch_state(self.pmodel, state); adam = CommonAdam(trainable_values(state))
        self.pmodel.train(); self.pmodel.trace_gradients = traced; self.pmodel.zero_grad(set_to_none=True)
        logits = self.pmodel(self.px); loss = torch.nn.functional.cross_entropy(logits, self.py); loss.backward()
        trace = dict(self.pmodel.last_trace); parameter = torch_gradients(self.pmodel)
        activation = None
        if traced:
            activation = {name: trace[name].grad for name in ACTIVATION_STAGES}
            if any(value is None for value in activation.values()): raise RuntimeError("Missing PyTorch V4 gradient")
        before = export_torch_state(self.pmodel); updated, updates = adam.update(trainable_values(before), parameter)
        merged = dict(before); merged.update(updated); load_torch_state(self.pmodel, merged)
        result = {"logits": _numpy(logits), "loss": float(loss.detach().cpu()), "parameter_gradients": parameter,
                  "updates": updates, "adam_m": adam.m, "adam_v": adam.v, "post": export_torch_state(self.pmodel)}
        if traced: result.update({"forward": trace, "activation": activation})
        return result

    def pair(self, state, traced):
        exact = self.sync(state)
        keras = self.keras_step(state, traced)
        pytorch = self.torch_step(state, traced)
        for branch, result in (("keras", keras), ("pytorch", pytorch)):
            for key in ("parameter_gradients", "updates", "adam_m", "adam_v", "post"):
                assert_finite(result[key], f"V4 {branch} {key}")
            assert_finite({"logits": result["logits"], "loss": np.asarray([result["loss"]])}, f"V4 {branch} outputs")
            if traced:
                assert_finite({name: _numpy(value) for name, value in result["forward"].items()}, f"V4 {branch} forward")
                assert_finite({name: _numpy(value) for name, value in result["activation"].items()}, f"V4 {branch} activation gradients")
        return exact, keras, pytorch


def _global(result):
    return {
        "logits": np.asarray(result["logits"]), "loss": np.asarray([result["loss"]], np.float32),
        "global_gradient": np.concatenate([result["parameter_gradients"][name].ravel() for name in TRAINABLE_NAMES]),
        "global_update": np.concatenate([result["updates"][name].ravel() for name in TRAINABLE_NAMES]),
        "post_weight": np.concatenate([result["post"][name].ravel() for name in TRAINABLE_NAMES]),
    }


def _validation_rows(case, kind, framework, left, right):
    rows = []
    for metric, a in _global(left).items():
        b = _global(right)[metric]; stats = compare(a, b)
        rows.append({"case_id": case, "comparison": kind, "framework": framework, "metric": metric,
                     "exact_equal": bool(np.array_equal(a, b)),
                     "within_qualified_tolerance": bool(np.allclose(a, b, rtol=REPEATABILITY_RTOL, atol=REPEATABILITY_ATOL)),
                     "max_abs_diff": stats["max_abs_diff"], "relative_l2": stats["relative_l2_error"]})
    return rows


def _first(rows, field="stage"):
    return next((row[field] for row in rows if float(row["max_abs_diff"]) > 0), "none")


def _bn_rows(seed, forward_rows, parameter_rows, update_rows, keras, pytorch):
    """Collect the CommonBN internals without executing an extra forward pass."""
    forward = {row["stage"]: row for row in forward_rows}
    parameters = {row["parameter"]: row for row in parameter_rows}
    updates = {row["parameter"]: row for row in update_rows}
    rows = []
    for index in range(1, 5):
        prefix = f"bn{index}"
        mean = compare(keras["post"][f"{prefix}/mean"], pytorch["post"][f"{prefix}/mean"])
        variance = compare(
            keras["post"][f"{prefix}/variance"], pytorch["post"][f"{prefix}/variance"],
        )
        rows.append({
            "seed": seed, "bn_layer": prefix,
            "input_relative_l2": forward[f"{prefix}.input"]["relative_l2"],
            "batch_mean_relative_l2": forward[f"{prefix}.batch_mean"]["relative_l2"],
            "batch_variance_relative_l2": forward[f"{prefix}.batch_variance"]["relative_l2"],
            "x_hat_relative_l2": forward[f"{prefix}.x_hat"]["relative_l2"],
            "affine_output_relative_l2": forward[f"{prefix}.output"]["relative_l2"],
            "gamma_gradient_relative_l2": parameters[f"{prefix}/gamma"]["relative_l2"],
            "beta_gradient_relative_l2": parameters[f"{prefix}/beta"]["relative_l2"],
            "gamma_update_relative_l2": updates[f"{prefix}/gamma"]["update_relative_l2"],
            "beta_update_relative_l2": updates[f"{prefix}/beta"]["update_relative_l2"],
            "running_mean_post_step_relative_l2": mean["relative_l2_error"],
            "running_variance_post_step_relative_l2": variance["relative_l2_error"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path("data/cifar10")); args = parser.parse_args()
    if RUN_FULL_TRAINING: raise RuntimeError("V4 Stage A cannot run full training")
    required = {
        "architecture_equivalence.json": "v4_architecture_equivalence", "parameter_equivalence.json": "v4_parameter_equivalence",
        "gpu_device_check.json": "v4_gpu_device_check", "fixed_batch_check.json": "v4_fixed_batch_check",
        "shared_w0_check.json": "v4_shared_w0_check", "phase2_shared_body_w0_check.json": "v4_phase2_shared_body_w0",
        "resync_precheck.json": "v4_resync_precheck",
    }
    for filename, key in required.items():
        report = json.loads((RESULTS / "preflight" / filename).read_text())
        if report.get(key) != "VALID" or report.get("config_sha256") != config_hash(): raise RuntimeError(f"Invalid V4 preflight: {filename}")
    train, _ = require_local_cifar10(args.data_root); train_indices, _ = stratified_split(train.targets)
    _, images, labels = canonical_images(train, train_indices[:32]); runner = Runner(images, labels)
    summaries, equivalence, repeatability = [], [], []
    for seed in SEEDS:
        case = f"seed{seed}_initial_shared"; state = create_initial_state(seed)
        sync1, pk, pp = runner.pair(state, False); sync2, tk, tp = runner.pair(state, True)
        equivalence += _validation_rows(case, "plain_vs_trace", "keras", pk, tk)
        equivalence += _validation_rows(case, "plain_vs_trace", "pytorch", pp, tp)
        if not all(row["exact_equal"] for row in equivalence if row["case_id"] == case): raise RuntimeError("V4 trace equivalence failed")
        frows = []
        for order, stage in enumerate(tk["forward"]):
            row = {"seed": seed, "stage_order": order, "stage": stage}; row.update(_metric(tk["forward"][stage], tp["forward"][stage])); frows.append(row)
        loss_row = {"seed": seed, "stage_order": len(frows), "stage": "loss"}; loss_row.update(_metric(np.asarray([tk["loss"]], np.float32), np.asarray([tp["loss"]], np.float32), False)); frows.append(loss_row)
        arows = []
        for order, stage in enumerate(ACTIVATION_STAGES):
            row = {"seed": seed, "backward_order": order, "stage": stage}; row.update(_metric(tk["activation"][stage], tp["activation"][stage])); arows.append(row)
        prows = []
        for order, name in enumerate(TRAINABLE_NAMES):
            row = {"seed": seed, "parameter_order": order, "parameter": name}; row.update(_metric(tk["parameter_gradients"][name], tp["parameter_gradients"][name], False)); prows.append(row)
        urows = []
        for order, name in enumerate(TRAINABLE_NAMES):
            urows.append({"seed": seed, "parameter_order": order, "parameter": name,
                          "gradient_relative_l2": compare(tk["parameter_gradients"][name], tp["parameter_gradients"][name])["relative_l2_error"],
                          "m_relative_l2": compare(tk["adam_m"][name], tp["adam_m"][name])["relative_l2_error"],
                          "v_relative_l2": compare(tk["adam_v"][name], tp["adam_v"][name])["relative_l2_error"],
                          "update_relative_l2": compare(tk["updates"][name], tp["updates"][name])["relative_l2_error"],
                          "post_weight_relative_l2": compare(tk["post"][name], tp["post"][name])["relative_l2_error"]})
        bnrows = _bn_rows(seed, frows, prows, urows, tk, tp)
        del tk["forward"], tk["activation"], tp["forward"], tp["activation"]
        runner.kmodel.last_trace = {}; runner.pmodel.last_trace = {}; gc.collect(); runner.torch.mps.empty_cache()
        _, rk, rp = runner.pair(state, True); del rk["forward"], rk["activation"], rp["forward"], rp["activation"]
        repeatability += _validation_rows(case, "trace_run1_vs_run2", "keras", tk, rk)
        repeatability += _validation_rows(case, "trace_run1_vs_run2", "pytorch", tp, rp)
        prefix = []
        for row in frows:
            if row["stage"] == "loss" or float(row["max_abs_diff"]) > 0: break
            prefix.append(row["stage"])
        largest_f, largest_a = max(frows, key=lambda r: float(r["relative_l2"])), max(arows, key=lambda r: float(r["relative_l2"]))
        largest_p, largest_u = max(prows, key=lambda r: float(r["relative_l2"])), max(urows, key=lambda r: float(r["update_relative_l2"]))
        summaries.append({"seed": seed, "checkpoint": "initial", "anchor": "shared", "full_state_sync_exact": sync1 and sync2,
                          "first_nonzero_forward_stage": _first(frows), "exact_forward_prefix_length": len(prefix),
                          "largest_forward_stage": largest_f["stage"], "largest_forward_rel_l2": largest_f["relative_l2"],
                          "logits_rel_l2": next(r["relative_l2"] for r in frows if r["stage"] == "logits"),
                          "loss_abs_diff": abs(tk["loss"] - tp["loss"]), "first_nonzero_backward_stage": _first(arows),
                          "largest_backward_stage": largest_a["stage"], "largest_backward_rel_l2": largest_a["relative_l2"],
                          "largest_parameter_gradient_group": largest_p["parameter"], "largest_parameter_gradient_rel_l2": largest_p["relative_l2"],
                          "largest_update_group": largest_u["parameter"], "largest_update_rel_l2": largest_u["update_relative_l2"],
                          "global_gradient_rel_l2": compare_group(tk["parameter_gradients"], tp["parameter_gradients"], TRAINABLE_NAMES)["relative_l2_error"],
                          "global_update_rel_l2": compare_group(tk["updates"], tp["updates"], TRAINABLE_NAMES)["relative_l2_error"],
                          "global_post_weight_rel_l2": compare_group(tk["post"], tp["post"], TRAINABLE_NAMES)["relative_l2_error"], "nan_inf": False})
        write_csv(RESULTS / "forward" / f"seed{seed}_initial_forward.csv", frows)
        write_csv(RESULTS / "backward" / f"seed{seed}_initial_activation_gradients.csv", arows)
        write_csv(RESULTS / "backward" / f"seed{seed}_initial_parameter_gradients.csv", prows)
        write_csv(RESULTS / "bn" / f"seed{seed}_initial_bn_trace.csv", bnrows)
        write_csv(RESULTS / "optimizer" / f"seed{seed}_initial_updates.csv", urows)
    exact_equiv = all(row["exact_equal"] for row in equivalence); repeat_exact = all(row["exact_equal"] for row in repeatability)
    repeat_close = all(row["within_qualified_tolerance"] for row in repeatability)
    repeat_status = "VALID" if repeat_exact else "QUALIFIED" if repeat_close else "INVALID"
    write_csv(RESULTS / "summaries/stage_a_case_summary.csv", summaries); write_csv(RESULTS / "preflight/trace_equivalence_rows.csv", equivalence); write_csv(RESULTS / "preflight/repeatability_rows.csv", repeatability)
    write_json(RESULTS / "preflight/trace_equivalence.json", {"v4_trace_equivalence": "VALID" if exact_equiv else "INVALID", "comparisons": len(equivalence), "config_sha256": config_hash()})
    write_json(RESULTS / "preflight/repeatability.json", {"v4_gpu_repeatability": repeat_status, "comparisons": len(repeatability), "config_sha256": config_hash()})
    print(f"V4_TRACE_EQUIVALENCE = {'VALID' if exact_equiv else 'INVALID'}"); print(f"V4_GPU_REPEATABILITY = {repeat_status}"); print("V4_SELECTED_CASES_COMPLETE = TRUE")


if __name__ == "__main__": main()
