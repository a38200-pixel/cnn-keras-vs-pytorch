"""Run the three V3 synchronized initial-state GPU one-step traces."""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP07 = ROOT / "experiments/07_multistep_state_resynchronization"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP07))
sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from common.reference_adam import CommonAdam
from common.state_resynchronization import compare_group, states_exact
from common.trace_utils import assert_finite, compare, write_csv, write_json
from probe_utils import fixed_probe_batch
from bnfree_keras import build_keras_bnfree
from bnfree_pytorch import build_torch_bnfree
from bnfree_state import (
    TRAINABLE_NAMES, export_keras_state, export_torch_state, keras_gradients,
    load_keras_state, load_torch_state, phase2_shared_state, torch_gradients,
    trainable_values,
)
from v3_config import (
    REPEATABILITY_ATOL, REPEATABILITY_RTOL, RESULTS, RUN_FULL_TRAINING,
    cases, config_hash,
)

FORWARD_STAGES = ["input"] + [
    name for index in range(1, 5)
    for name in (f"conv{index}", f"relu{index}", f"pool{index}")
] + ["gap", "fc128.preactivation", "fc128.relu", "logits"]

ACTIVATION_STAGES = [
    "logits", "fc128.relu", "fc128.preactivation", "gap",
    *(name for index in range(4, 0, -1)
      for name in (f"pool{index}", f"relu{index}", f"conv{index}")),
]


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy()).copy()
    return np.asarray(value).copy()


def _canonical(value, framework: str) -> np.ndarray:
    array = _numpy(value)
    if framework == "pytorch" and array.ndim == 4:
        return array.transpose(0, 2, 3, 1).copy()
    return array


def _metric_row(keras_value, torch_value, *, activation_layout: bool = True) -> dict[str, object]:
    a = _canonical(keras_value, "keras")
    b = _canonical(torch_value, "pytorch" if activation_layout else "canonical")
    stats = compare(a, b)
    return {
        "shape": json.dumps(a.shape), "keras_dtype": str(a.dtype), "pytorch_dtype": str(b.dtype),
        "exact_equal": bool(np.array_equal(a, b)),
        "keras_norm": stats["keras_norm"], "pytorch_norm": stats["pytorch_norm"],
        "max_abs_diff": stats["max_abs_diff"], "mean_abs_diff": stats["mean_abs_diff"],
        "relative_l2": stats["relative_l2_error"],
        "cosine_similarity": stats["cosine_similarity"],
        "sign_mismatch_fraction": stats["sign_mismatch_ratio"],
    }


class TraceRunner:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        import tensorflow as tf
        import torch

        self.tf, self.torch = tf, torch
        self.device = torch.device("mps")
        if not tf.config.list_physical_devices("GPU") or not torch.backends.mps.is_available():
            raise RuntimeError("V3 requires TensorFlow Metal GPU and PyTorch MPS")
        self.images = np.ascontiguousarray(images.astype(np.float32, copy=False))
        self.labels = np.asarray(labels, dtype=np.int32)
        with tf.device("/GPU:0"):
            self.kx = tf.convert_to_tensor(self.images, tf.float32)
            self.ky = tf.convert_to_tensor(self.labels, tf.int32)
            self.keras_model = build_keras_bnfree()
        self.px = torch.from_numpy(self.images.transpose(0, 3, 1, 2).copy()).to(self.device)
        self.py = torch.from_numpy(self.labels.astype(np.int64)).to(self.device)
        self.torch_model = build_torch_bnfree().to(self.device)
        self.torch_model.gradient_trace_names = set(ACTIVATION_STAGES)
        restored = self.px.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.input_exact = bool(np.array_equal(self.images, restored))
        self.label_exact = bool(np.array_equal(self.labels, self.py.detach().cpu().numpy().astype(np.int32)))
        if not self.input_exact or not self.label_exact:
            raise RuntimeError("V3 canonical input/label conversion mismatch")

    def validate_resync(self, state: dict[str, np.ndarray]) -> dict[str, object]:
        load_keras_state(self.keras_model, state)
        load_torch_state(self.torch_model, state)
        keras_state = export_keras_state(self.keras_model)
        torch_state = export_torch_state(self.torch_model)
        k_adam = CommonAdam(trainable_values(state))
        p_adam = CommonAdam(trainable_values(state))
        checks = {
            "model_state_exact": states_exact(keras_state, torch_state),
            "adam_m_exact": states_exact(k_adam.m, p_adam.m),
            "adam_v_exact": states_exact(k_adam.v, p_adam.v),
            "adam_step_exact": k_adam.step == p_adam.step == 0,
            "input_exact": self.input_exact, "label_exact": self.label_exact,
        }
        checks["all_exact"] = all(checks.values())
        if not checks["all_exact"]:
            raise RuntimeError(f"V3 full-state synchronization failed: {checks}")
        return checks

    def _keras_step(self, state, traced: bool):
        tf = self.tf
        load_keras_state(self.keras_model, state)
        adam = CommonAdam(trainable_values(state))
        with tf.device("/GPU:0"):
            with tf.GradientTape() as tape:
                logits = self.keras_model(self.kx, training=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    self.ky, logits, from_logits=True,
                ))
            trace = dict(self.keras_model.last_trace)
            if list(trace) != FORWARD_STAGES:
                raise RuntimeError("Keras V3 forward stage order mismatch")
            if traced:
                activation_targets = [trace[name] for name in ACTIVATION_STAGES]
                targets = activation_targets + list(self.keras_model.trainable_variables)
                gradients = tape.gradient(loss, targets)
                activation_values = gradients[:len(activation_targets)]
                parameter_values = gradients[len(activation_targets):]
                if any(value is None for value in gradients):
                    raise RuntimeError("Missing Keras V3 traced gradient")
                activation_gradients = dict(zip(ACTIVATION_STAGES, activation_values))
            else:
                parameter_values = tape.gradient(loss, self.keras_model.trainable_variables)
                if any(value is None for value in parameter_values):
                    raise RuntimeError("Missing Keras V3 plain gradient")
                activation_gradients = None
            parameter_gradients = keras_gradients(self.keras_model, parameter_values)
        before_update = export_keras_state(self.keras_model)
        updated, updates = adam.update(trainable_values(before_update), parameter_gradients)
        load_keras_state(self.keras_model, updated)
        result = {
            "logits": _numpy(logits), "loss": float(loss.numpy()),
            "parameter_gradients": parameter_gradients, "updates": updates,
            "adam_m": {name: value.copy() for name, value in adam.m.items()},
            "adam_v": {name: value.copy() for name, value in adam.v.items()},
            "post": export_keras_state(self.keras_model),
        }
        if traced:
            result["forward_raw"] = trace
            result["activation_raw"] = activation_gradients
        return result

    def _torch_step(self, state, traced: bool):
        torch = self.torch
        load_torch_state(self.torch_model, state)
        adam = CommonAdam(trainable_values(state))
        self.torch_model.train()
        self.torch_model.trace_gradients = traced
        self.torch_model.zero_grad(set_to_none=True)
        logits = self.torch_model(self.px)
        loss = torch.nn.functional.cross_entropy(logits, self.py)
        loss.backward()
        trace = dict(self.torch_model.last_trace)
        if list(trace) != FORWARD_STAGES:
            raise RuntimeError("PyTorch V3 forward stage order mismatch")
        parameter_gradients = torch_gradients(self.torch_model)
        activation_gradients = None
        if traced:
            activation_gradients = {}
            for name in ACTIVATION_STAGES:
                value = trace[name].grad
                if value is None:
                    raise RuntimeError(f"Missing PyTorch V3 activation gradient: {name}")
                activation_gradients[name] = value
        before_update = export_torch_state(self.torch_model)
        updated, updates = adam.update(trainable_values(before_update), parameter_gradients)
        load_torch_state(self.torch_model, updated)
        result = {
            "logits": _numpy(logits), "loss": float(loss.detach().cpu()),
            "parameter_gradients": parameter_gradients, "updates": updates,
            "adam_m": {name: value.copy() for name, value in adam.m.items()},
            "adam_v": {name: value.copy() for name, value in adam.v.items()},
            "post": export_torch_state(self.torch_model),
        }
        if traced:
            result["forward_raw"] = trace
            result["activation_raw"] = activation_gradients
        return result

    def run_pair(self, state, traced: bool):
        exact = self.validate_resync(state)
        return exact, self._keras_step(state, traced), self._torch_step(state, traced)


def _comparison_arrays(values: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "logits": np.asarray(values["logits"]),
        "loss": np.asarray([values["loss"]], dtype=np.float32),
        "global_gradient": np.concatenate([values["parameter_gradients"][name].ravel() for name in TRAINABLE_NAMES]),
        "global_update": np.concatenate([values["updates"][name].ravel() for name in TRAINABLE_NAMES]),
        "post_weight": np.concatenate([values["post"][name].ravel() for name in TRAINABLE_NAMES]),
    }


def _comparison_rows(case_id: str, comparison_name: str, framework: str, left, right):
    rows = []
    for metric, a in _comparison_arrays(left).items():
        b = _comparison_arrays(right)[metric]
        stats = compare(a, b)
        rows.append({
            "case_id": case_id, "comparison": comparison_name, "framework": framework,
            "metric": metric, "exact_equal": bool(np.array_equal(a, b)),
            "within_qualified_tolerance": bool(np.allclose(
                a, b, rtol=REPEATABILITY_RTOL, atol=REPEATABILITY_ATOL, equal_nan=False,
            )),
            "max_abs_diff": stats["max_abs_diff"], "mean_abs_diff": stats["mean_abs_diff"],
            "relative_l2": stats["relative_l2_error"],
        })
    return rows


def _forward_rows(case, keras, pytorch):
    ktrace, ptrace = keras["forward_raw"], pytorch["forward_raw"]
    if list(ktrace) != list(ptrace) or list(ktrace) != FORWARD_STAGES:
        raise RuntimeError("V3 forward semantic order mismatch")
    rows = []
    for order, stage in enumerate(FORWARD_STAGES):
        row = {"seed": case["seed"], "stage_order": order, "stage": stage}
        row.update(_metric_row(ktrace[stage], ptrace[stage]))
        rows.append(row)
    loss_row = {"seed": case["seed"], "stage_order": len(rows), "stage": "loss"}
    loss_row.update(_metric_row(
        np.asarray([keras["loss"]], np.float32), np.asarray([pytorch["loss"]], np.float32),
    ))
    rows.append(loss_row)
    return rows


def _activation_rows(case, keras, pytorch):
    rows = []
    for order, stage in enumerate(ACTIVATION_STAGES):
        row = {"seed": case["seed"], "backward_order": order, "stage": stage}
        row.update(_metric_row(keras["activation_raw"][stage], pytorch["activation_raw"][stage]))
        rows.append(row)
    return rows


def _parameter_rows(case, keras, pytorch):
    rows = []
    for order, name in enumerate(TRAINABLE_NAMES):
        row = {"seed": case["seed"], "parameter_order": order, "parameter": name}
        row.update(_metric_row(
            keras["parameter_gradients"][name], pytorch["parameter_gradients"][name],
            activation_layout=False,
        ))
        rows.append(row)
    return rows


def _update_rows(case, keras, pytorch):
    rows = []
    for order, name in enumerate(TRAINABLE_NAMES):
        gradient = compare(keras["parameter_gradients"][name], pytorch["parameter_gradients"][name])
        m = compare(keras["adam_m"][name], pytorch["adam_m"][name])
        v = compare(keras["adam_v"][name], pytorch["adam_v"][name])
        update = compare(keras["updates"][name], pytorch["updates"][name])
        post = compare(keras["post"][name], pytorch["post"][name])
        rows.append({
            "seed": case["seed"], "parameter_order": order, "parameter": name,
            "gradient_relative_l2": gradient["relative_l2_error"],
            "gradient_keras_norm": gradient["keras_norm"],
            "gradient_pytorch_norm": gradient["pytorch_norm"],
            "m_relative_l2": m["relative_l2_error"], "v_relative_l2": v["relative_l2_error"],
            "update_relative_l2": update["relative_l2_error"],
            "post_weight_relative_l2": post["relative_l2_error"],
        })
    return rows


def _first_nonzero(rows, key="stage"):
    return next((row[key] for row in rows if float(row["max_abs_diff"]) > 0), "none")


def _exact_prefix(forward_rows):
    stages = []
    for row in forward_rows:
        if row["stage"] == "loss" or float(row["max_abs_diff"]) > 0:
            break
        stages.append(str(row["stage"]))
    return stages


def _finite_result(case_id, branch, result):
    for key in ("parameter_gradients", "updates", "adam_m", "adam_v", "post"):
        assert_finite(result[key], f"{case_id} {branch} {key}")
    assert_finite({"logits": result["logits"]}, f"{case_id} {branch} logits")
    if not np.isfinite(result["loss"]):
        raise RuntimeError(f"NaN/Inf in {case_id} {branch} loss")


def run() -> dict[str, object]:
    if RUN_FULL_TRAINING:
        raise RuntimeError("V3 safety guard violated")
    required = {
        "architecture_equivalence.json": ("v3_architecture_equivalence", "VALID"),
        "parameter_equivalence.json": ("v3_parameter_equivalence", "VALID"),
        "no_batchnorm_check.json": ("v3_no_batchnorm_check", "VALID"),
        "gpu_device_check.json": ("v3_gpu_device_check", "VALID"),
        "fixed_batch_check.json": ("v3_fixed_batch_check", "VALID"),
        "shared_w0_check.json": ("v3_shared_w0_check", "VALID"),
        "resync_precheck.json": ("v3_resync_precheck", "VALID"),
    }
    for filename, (key, expected) in required.items():
        path = RESULTS / "preflight" / filename
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if report.get(key) != expected or report.get("config_sha256") != config_hash():
            raise RuntimeError(f"V3 preflight is not current and VALID: {filename}")

    images, labels, _ = fixed_probe_batch(write_manifest=False)
    runner = TraceRunner(images, labels)
    summaries, equivalence_rows, repeatability_rows = [], [], []

    for case in cases():
        case_id, seed = str(case["case_id"]), int(case["seed"])
        state = phase2_shared_state(seed)
        plain_exact, plain_k, plain_p = runner.run_pair(state, traced=False)
        trace_exact, trace1_k, trace1_p = runner.run_pair(state, traced=True)
        for branch, values in (
            ("plain_keras", plain_k), ("plain_pytorch", plain_p),
            ("trace1_keras", trace1_k), ("trace1_pytorch", trace1_p),
        ):
            _finite_result(case_id, branch, values)

        equivalence_rows.extend(_comparison_rows(case_id, "plain_vs_trace", "keras", plain_k, trace1_k))
        equivalence_rows.extend(_comparison_rows(case_id, "plain_vs_trace", "pytorch", plain_p, trace1_p))
        case_equivalent = all(row["exact_equal"] for row in equivalence_rows if row["case_id"] == case_id)
        if not case_equivalent:
            raise RuntimeError(f"V3 trace instrumentation changed one-step values: {case_id}")

        forward_rows = _forward_rows(case, trace1_k, trace1_p)
        activation_rows = _activation_rows(case, trace1_k, trace1_p)
        parameter_rows = _parameter_rows(case, trace1_k, trace1_p)
        update_rows = _update_rows(case, trace1_k, trace1_p)

        del trace1_k["forward_raw"], trace1_k["activation_raw"]
        del trace1_p["forward_raw"], trace1_p["activation_raw"]
        runner.keras_model.last_trace = {}
        runner.torch_model.last_trace = {}
        gc.collect()
        runner.torch.mps.empty_cache()

        _, trace2_k, trace2_p = runner.run_pair(state, traced=True)
        _finite_result(case_id, "trace2_keras", trace2_k)
        _finite_result(case_id, "trace2_pytorch", trace2_p)
        del trace2_k["forward_raw"], trace2_k["activation_raw"]
        del trace2_p["forward_raw"], trace2_p["activation_raw"]
        runner.keras_model.last_trace = {}
        runner.torch_model.last_trace = {}
        repeatability_rows.extend(_comparison_rows(
            case_id, "trace_run1_vs_run2", "keras", trace1_k, trace2_k,
        ))
        repeatability_rows.extend(_comparison_rows(
            case_id, "trace_run1_vs_run2", "pytorch", trace1_p, trace2_p,
        ))

        largest_forward = max(forward_rows, key=lambda row: float(row["relative_l2"]))
        largest_backward = max(activation_rows, key=lambda row: float(row["relative_l2"]))
        largest_parameter = max(parameter_rows, key=lambda row: float(row["relative_l2"]))
        largest_update = max(update_rows, key=lambda row: float(row["update_relative_l2"]))
        global_gradient = compare_group(
            trace1_k["parameter_gradients"], trace1_p["parameter_gradients"], TRAINABLE_NAMES,
        )
        global_update = compare_group(trace1_k["updates"], trace1_p["updates"], TRAINABLE_NAMES)
        global_post = compare_group(trace1_k["post"], trace1_p["post"], TRAINABLE_NAMES)
        logits_row = next(row for row in forward_rows if row["stage"] == "logits")
        prefix = _exact_prefix(forward_rows)
        summaries.append({
            "case_id": case_id, "seed": seed, "checkpoint": "initial", "anchor": "shared",
            "full_state_sync_exact": plain_exact["all_exact"] and trace_exact["all_exact"],
            "first_nonzero_forward_stage": _first_nonzero(forward_rows),
            "exact_forward_prefix_length": len(prefix),
            "exact_forward_prefix_stages": json.dumps(prefix),
            "largest_forward_stage": largest_forward["stage"],
            "largest_forward_rel_l2": largest_forward["relative_l2"],
            "logits_rel_l2": logits_row["relative_l2"],
            "loss_abs_diff": abs(trace1_k["loss"] - trace1_p["loss"]),
            "first_nonzero_backward_stage": _first_nonzero(activation_rows),
            "largest_backward_stage": largest_backward["stage"],
            "largest_backward_rel_l2": largest_backward["relative_l2"],
            "largest_parameter_gradient_group": largest_parameter["parameter"],
            "largest_parameter_gradient_rel_l2": largest_parameter["relative_l2"],
            "largest_update_group": largest_update["parameter"],
            "largest_update_rel_l2": largest_update["update_relative_l2"],
            "global_gradient_rel_l2": global_gradient["relative_l2_error"],
            "global_update_rel_l2": global_update["relative_l2_error"],
            "global_post_weight_rel_l2": global_post["relative_l2_error"],
            "keras_repeatability": "pending", "pytorch_repeatability": "pending",
            "nan_inf": False,
        })
        write_csv(RESULTS / "forward" / f"seed{seed}_initial_forward.csv", forward_rows)
        write_csv(RESULTS / "backward" / f"seed{seed}_initial_activation_gradients.csv", activation_rows)
        write_csv(RESULTS / "backward" / f"seed{seed}_initial_parameter_gradients.csv", parameter_rows)
        write_csv(RESULTS / "optimizer" / f"seed{seed}_initial_updates.csv", update_rows)
        print(
            f"V3 seed={seed} first={summaries[-1]['first_nonzero_forward_stage']} "
            f"prefix={len(prefix)} forward_max={float(summaries[-1]['largest_forward_rel_l2']):.6g} "
            f"trace_equivalent={case_equivalent}", flush=True,
        )
        del plain_k, plain_p, trace1_k, trace1_p, trace2_k, trace2_p
        runner.keras_model.last_trace = {}
        runner.torch_model.last_trace = {}
        gc.collect()
        runner.torch.mps.empty_cache()

    equivalence_valid = all(row["exact_equal"] for row in equivalence_rows)
    repeat_exact = all(row["exact_equal"] for row in repeatability_rows)
    repeat_close = all(row["within_qualified_tolerance"] for row in repeatability_rows)
    repeat_status = "VALID" if repeat_exact else ("QUALIFIED" if repeat_close else "INVALID")
    for summary in summaries:
        summary["keras_repeatability"] = repeat_status
        summary["pytorch_repeatability"] = repeat_status
    write_csv(RESULTS / "summaries/case_summary.csv", summaries)
    write_csv(RESULTS / "preflight/trace_equivalence_rows.csv", equivalence_rows)
    write_csv(RESULTS / "preflight/repeatability_rows.csv", repeatability_rows)
    write_json(RESULTS / "preflight/trace_equivalence.json", {
        "v3_trace_equivalence": "VALID" if equivalence_valid else "INVALID",
        "all_exact": equivalence_valid, "comparisons": len(equivalence_rows),
        "config_sha256": config_hash(), "new_full_training_executed": False,
    })
    write_json(RESULTS / "preflight/repeatability.json", {
        "v3_gpu_repeatability": repeat_status, "all_exact": repeat_exact,
        "all_within_qualified_tolerance": repeat_close,
        "qualified_atol": REPEATABILITY_ATOL, "qualified_rtol": REPEATABILITY_RTOL,
        "comparisons": len(repeatability_rows), "cases": len(summaries),
        "reason": (
            "all compared arrays were exact" if repeat_exact else
            "non-zero differences remained within the predeclared tolerance" if repeat_close else
            "at least one within-framework comparison exceeded the predeclared tolerance"
        ),
        "config_sha256": config_hash(), "new_full_training_executed": False,
    })
    print(f"V3_TRACE_EQUIVALENCE = {'VALID' if equivalence_valid else 'INVALID'}")
    print(f"V3_GPU_REPEATABILITY = {repeat_status}")
    print(f"V3_SELECTED_CASES_COMPLETE = {str(len(summaries) == 3).upper()}")
    print("NAN_INF_FOUND = FALSE")
    print("NEW_FULL_TRAINING_EXECUTED = FALSE")
    return {"summaries": summaries, "repeatability": repeat_status}


if __name__ == "__main__":
    run()
