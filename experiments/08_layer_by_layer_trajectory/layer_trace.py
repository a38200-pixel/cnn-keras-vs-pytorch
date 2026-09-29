"""Run nine checkpoint-local, one-step, layer-by-layer traces for Experiment 08."""

from __future__ import annotations

import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP07 = ROOT / "experiments/07_multistep_state_resynchronization"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP07))

from common.common_adam_control import (
    TRAINABLE_NAMES, keras_gradients, load_keras_trainable, load_torch_trainable,
    torch_gradients, trainable_values,
)
from common.controlled_initialization import (
    export_keras_weights, export_torch_weights, load_keras_weights, load_torch_weights,
)
from common.state_resynchronization import (
    BN_MEAN_NAMES, BN_VARIANCE_NAMES, adam_from_checkpoint, adam_m_state,
    adam_v_state, compare_group,
)
from common.trace_utils import assert_finite, compare, write_csv
from layer_config import RESULTS, UPDATE_RATIO_THRESHOLD, config_hash, selected_cases
from probe_utils import ProbeRunner, fixed_probe_batch, load_checkpoint


STANDARD_FORWARD_STAGES = [
    *(name for index in range(1, 5) for name in (
        f"conv{index}", f"bn{index}", f"relu{index}", f"pool{index}",
    )),
    "gap", "fc128", "fc128_relu", "logits",
]

ACTIVATION_GRADIENT_STAGES = [
    "logits", "fc128_relu", "fc128", "gap",
    *(name for index in range(4, 0, -1) for name in (
        f"pool{index}", f"relu{index}", f"bn{index}", f"conv{index}",
    )),
]


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy()).copy()
    return np.asarray(value).copy()


def _canonical_torch(name: str, value) -> np.ndarray:
    array = _numpy(value)
    if name == "gap" and array.ndim == 4:
        return array[:, :, 0, 0]
    if array.ndim == 4:
        return array.transpose(0, 2, 3, 1).copy()
    return array


def _metric_row(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    stats = compare(a, b)
    return {
        "shape": stats["shape"],
        "keras_dtype": str(a.dtype),
        "pytorch_dtype": str(b.dtype),
        "exact_equal": bool(np.array_equal(a, b)),
        "keras_norm": stats["keras_norm"],
        "pytorch_norm": stats["pytorch_norm"],
        "max_abs_diff": stats["max_abs_diff"],
        "mean_abs_diff": stats["mean_abs_diff"],
        "relative_l2": stats["relative_l2_error"],
        "cosine_similarity": stats["cosine_similarity"],
        "sign_mismatch_fraction": stats["sign_mismatch_ratio"],
    }


class LayerTraceRunner(ProbeRunner):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        super().__init__(images, labels)
        keras_outputs = []
        for name in STANDARD_FORWARD_STAGES:
            layer_name = "dense128_relu" if name == "fc128_relu" else name
            keras_outputs.append(self.keras_model.get_layer(layer_name).output)
        self.keras_trace_model = self.tf.keras.Model(self.keras_model.input, keras_outputs)
        self.torch_activations: dict[str, object] = {}
        self._hook_handles = []
        for name in STANDARD_FORWARD_STAGES:
            module_name = "dense128_relu" if name == "fc128_relu" else name
            module = getattr(self.torch_model, module_name)
            self._hook_handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            output.retain_grad()
            self.torch_activations[name] = output
        return hook

    def _keras_one_step(self, model_state, optimizer_state, step):
        tf = self.tf
        load_keras_weights(self.keras_model, model_state)
        adam = adam_from_checkpoint(model_state, optimizer_state, step)
        with tf.GradientTape() as tape:
            values = self.keras_trace_model(self.kx, training=True)
            activations = dict(zip(STANDARD_FORWARD_STAGES, values))
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    self.ky, activations["logits"], from_logits=True,
                )
            )
        activation_targets = [activations[name] for name in ACTIVATION_GRADIENT_STAGES]
        targets = activation_targets + list(self.keras_model.trainable_variables)
        gradients = tape.gradient(loss, targets)
        activation_gradient_values = gradients[:len(activation_targets)]
        parameter_gradient_values = gradients[len(activation_targets):]
        if any(value is None for value in gradients):
            missing = [
                name for name, value in zip(
                    ACTIVATION_GRADIENT_STAGES
                    + [variable.path for variable in self.keras_model.trainable_variables], gradients,
                ) if value is None
            ]
            raise RuntimeError(f"Missing Keras traced gradients: {missing}")
        parameter_gradients = keras_gradients(self.keras_model, parameter_gradient_values)

        forward = {name: _numpy(value) for name, value in activations.items()}
        forward["input"] = self.images.copy()
        for index in range(1, 5):
            trace = self.keras_model.get_layer(f"bn{index}").last_trace
            for field, value in trace.items():
                forward[f"bn{index}_{field}"] = _numpy(value)
        forward["loss"] = np.asarray([float(loss.numpy())], dtype=np.float32)
        activation_gradients = {
            name: _numpy(value)
            for name, value in zip(ACTIVATION_GRADIENT_STAGES, activation_gradient_values)
        }

        after_forward = export_keras_weights(self.keras_model)
        updated, updates = adam.update(trainable_values(after_forward), parameter_gradients)
        load_keras_trainable(self.keras_model, updated)
        post = export_keras_weights(self.keras_model)
        return {
            "loss": float(loss.numpy()), "forward": forward,
            "activation_gradients": activation_gradients,
            "parameter_gradients": parameter_gradients, "updates": updates,
            "adam_m": adam_m_state(adam), "adam_v": adam_v_state(adam),
            "post": post,
        }

    def _torch_one_step(self, model_state, optimizer_state, step):
        torch = self.torch
        load_torch_weights(self.torch_model, model_state)
        adam = adam_from_checkpoint(model_state, optimizer_state, step)
        self.torch_model.train()
        self.torch_model.zero_grad(set_to_none=True)
        self.torch_activations = {}
        logits = self.torch_model(self.px)
        loss = torch.nn.functional.cross_entropy(logits, self.py)
        loss.backward()
        if set(self.torch_activations) != set(STANDARD_FORWARD_STAGES):
            raise RuntimeError("PyTorch activation hook coverage mismatch")
        parameter_gradients = torch_gradients(self.torch_model)

        forward = {
            name: _canonical_torch(name, value)
            for name, value in self.torch_activations.items()
        }
        forward["input"] = self.images.copy()
        for index in range(1, 5):
            trace = getattr(self.torch_model, f"bn{index}").last_trace
            for field, value in trace.items():
                forward[f"bn{index}_{field}"] = _canonical_torch(field, value)
        forward["loss"] = np.asarray([float(loss.detach().cpu())], dtype=np.float32)
        activation_gradients = {
            name: _canonical_torch(name, self.torch_activations[name].grad)
            for name in ACTIVATION_GRADIENT_STAGES
        }
        if any(self.torch_activations[name].grad is None for name in ACTIVATION_GRADIENT_STAGES):
            raise RuntimeError("Missing PyTorch traced activation gradients")

        after_forward = export_torch_weights(self.torch_model)
        updated, updates = adam.update(trainable_values(after_forward), parameter_gradients)
        load_torch_trainable(self.torch_model, updated)
        post = export_torch_weights(self.torch_model)
        return {
            "loss": float(loss.detach().cpu()), "forward": forward,
            "activation_gradients": activation_gradients,
            "parameter_gradients": parameter_gradients, "updates": updates,
            "adam_m": adam_m_state(adam), "adam_v": adam_v_state(adam),
            "post": post,
        }

    def run_trace(self, model_state, optimizer_state, step):
        exact = self.validate_resync(model_state, optimizer_state, step)
        keras = self._keras_one_step(model_state, optimizer_state, step)
        pytorch = self._torch_one_step(model_state, optimizer_state, step)
        return exact, keras, pytorch


def _forward_order() -> list[str]:
    result = ["input"]
    for index in range(1, 5):
        result.extend([
            f"conv{index}", f"bn{index}_input", f"bn{index}_batch_mean",
            f"bn{index}_batch_variance", f"bn{index}_x_hat",
            f"bn{index}_output", f"relu{index}", f"pool{index}",
        ])
    result.extend(["gap", "fc128", "fc128_relu", "logits", "loss"])
    return result


def _build_forward_rows(case, keras, pytorch):
    rows = []
    previous = None
    for order, stage in enumerate(_forward_order()):
        row = {"case_id": case["case_id"], "stage_order": order, "stage": stage}
        row.update(_metric_row(keras["forward"][stage], pytorch["forward"][stage]))
        current = float(row["relative_l2"])
        row["previous_relative_l2"] = "" if previous is None else previous
        row["amplification_ratio"] = (
            "" if previous is None or previous <= UPDATE_RATIO_THRESHOLD else current / previous
        )
        row["shape_changed_from_previous"] = (
            "" if not rows else row["shape"] != rows[-1]["shape"]
        )
        rows.append(row)
        previous = current
    return rows


def _build_activation_gradient_rows(case, keras, pytorch):
    rows = []
    for order, stage in enumerate(ACTIVATION_GRADIENT_STAGES):
        row = {"case_id": case["case_id"], "backward_order": order, "stage": stage}
        row.update(_metric_row(
            keras["activation_gradients"][stage], pytorch["activation_gradients"][stage],
        ))
        rows.append(row)
    return rows


def _build_parameter_rows(case, keras, pytorch):
    rows = []
    for order, name in enumerate(TRAINABLE_NAMES):
        row = {"case_id": case["case_id"], "parameter_order": order, "parameter": name}
        row.update(_metric_row(
            keras["parameter_gradients"][name], pytorch["parameter_gradients"][name],
        ))
        rows.append(row)
    return rows


def _build_update_rows(case, keras, pytorch):
    rows = []
    for order, name in enumerate(TRAINABLE_NAMES):
        gradient = compare(keras["parameter_gradients"][name], pytorch["parameter_gradients"][name])
        m = compare(keras["adam_m"][name], pytorch["adam_m"][name])
        v = compare(keras["adam_v"][name], pytorch["adam_v"][name])
        update = compare(keras["updates"][name], pytorch["updates"][name])
        post = compare(keras["post"][name], pytorch["post"][name])
        grad_rel = float(gradient["relative_l2_error"])
        rows.append({
            "case_id": case["case_id"], "parameter_order": order, "parameter": name,
            "gradient_relative_l2": grad_rel,
            "gradient_keras_norm": gradient["keras_norm"],
            "gradient_pytorch_norm": gradient["pytorch_norm"],
            "m_relative_l2": m["relative_l2_error"],
            "v_relative_l2": v["relative_l2_error"],
            "update_relative_l2": update["relative_l2_error"],
            "post_weight_relative_l2": post["relative_l2_error"],
            "update_to_gradient_ratio": (
                "" if grad_rel <= UPDATE_RATIO_THRESHOLD
                else float(update["relative_l2_error"]) / grad_rel
            ),
            "ratio_denominator_threshold": UPDATE_RATIO_THRESHOLD,
        })
    return rows


def _by_name(rows, key):
    return {row[key]: row for row in rows}


def _build_bn_rows(case, forward_rows, parameter_rows, update_rows, keras, pytorch):
    forward = _by_name(forward_rows, "stage")
    parameters = _by_name(parameter_rows, "parameter")
    updates = _by_name(update_rows, "parameter")
    rows = []
    for index in range(1, 5):
        mean = compare(keras["post"][f"bn{index}/mean"], pytorch["post"][f"bn{index}/mean"])
        variance = compare(
            keras["post"][f"bn{index}/variance"], pytorch["post"][f"bn{index}/variance"],
        )
        rows.append({
            "case_id": case["case_id"], "bn_layer": f"bn{index}",
            "input_relative_l2": forward[f"bn{index}_input"]["relative_l2"],
            "batch_mean_relative_l2": forward[f"bn{index}_batch_mean"]["relative_l2"],
            "batch_variance_relative_l2": forward[f"bn{index}_batch_variance"]["relative_l2"],
            "x_hat_relative_l2": forward[f"bn{index}_x_hat"]["relative_l2"],
            "affine_output_relative_l2": forward[f"bn{index}_output"]["relative_l2"],
            "gamma_gradient_relative_l2": parameters[f"bn{index}/gamma"]["relative_l2"],
            "beta_gradient_relative_l2": parameters[f"bn{index}/beta"]["relative_l2"],
            "gamma_update_relative_l2": updates[f"bn{index}/gamma"]["update_relative_l2"],
            "beta_update_relative_l2": updates[f"bn{index}/beta"]["update_relative_l2"],
            "running_mean_post_step_relative_l2": mean["relative_l2_error"],
            "running_variance_post_step_relative_l2": variance["relative_l2_error"],
        })
    return rows


def _first_nonzero(rows):
    return next((str(row["stage"]) for row in rows if float(row["max_abs_diff"]) > 0.0), "none")


def run() -> None:
    preflight_path = RESULTS / "preflight/layer_trace_precheck.json"
    if not preflight_path.exists():
        raise RuntimeError("Run layer_trace_preflight.py first")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("layer_trace_precheck") != "VALID" or preflight.get("config_sha256") != config_hash():
        raise RuntimeError("Experiment 08 layer trace preflight is not current and VALID")

    images, labels, _ = fixed_probe_batch(write_manifest=False)
    runner = LayerTraceRunner(images, labels)
    summaries = []
    for case in selected_cases():
        state, optimizer, step, _ = load_checkpoint(
            int(case["seed"]), str(case["source_framework"]), str(case["checkpoint"]),
        )
        exact, keras, pytorch = runner.run_trace(state, optimizer, step)
        for branch, values in (("keras", keras), ("pytorch", pytorch)):
            for kind in ("forward", "activation_gradients", "parameter_gradients", "updates", "adam_m", "adam_v", "post"):
                assert_finite(values[kind], f"{case['case_id']} {branch} {kind}")

        forward_rows = _build_forward_rows(case, keras, pytorch)
        activation_rows = _build_activation_gradient_rows(case, keras, pytorch)
        parameter_rows = _build_parameter_rows(case, keras, pytorch)
        update_rows = _build_update_rows(case, keras, pytorch)
        bn_rows = _build_bn_rows(case, forward_rows, parameter_rows, update_rows, keras, pytorch)
        case_id = str(case["case_id"])
        write_csv(RESULTS / "forward" / f"{case_id}_forward.csv", forward_rows)
        write_csv(RESULTS / "backward" / f"{case_id}_activation_gradients.csv", activation_rows)
        write_csv(RESULTS / "backward" / f"{case_id}_parameter_gradients.csv", parameter_rows)
        write_csv(RESULTS / "optimizer" / f"{case_id}_parameter_updates.csv", update_rows)
        write_csv(RESULTS / "bn" / f"{case_id}_bn_trace.csv", bn_rows)

        largest_forward = max(forward_rows, key=lambda row: float(row["relative_l2"]))
        largest_activation = max(activation_rows, key=lambda row: float(row["relative_l2"]))
        largest_parameter = max(parameter_rows, key=lambda row: float(row["relative_l2"]))
        lowest_cosine = min(parameter_rows, key=lambda row: float(row["cosine_similarity"]))
        largest_update = max(update_rows, key=lambda row: float(row["update_relative_l2"]))
        global_gradient = compare_group(
            keras["parameter_gradients"], pytorch["parameter_gradients"], TRAINABLE_NAMES,
        )
        global_update = compare_group(keras["updates"], pytorch["updates"], TRAINABLE_NAMES)
        global_post = compare_group(keras["post"], pytorch["post"], TRAINABLE_NAMES)
        summaries.append({
            "case_id": case_id, "seed": case["seed"], "checkpoint": case["checkpoint"],
            "optimizer_step": step, "anchor": case["anchor"],
            "full_state_sync_exact": exact["all_exact"],
            "first_nonzero_forward_stage": _first_nonzero(forward_rows),
            "largest_forward_rel_l2_stage": largest_forward["stage"],
            "largest_forward_rel_l2": largest_forward["relative_l2"],
            "forward_relative_l2_progression": json.dumps({
                row["stage"]: row["relative_l2"] for row in forward_rows
            }, separators=(",", ":")),
            "loss_keras": keras["loss"], "loss_pytorch": pytorch["loss"],
            "loss_abs_diff": abs(keras["loss"] - pytorch["loss"]),
            "loss_relative_diff": abs(keras["loss"] - pytorch["loss"]) /
                max(abs(keras["loss"]), abs(pytorch["loss"]), 1e-12),
            "first_nonzero_backward_stage": _first_nonzero(activation_rows),
            "largest_activation_gradient_stage": largest_activation["stage"],
            "largest_activation_gradient_rel_l2": largest_activation["relative_l2"],
            "largest_parameter_gradient_group": largest_parameter["parameter"],
            "largest_parameter_gradient_rel_l2": largest_parameter["relative_l2"],
            "lowest_gradient_cosine_group": lowest_cosine["parameter"],
            "lowest_gradient_cosine": lowest_cosine["cosine_similarity"],
            "largest_update_group": largest_update["parameter"],
            "largest_update_rel_l2": largest_update["update_relative_l2"],
            "global_gradient_rel_l2": global_gradient["relative_l2_error"],
            "global_update_rel_l2": global_update["relative_l2_error"],
            "global_post_weight_rel_l2": global_post["relative_l2_error"],
            "bn_running_mean_rel_l2": compare_group(
                keras["post"], pytorch["post"], BN_MEAN_NAMES,
            )["relative_l2_error"],
            "bn_running_variance_rel_l2": compare_group(
                keras["post"], pytorch["post"], BN_VARIANCE_NAMES,
            )["relative_l2_error"],
            "nan_or_inf": False,
        })
        print(
            f"trace case={case_id} first_forward={summaries[-1]['first_nonzero_forward_stage']} "
            f"global_grad={summaries[-1]['global_gradient_rel_l2']:.6g} "
            f"global_update={summaries[-1]['global_update_rel_l2']:.6g}", flush=True,
        )
        del keras, pytorch
        gc.collect()
        if runner.device.type == "mps":
            runner.torch.mps.empty_cache()

    write_csv(RESULTS / "summaries/case_summary.csv", summaries)
    print("SELECTED_CASES_COMPLETE = TRUE")
    print("NEW_FULL_TRAINING_EXECUTED = FALSE")


if __name__ == "__main__":
    run()
