"""One isolated optimizer step per seed; never modifies the full-training model."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_config import BETA1, BETA2, EPSILON, LEARNING_RATE, RESULTS, SEEDS, config_hash
from common.controlled_data import canonical_batch, canonical_rows, load_orders, numpy_reference_ce, tensor_hash
from common.controlled_initialization import (
    build_keras_model, build_torch_model, export_keras_weights, export_torch_weights,
    load_initial_weights, load_keras_weights, load_torch_weights,
)
from common.environment_utils import select_torch_device
from common.controlled_optimizer import export_keras_adam, export_torch_adam
from common.reference_adam import adam_step
from common.trace_utils import (
    assert_finite, compare, outliers, write_csv as write_protected_csv,
    write_json as write_protected_json,
)

LAYERS = ["input"] + [
    f"{part}{index}" for index in range(1, 5)
    for part in ("conv", "bn", "relu", "pool")
] + ["gap", "fc128", "dense128_relu", "logits"]


def difference(a: np.ndarray, b: np.ndarray) -> dict[str, float | list[int]]:
    if a.shape != b.shape:
        raise RuntimeError(f"Trace shape mismatch: {a.shape} vs {b.shape}")
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    delta = x - y
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return {
        "shape": list(a.shape), "keras_mean": float(x.mean()), "pytorch_mean": float(y.mean()),
        "keras_std": float(x.std()), "pytorch_std": float(y.std()),
        "mean_abs_diff": float(np.mean(np.abs(delta))),
        "max_abs_diff": float(np.max(np.abs(delta))),
        "mse": float(np.mean(delta * delta)), "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosine_similarity": float(np.dot(x, y) / denom) if denom else (1.0 if np.array_equal(x, y) else 0.0),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    # Original VALID traces are immutable; extensions use new filenames below.
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def keras_gradient_map(model, gradients) -> dict[str, np.ndarray]:
    lookup = {id(variable): gradient.numpy() for variable, gradient in zip(model.trainable_variables, gradients)}
    result = {}
    for index in range(1, 5):
        result[f"conv{index}/kernel"] = lookup[id(model.get_layer(f"conv{index}").kernel)]
        bn = model.get_layer(f"bn{index}")
        result[f"bn{index}/gamma"] = lookup[id(bn.gamma)]
        result[f"bn{index}/beta"] = lookup[id(bn.beta)]
    for name in ("fc128", "logits"):
        layer = model.get_layer(name)
        result[f"{name}/kernel"] = lookup[id(layer.kernel)]
        result[f"{name}/bias"] = lookup[id(layer.bias)]
    return result


def torch_gradient_map(model) -> dict[str, np.ndarray]:
    result = {}
    for index in range(1, 5):
        result[f"conv{index}/kernel"] = getattr(model, f"conv{index}").weight.grad.detach().cpu().numpy().transpose(2, 3, 1, 0).copy()
        bn = getattr(model, f"bn{index}")
        result[f"bn{index}/gamma"] = bn.weight.grad.detach().cpu().numpy().copy()
        result[f"bn{index}/beta"] = bn.bias.grad.detach().cpu().numpy().copy()
    for name in ("fc128", "logits"):
        layer = getattr(model, name)
        result[f"{name}/kernel"] = layer.weight.grad.detach().cpu().numpy().T.copy()
        result[f"{name}/bias"] = layer.bias.grad.detach().cpu().numpy().copy()
    return result


def run_seed(seed: int) -> dict[str, object]:
    import tensorflow as tf
    import torch

    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx("float32")
    rows = canonical_rows("train")
    order = load_orders(seed, len(rows))[0]
    images, labels = canonical_batch(rows, order[:32], seed, 0, True)
    w0 = load_initial_weights(seed)
    keras_input = tf.convert_to_tensor(images, dtype=tf.float32)
    torch_input = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(select_torch_device())
    torch_labels = torch.from_numpy(labels.astype(np.int64)).to(torch_input.device)
    if "GPU:" not in keras_input.device.upper():
        raise RuntimeError(f"Keras first-step input is not on GPU: {keras_input.device}")
    if torch_input.device.type != "mps":
        raise RuntimeError(f"PyTorch first-step input is not on MPS: {torch_input.device}")
    if not np.array_equal(images, torch_input.detach().cpu().numpy().transpose(0, 2, 3, 1)):
        raise RuntimeError("First-step input tensors differ.")
    if not np.array_equal(labels, torch_labels.cpu().numpy()):
        raise RuntimeError("First-step labels differ.")

    keras_model = build_keras_model()
    load_keras_weights(keras_model, w0)
    torch_model = build_torch_model().to(torch_input.device)
    load_torch_weights(torch_model, w0)
    if any(not np.array_equal(export_keras_weights(keras_model)[key], export_torch_weights(torch_model)[key]) for key in w0):
        raise RuntimeError("First-step initial weights differ.")
    initial_hash = tensor_hash(np.concatenate([w0[key].ravel() for key in sorted(w0)]))
    trace_model = tf.keras.Model(keras_model.input, [keras_model.get_layer(name).output for name in LAYERS[1:]])
    keras_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=BETA1, beta_2=BETA2,
        epsilon=EPSILON, amsgrad=False, clipnorm=None, clipvalue=None,
        global_clipnorm=None, use_ema=False,
    )
    keras_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        outputs = trace_model(keras_input, training=True)
        keras_logits = outputs[-1]
        keras_loss = keras_ce(tf.convert_to_tensor(labels, dtype=tf.int32), keras_logits)
    keras_gradients = tape.gradient(keras_loss, keras_model.trainable_variables)
    if any(gradient is None for gradient in keras_gradients):
        raise RuntimeError("Missing Keras gradient.")
    keras_trace = {"input": images}
    keras_trace.update({name: value.numpy() for name, value in zip(LAYERS[1:], outputs)})
    keras_grads = keras_gradient_map(keras_model, keras_gradients)
    assert_finite(keras_trace, f"Keras forward seed={seed}")
    assert_finite(keras_grads, f"Keras gradients seed={seed}")

    torch_trace = {"input": images}
    hooks = []
    for name in LAYERS[1:]:
        module = getattr(torch_model, name)
        def capture(_module, _inputs, output, layer_name=name):
            value = output.detach().cpu().numpy()
            if layer_name not in {"gap", "fc128", "dense128_relu", "logits"}:
                value = value.transpose(0, 2, 3, 1)
            elif layer_name == "gap":
                value = value.reshape(value.shape[0], -1)
            torch_trace[layer_name] = value.copy()
        hooks.append(module.register_forward_hook(capture))
    torch_model.train()
    torch_optimizer = torch.optim.Adam(
        torch_model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2),
        eps=EPSILON, weight_decay=0.0, amsgrad=False, foreach=False,
    )
    torch_optimizer.zero_grad(set_to_none=True)
    torch_logits = torch_model(torch_input)
    torch_loss = torch.nn.functional.cross_entropy(torch_logits, torch_labels, reduction="mean")
    torch_loss.backward()
    torch_grads = torch_gradient_map(torch_model)
    assert_finite(torch_trace, f"PyTorch forward seed={seed}")
    assert_finite(torch_grads, f"PyTorch gradients seed={seed}")
    for hook in hooks:
        hook.remove()

    output_dir = RESULTS / "first_step"
    forward_rows = [{"layer": name, **difference(keras_trace[name], torch_trace[name])} for name in LAYERS]
    write_csv(output_dir / f"seed{seed}_forward_trace.csv", forward_rows)
    loss_trace = {
        "seed": seed, "labels_hash": tensor_hash(labels), "input_hash": tensor_hash(images),
        "weight_hash": initial_hash,
        "keras_native_loss": float(keras_loss.numpy()),
        "pytorch_native_loss": float(torch_loss.detach().cpu()),
        "numpy_reference_loss_from_keras_logits": numpy_reference_ce(keras_logits.numpy(), labels),
        "numpy_reference_loss_from_pytorch_logits": numpy_reference_ce(torch_logits.detach().cpu().numpy(), labels),
    }
    legacy_loss_path = output_dir / f"seed{seed}_loss_trace.json"
    if not legacy_loss_path.exists():
        legacy_loss_path.write_text(json.dumps(loss_trace, indent=2), encoding="utf-8")
    gradient_rows = [{"parameter": name, **difference(keras_grads[name], torch_grads[name])} for name in keras_grads]
    write_csv(output_dir / f"seed{seed}_gradient_trace.csv", gradient_rows)

    keras_optimizer.apply_gradients(zip(keras_gradients, keras_model.trainable_variables))
    torch_optimizer.step()
    keras_adam, keras_step = export_keras_adam(keras_model, keras_optimizer)
    torch_adam, torch_step = export_torch_adam(torch_model, torch_optimizer)
    if keras_step != torch_step or keras_step != 1 or set(keras_adam) != set(torch_adam):
        raise RuntimeError("Adam step counter or canonical slot mapping mismatch")
    assert_finite(keras_adam, f"Keras Adam seed={seed}")
    assert_finite(torch_adam, f"PyTorch Adam seed={seed}")
    keras_w1, torch_w1 = export_keras_weights(keras_model), export_torch_weights(torch_model)
    update_rows = []
    for name in keras_grads:
        keras_delta, torch_delta = keras_w1[name] - w0[name], torch_w1[name] - w0[name]
        update_rows.append({
            "parameter": name,
            "delta_norm_keras": float(np.linalg.norm(keras_delta)),
            "delta_norm_pytorch": float(np.linalg.norm(torch_delta)),
            "delta_mean_abs_diff": float(np.mean(np.abs(keras_delta - torch_delta))),
            "delta_max_abs_diff": float(np.max(np.abs(keras_delta - torch_delta))),
            "updated_weight_max_abs_diff": float(np.max(np.abs(keras_w1[name] - torch_w1[name]))),
        })
    write_csv(output_dir / f"seed{seed}_update_trace.csv", update_rows)
    bn_rows = []
    for index in range(1, 5):
        for field in ("mean", "variance"):
            name = f"bn{index}/{field}"
            bn_rows.append({"state": name, **difference(keras_w1[name], torch_w1[name])})
    write_csv(output_dir / f"seed{seed}_bn_state_trace.csv", bn_rows)
    assert_finite(keras_w1, f"Keras updated weights seed={seed}")
    assert_finite(torch_w1, f"PyTorch updated weights seed={seed}")
    extended_forward = [{"seed": seed, "layer": name, **compare(keras_trace[name], torch_trace[name])} for name in LAYERS]
    extended_gradients = [{"seed": seed, "parameter": name, "layer": name.split("/")[0], **compare(keras_grads[name], torch_grads[name])} for name in keras_grads]
    extended_updates, gradient_outliers, update_outliers = [], [], []
    adam_rows, reference_rows = [], []
    for name in keras_grads:
        kg, pg = keras_grads[name], torch_grads[name]
        kd, pd = keras_w1[name] - w0[name], torch_w1[name] - w0[name]
        extended_updates.append({"seed": seed, "parameter": name, "layer": name.split("/")[0], **compare(kd, pd)})
        gradient_outliers.extend(outliers(kg, pg, seed=seed, parameter=name))
        update_outliers.extend(outliers(kd, pd, seed=seed, parameter=name))
        km, pm = keras_adam[f"{name}/m"], torch_adam[f"{name}/m"]
        kv, pv = keras_adam[f"{name}/v"], torch_adam[f"{name}/v"]
        m_stats, v_stats = compare(km, pm), compare(kv, pv)
        adam_rows.append({
            "seed": seed, "parameter": name,
            **{f"m_{key}": value for key, value in m_stats.items()},
            **{f"v_{key}": value for key, value in v_stats.items()},
        })
        _, _, ref_k = adam_step(kg)
        _, _, ref_p = adam_step(pg)
        k_native_ref = compare(kd, ref_k)
        p_native_ref = compare(pd, ref_p)
        ref_pair = compare(ref_k, ref_p)
        reference_rows.append({
            "seed": seed, "parameter": name,
            "keras_native_delta_norm": k_native_ref["keras_norm"],
            "keras_reference_delta_norm": k_native_ref["pytorch_norm"],
            "keras_native_vs_reference_mean_abs_diff": k_native_ref["mean_abs_diff"],
            "keras_native_vs_reference_max_abs_diff": k_native_ref["max_abs_diff"],
            "keras_native_vs_reference_relative_l2": k_native_ref["relative_l2_error"],
            "pytorch_native_delta_norm": p_native_ref["keras_norm"],
            "pytorch_reference_delta_norm": p_native_ref["pytorch_norm"],
            "pytorch_native_vs_reference_mean_abs_diff": p_native_ref["mean_abs_diff"],
            "pytorch_native_vs_reference_max_abs_diff": p_native_ref["max_abs_diff"],
            "pytorch_native_vs_reference_relative_l2": p_native_ref["relative_l2_error"],
            "reference_keras_grad_vs_pytorch_grad_delta_mean_abs_diff": ref_pair["mean_abs_diff"],
            "reference_keras_grad_vs_pytorch_grad_delta_max_abs_diff": ref_pair["max_abs_diff"],
            "reference_keras_grad_vs_pytorch_grad_delta_relative_l2": ref_pair["relative_l2_error"],
        })
    write_protected_csv(output_dir / f"seed{seed}_forward_distribution.csv", extended_forward)
    write_protected_csv(output_dir / f"seed{seed}_gradient_distribution.csv", extended_gradients)
    write_protected_csv(output_dir / f"seed{seed}_update_distribution.csv", extended_updates)
    write_protected_csv(output_dir / f"seed{seed}_gradient_outliers.csv", gradient_outliers)
    write_protected_csv(output_dir / f"seed{seed}_update_outliers.csv", update_outliers)
    write_protected_csv(output_dir / f"seed{seed}_adam_state_trace.csv", adam_rows)
    write_protected_csv(output_dir / f"seed{seed}_reference_adam_trace.csv", reference_rows)
    write_protected_json(output_dir / f"seed{seed}_optimizer_metadata.json", {
        "seed": seed, "keras_iteration": keras_step, "pytorch_step": torch_step,
        "learning_rate": LEARNING_RATE, "beta1": BETA1, "beta2": BETA2,
        "epsilon": EPSILON, "weight_decay": 0.0, "amsgrad": False,
        "config_sha256": config_hash(),
        "keras_slot_mapping": "variable.path -> <path_with_underscores>_momentum/_velocity; verified names/shapes",
        "pytorch_slot_mapping": "named_parameters -> exp_avg/exp_avg_sq; verified state/shape",
    })
    first_difference = next((row["layer"] for row in forward_rows if row["max_abs_diff"] > 0), None)
    summary = {
        "seed": seed, "diagnostic_only": True, "initial_weight_exact": True,
        "input_exact": True, "labels_exact": True,
        "keras_device_available": bool(tf.config.list_physical_devices("GPU")),
        "keras_tensor_device": keras_input.device,
        "pytorch_device": str(torch_input.device),
        "first_nonzero_forward_layer": first_difference,
        "first_nonzero_not_significance_claim": True,
        "forward_max_abs_by_layer": {row["layer"]: row["max_abs_diff"] for row in forward_rows},
        "loss": loss_trace,
    }
    legacy_summary_path = output_dir / f"seed{seed}_summary.json"
    if not legacy_summary_path.exists():
        legacy_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    def top(rows, metric):
        row = max(rows, key=lambda item: float(item[metric]))
        return {"parameter": row["parameter"], "value": row[metric]}
    diagnostic_summary = {
        "seed": seed, "config_sha256": config_hash(), "initial_weight_exact": True,
        "input_exact": True, "conv1_exact": extended_forward[1]["max_abs_diff"] == 0,
        "first_nonzero_forward_layer": first_difference,
        "first_nonzero_forward_max_diff": next((row["max_abs_diff"] for row in extended_forward if row["layer"] == first_difference), 0),
        "loss_abs_diff": abs(loss_trace["keras_native_loss"] - loss_trace["pytorch_native_loss"]),
        "largest_gradient_relative_l2": top(extended_gradients, "relative_l2_error"),
        "largest_gradient_max_diff": top(extended_gradients, "max_abs_diff"),
        "largest_update_relative_l2": top(extended_updates, "relative_l2_error"),
        "largest_update_max_diff": top(extended_updates, "max_abs_diff"),
        "largest_m_relative_l2": top(adam_rows, "m_relative_l2_error"),
        "largest_v_relative_l2": top(adam_rows, "v_relative_l2_error"),
        "bn_state_max_abs_diff": max(float(row["max_abs_diff"]) for row in bn_rows),
        "nan_count": 0, "inf_count": 0,
    }
    write_protected_json(output_dir / f"seed{seed}_diagnostic_summary.json", diagnostic_summary)
    print(f"Seed {seed}: first nonzero forward layer={first_difference}; Keras GPU={summary['keras_device_available']} PyTorch={summary['pytorch_device']}")
    return summary


def main() -> None:
    RESULTS.joinpath("first_step").mkdir(parents=True, exist_ok=True)
    summaries = [run_seed(seed) for seed in SEEDS]
    aggregate = {}
    for seed in SEEDS:
        item = json.loads((RESULTS / "first_step" / f"seed{seed}_diagnostic_summary.json").read_text(encoding="utf-8"))
        aggregate[str(seed)] = {key: item[key] for key in (
            "first_nonzero_forward_layer", "largest_gradient_relative_l2",
            "largest_gradient_max_diff", "largest_update_relative_l2",
            "largest_update_max_diff", "largest_m_relative_l2", "largest_v_relative_l2",
        )}
    write_protected_json(RESULTS / "first_step" / "diagnostic_3seed_summary.json", {
        "config_sha256": config_hash(), "n_seeds": len(SEEDS),
        "significance_claim": False, "by_seed": aggregate,
    })
    print("First-step divergence diagnostic completed; no full training was run.")
    print({row["seed"]: row["first_nonzero_forward_layer"] for row in summaries})


if __name__ == "__main__":
    main()
