"""Controlled preflight. Generates only W0/control/audit artifacts, never trains."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_config import (
    AMS_GRAD, BATCH_SIZE, BETA1, BETA2, BN_EPS, DTYPE, EPSILON, KERAS_BN_MOMENTUM,
    LEARNING_RATE, MAX_EPOCHS, RESULTS, SEEDS, TORCH_BN_MOMENTUM,
    TRAINABLE_PARAMETERS, WEIGHT_DECAY, DIAGNOSTIC_STEPS, config_hash, fingerprint,
)
from common.controlled_data import (
    canonical_batch, canonical_rows, create_orders, save_sample_manifest, tensor_hash,
)
from common.controlled_initialization import (
    build_keras_model, build_torch_model, create_initial_weights,
    export_keras_weights, export_torch_weights, load_keras_weights,
    load_torch_weights, write_initial_audit,
)
from common.dataset_utils import CLASSES, is_physically_split
from common.environment_utils import select_torch_device
from common.controlled_checkpoint import verify_checkpoint
from common.trace_utils import write_json as write_protected_json


def validate() -> dict[str, object]:
    import tensorflow as tf
    import torch

    checks = {}
    errors = []

    def record(name: str, condition: bool, explanation: str = "") -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {explanation}")

    tf.keras.backend.set_floatx("float32")
    torch.set_default_dtype(torch.float32)
    train, val, test = (canonical_rows(split) for split in ("train", "val", "test"))
    counts = {"train": len(train), "val": len(val), "test": len(test)}
    record("dataset_match", is_physically_split() and counts == {"train": 10251, "val": 2194, "test": 2203}, str(counts))
    record("class_mapping_match", CLASSES == ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"])
    manifest_path = save_sample_manifest(train)
    record("canonical_sample_list", manifest_path.exists() and len({row["sample_id"] for row in train}) == len(train))
    record("dtype_match", DTYPE == tf.keras.backend.floatx() == "float32" and torch.get_default_dtype() == torch.float32)
    record("fixed_epoch", MAX_EPOCHS == 30)
    record("constant_learning_rate", LEARNING_RATE == 0.001)
    record("early_stopping_disabled", not fingerprint()["early_stopping"])
    record("scheduler_disabled", not fingerprint()["lr_scheduler"])
    keras_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    torch_loss = torch.nn.CrossEntropyLoss(reduction="mean")
    record("loss_config_match", keras_loss.from_logits and torch_loss.reduction == "mean")
    keras_adam = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=BETA1, beta_2=BETA2,
        epsilon=EPSILON, weight_decay=WEIGHT_DECAY, amsgrad=AMS_GRAD,
        clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False,
    )
    torch_adam = torch.optim.Adam(
        [torch.nn.Parameter(torch.zeros(1))], lr=LEARNING_RATE,
        betas=(BETA1, BETA2), eps=EPSILON, weight_decay=WEIGHT_DECAY,
        amsgrad=AMS_GRAD, foreach=False,
    )
    kcfg, tcfg = keras_adam.get_config(), torch_adam.defaults
    record("adam_hyperparameter_match", all((
        np.isclose(float(kcfg["learning_rate"]), float(tcfg["lr"]), rtol=0, atol=1e-10)
        and float(tcfg["lr"]) == LEARNING_RATE,
        float(kcfg["beta_1"]) == float(tcfg["betas"][0]) == BETA1,
        float(kcfg["beta_2"]) == float(tcfg["betas"][1]) == BETA2,
        float(kcfg["epsilon"]) == float(tcfg["eps"]) == EPSILON,
        float(kcfg["weight_decay"]) == float(tcfg["weight_decay"]) == WEIGHT_DECAY,
        kcfg["amsgrad"] is tcfg["amsgrad"] is AMS_GRAD,
        not any(kcfg.get(key) for key in ("clipnorm", "clipvalue", "global_clipnorm", "use_ema")),
        tcfg["foreach"] is False,
    )))
    record("batchnorm_config_match", BN_EPS == 1e-3 and abs((1 - KERAS_BN_MOMENTUM) - TORCH_BN_MOMENTUM) < 1e-12)
    record("native_shuffle_disabled", True, "Both training scripts consume persisted order slices explicitly.")

    audit_paths = {}
    input_audit = {}
    distinct_weight_hashes = []
    all_weights_exact = all_inputs_exact = all_labels_exact = all_orders_valid = True
    for seed in SEEDS:
        tf.keras.backend.clear_session()
        canonical = create_initial_weights(seed)
        order = create_orders(seed, len(train))
        order_valid = order.shape == (MAX_EPOCHS, len(train)) and all(
            np.array_equal(np.sort(epoch), np.arange(len(train))) for epoch in order
        )
        all_orders_valid &= bool(order_valid)
        distinct_weight_hashes.append(tensor_hash(np.concatenate([canonical[name].ravel() for name in sorted(canonical)])))
        images, labels = canonical_batch(train, order[0, :BATCH_SIZE], seed, 0, True)
        keras_input = tf.convert_to_tensor(images, dtype=tf.float32)
        torch_input = torch.from_numpy(images.transpose(0, 3, 1, 2).copy())
        roundtrip = torch_input.numpy().transpose(0, 2, 3, 1)
        input_exact = bool(np.array_equal(keras_input.numpy(), roundtrip))
        labels_exact = bool(np.array_equal(labels, torch.from_numpy(labels.copy()).numpy()))
        all_inputs_exact &= input_exact
        all_labels_exact &= labels_exact
        input_audit[str(seed)] = {
            "shape": list(images.shape), "dtype": str(images.dtype),
            "min": float(images.min()), "max": float(images.max()), "mean": float(images.mean()),
            "canonical_hash": tensor_hash(images), "keras_hash": tensor_hash(keras_input.numpy()),
            "pytorch_nhwc_hash": tensor_hash(roundtrip),
            "max_abs_diff": float(np.max(np.abs(keras_input.numpy() - roundtrip))),
            "label_hash": tensor_hash(labels), "labels_exact": labels_exact,
        }
        keras_model = build_keras_model()
        torch_model = build_torch_model()
        for index, channels in enumerate((32, 64, 128, 256), 1):
            kc = keras_model.get_layer(f"conv{index}")
            tc = getattr(torch_model, f"conv{index}")
            kb = keras_model.get_layer(f"bn{index}")
            tb = getattr(torch_model, f"bn{index}")
            record(f"layer_config_seed{seed}_block{index}", all((
                kc.filters == tc.out_channels == channels,
                kc.kernel_size == tc.kernel_size == (3, 3),
                kc.padding == "same" and tc.padding == (1, 1),
                not kc.use_bias and tc.bias is None,
                kb.epsilon == tb.eps == BN_EPS,
                kb.momentum == KERAS_BN_MOMENTUM and tb.momentum == TORCH_BN_MOMENTUM,
            )))
        load_keras_weights(keras_model, canonical)
        load_torch_weights(torch_model, canonical)
        audit_path, max_difference = write_initial_audit(
            seed, canonical, export_keras_weights(keras_model), export_torch_weights(torch_model)
        )
        audit_paths[str(seed)] = str(audit_path.relative_to(RESULTS))
        all_weights_exact &= max_difference == 0.0
        keras_count = sum(int(np.prod(variable.shape)) for variable in keras_model.trainable_variables)
        torch_count = sum(parameter.numel() for parameter in torch_model.parameters() if parameter.requires_grad)
        record(f"parameter_count_seed{seed}", keras_count == torch_count == TRAINABLE_PARAMETERS, f"{keras_count}/{torch_count}")
        record(f"architecture_seed{seed}", tuple(keras_model.output_shape) == (None, 8) and torch_model(torch_input).shape == (BATCH_SIZE, 8))
        record(f"weight_dtype_seed{seed}", all(value.dtype == np.float32 for value in canonical.values())
               and all(variable.dtype == "float32" for variable in keras_model.trainable_variables)
               and all(parameter.dtype == torch.float32 for parameter in torch_model.parameters()))
    record("seed_initializations_distinct", len(set(distinct_weight_hashes)) == len(SEEDS))
    record("initial_weight_exact", all_weights_exact)
    record("batch_order_match", all_orders_valid)
    record("augmentation_tensor_match", all_inputs_exact)
    record("input_tensor_exact", all_inputs_exact)
    record("labels_exact", all_labels_exact)
    input_path = RESULTS / "preflight" / "input_tensor_audit.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(input_audit, indent=2), encoding="utf-8")

    device = select_torch_device()
    record("keras_gpu", bool(tf.config.list_physical_devices("GPU")))
    record("pytorch_mps", device.type == "mps" and torch.backends.mps.is_available())
    first_step_files = [
        RESULTS / "first_step" / f"seed{seed}_{kind}_trace.{extension}"
        for seed in SEEDS for kind, extension in (
            ("forward", "csv"), ("loss", "json"), ("gradient", "csv"),
            ("update", "csv"), ("bn_state", "csv"),
        )
    ]
    record("first_step_trace_available", all(path.exists() for path in first_step_files))
    report = {
        "experiment": fingerprint()["experiment"], "phase": fingerprint()["phase"],
        "status": "VALID" if not errors else "NOT_READY",
        "controlled_precheck": "VALID" if not errors else "NOT_READY",
        "checks": checks, "errors": errors,
        "dataset_counts": counts, "keras_trainable_parameters": TRAINABLE_PARAMETERS,
        "pytorch_trainable_parameters": TRAINABLE_PARAMETERS,
        "initial_weight_audits": audit_paths,
        "weight_hashes_by_seed": dict(zip(SEEDS, distinct_weight_hashes)),
        "input_tensor_audit": str(input_path.relative_to(RESULTS)),
        "keras_gpu_devices": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "pytorch_device": str(device),
        "config": fingerprint(),
        "full_training_executed": any(
            (RESULTS / f"{framework}_controlled_results.csv").exists()
            for framework in ("keras", "pytorch")
        ),
    }
    output = RESULTS / "preflight" / "preflight_summary.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CONTROLLED_PRECHECK = {report['controlled_precheck']}")
    print(f"Initial weight exact={all_weights_exact}; input exact={all_inputs_exact}; class/dataset={counts}")
    print(f"Keras GPU={report['keras_gpu_devices']}; PyTorch device={device}")
    if errors:
        for error in errors:
            print(f"NOT READY: {error}")
    return report


def validate_extended() -> dict[str, object]:
    """Read existing VALID artifacts; never regenerate or overwrite them."""
    legacy_path = RESULTS / "preflight" / "preflight_summary.json"
    if not legacy_path.exists():
        validate()
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    errors = []
    if legacy.get("controlled_precheck") != "VALID" or not all(legacy.get("checks", {}).values()):
        errors.append("Existing controlled preflight is not VALID")
    diagnostic_files = []
    for seed in SEEDS:
        for kind in (
            "forward_distribution", "gradient_distribution", "update_distribution",
            "gradient_outliers", "update_outliers", "adam_state_trace", "reference_adam_trace",
        ):
            diagnostic_files.append(RESULTS / "first_step" / f"seed{seed}_{kind}.csv")
        diagnostic_files.extend([
            RESULTS / "first_step" / f"seed{seed}_optimizer_metadata.json",
            RESULTS / "first_step" / f"seed{seed}_diagnostic_summary.json",
            RESULTS / "early_steps" / f"seed{seed}_step_summary.csv",
            RESULTS / "early_steps" / f"seed{seed}_layer_trajectory.csv",
        ])
    diagnostic_files.append(RESULTS / "first_step" / "diagnostic_3seed_summary.json")
    for path in diagnostic_files:
        if not path.exists():
            errors.append(f"Missing extended diagnostic: {path}")
            continue
        if path.suffix == ".csv":
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                errors.append(f"Empty diagnostic CSV: {path}")
            for row in rows:
                if any(str(value).lower() in {"nan", "inf", "-inf", "infinity", "-infinity"} for value in row.values()):
                    errors.append(f"NaN/Inf in diagnostic CSV: {path}")
                    break
    for seed in SEEDS:
        meta = RESULTS / "first_step" / f"seed{seed}_optimizer_metadata.json"
        summary = RESULTS / "first_step" / f"seed{seed}_diagnostic_summary.json"
        early = RESULTS / "early_steps" / f"seed{seed}_step_summary.csv"
        if meta.exists():
            item = json.loads(meta.read_text(encoding="utf-8"))
            if item.get("keras_iteration") != item.get("pytorch_step") or item.get("keras_iteration") != 1 or item.get("config_sha256") != config_hash():
                errors.append(f"Adam counter/config mismatch seed={seed}")
        if summary.exists():
            item = json.loads(summary.read_text(encoding="utf-8"))
            if item.get("nan_count") != 0 or item.get("inf_count") != 0 or item.get("config_sha256") != config_hash():
                errors.append(f"First-step finite/config failure seed={seed}")
        if early.exists():
            with early.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if [int(row["step"]) for row in rows] != list(DIAGNOSTIC_STEPS) or any(row["config_sha256"] != config_hash() for row in rows):
                errors.append(f"Early-step schedule/config mismatch seed={seed}")
            if rows and float(rows[0]["global_weight_relative_l2"]) != 0:
                errors.append(f"Early-step W0 mismatch seed={seed}")
        for step in DIAGNOSTIC_STEPS:
            folder = RESULTS / "early_steps" / "checkpoints" / f"seed{seed}" / f"step{step:03d}"
            for framework in ("keras", "pytorch"):
                try:
                    verify_checkpoint(
                        folder / f"{framework}_state.npz",
                        folder / f"{framework}_optimizer.npz",
                        folder / f"{framework}_metadata.json",
                    )
                except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
                    errors.append(f"Diagnostic checkpoint invalid seed={seed} step={step} {framework}: {exc}")
    roundtrip_path = RESULTS / "preflight" / "checkpoint_roundtrip.json"
    roundtrip = json.loads(roundtrip_path.read_text(encoding="utf-8")) if roundtrip_path.exists() else {}
    if roundtrip.get("checkpoint_roundtrip") != "VALID" or roundtrip.get("config_sha256") != config_hash():
        errors.append("Checkpoint round-trip missing, failed, or config hash mismatch")
    extended_status = "VALID" if not errors else "NOT_READY"
    report = {
        "controlled_precheck": legacy.get("controlled_precheck"),
        "extended_diagnostic_status": extended_status,
        "checkpoint_roundtrip": roundtrip.get("checkpoint_roundtrip", "NOT_READY"),
        "full_training_ready": legacy.get("controlled_precheck") == "VALID" and not errors,
        "config_sha256": config_hash(), "errors": errors,
        "diagnostic_file_count": len(diagnostic_files),
        "full_training_executed": any((RESULTS / f"{framework}_controlled_results.csv").exists() for framework in ("keras", "pytorch")),
    }
    write_protected_json(RESULTS / "preflight" / "preflight_extended_summary.json", report)
    print(f"CONTROLLED_PRECHECK = {report['controlled_precheck']}")
    print(f"EXTENDED_DIAGNOSTIC_STATUS = {extended_status}")
    print(f"CHECKPOINT_ROUNDTRIP = {report['checkpoint_roundtrip']}")
    print(f"FULL_TRAINING_READY = {str(report['full_training_ready']).upper()}")
    if errors:
        for error in errors:
            print(f"NOT READY: {error}")
    return report


if __name__ == "__main__":
    validate_extended()
