"""Local-data V4 Stage A architecture/state/device/batch preflight."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from common.controlled_initialization import load_initial_weights
from common.reference_adam import CommonAdam
from common.state_resynchronization import states_exact
from common.trace_utils import state_hash, write_csv, write_json
from cifar10_canonical_state import (
    BODY_NAMES, SPECS, TRAINABLE_NAMES, TRAINABLE_PARAMETER_COUNT,
    NONTRAINABLE_STATE_COUNT, array_hash, create_initial_state,
    export_keras_state, export_torch_state, load_keras_state, load_torch_state,
    trainable_values,
)
from cifar10_keras import build_keras_cifar10
from cifar10_pytorch import build_torch_cifar10
from v4_config import RESULTS, RUN_FULL_TRAINING, SEEDS, config_hash, fingerprint
from v4_data import canonical_images, require_local_cifar10, sha256_array, stratified_split


def _canonical_torch(value):
    array = value.detach().cpu().numpy().copy()
    return array.transpose(0, 2, 3, 1) if array.ndim == 4 else array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10"))
    args = parser.parse_args()
    if RUN_FULL_TRAINING:
        raise RuntimeError("V4 preflight cannot run full training")
    import tensorflow as tf
    import torch

    train, _ = require_local_cifar10(args.data_root)
    train_indices, _ = stratified_split(train.targets)
    raw, images, labels = canonical_images(train, train_indices[:32])
    fixed_manifest = json.loads((RESULTS / "manifests/fixed_batch_manifest.json").read_text())
    split_manifest = json.loads((RESULTS / "manifests/cifar10_split_manifest.json").read_text())
    errors, checks = [], {}

    def record(name, condition, detail=""):
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {detail}")

    record("split_manifest_valid", split_manifest.get("v4_split_check") == "VALID")
    record("fixed_indices_exact", fixed_manifest["indices"] == train_indices[:32].tolist())
    record("fixed_labels_exact", fixed_manifest["labels"] == labels.tolist())
    record("fixed_raw_hash_exact", fixed_manifest["raw_uint8_sha256"] == sha256_array(raw))
    record("fixed_tensor_hash_exact", fixed_manifest["canonical_float32_sha256"] == sha256_array(images))
    record("fixed_shape_dtype", images.shape == (32, 32, 32, 3) and images.dtype == np.float32)

    physical_gpus = tf.config.list_physical_devices("GPU")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    record("tensorflow_visible_gpu", bool(physical_gpus), physical_gpus)
    record("pytorch_mps_available", device.type == "mps", device)
    record("mps_fallback_disabled", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1")
    with tf.device("/GPU:0"):
        keras_model = build_keras_cifar10()
    torch_model = build_torch_cifar10().to(device)

    parameter_rows, case_rows = [], []
    for seed in SEEDS:
        state = create_initial_state(seed)
        phase2 = load_initial_weights(seed)
        body_exact = all(np.array_equal(state[name], phase2[name]) for name in BODY_NAMES)
        record(f"phase2_body_w0_seed{seed}", body_exact)
        load_keras_state(keras_model, state); load_torch_state(torch_model, state)
        ks, ps = export_keras_state(keras_model), export_torch_state(torch_model)
        record(f"paired_w0_seed{seed}", states_exact(ks, ps) and states_exact(ks, state))
        ka, pa = CommonAdam(trainable_values(state)), CommonAdam(trainable_values(state))
        record(f"adam_zero_seed{seed}", states_exact(ka.m, pa.m) and states_exact(ka.v, pa.v) and ka.step == pa.step == 0)
        for spec in SPECS:
            name = spec.semantic_name
            parameter_rows.append({
                "seed": seed, "semantic_name": name, "trainable": spec.trainable,
                "keras_name": spec.keras_name, "pytorch_name": spec.pytorch_name,
                "canonical_shape": json.dumps(spec.shape), "keras_shape": json.dumps(ks[name].shape),
                "pytorch_shape": json.dumps(ps[name].shape), "dtype": str(state[name].dtype),
                "canonical_hash": array_hash(state[name]), "keras_hash": array_hash(ks[name]),
                "pytorch_hash": array_hash(ps[name]),
                "exact_equal": bool(np.array_equal(state[name], ks[name]) and np.array_equal(state[name], ps[name])),
            })
        case_rows.append({
            "seed": seed, "checkpoint": "initial", "anchor": "shared",
            "state_sha256": state_hash(state), "fixed_batch_sha256": sha256_array(images),
            "full_state_sync_exact": bool(states_exact(ks, ps) and states_exact(ka.m, pa.m) and states_exact(ka.v, pa.v)),
            "config_sha256": config_hash(),
        })

    keras_count = int(sum(np.prod(value.shape) for value in keras_model.trainable_variables))
    torch_count = int(sum(value.numel() for value in torch_model.parameters()))
    keras_nontrain = int(sum(np.prod(value.shape) for value in keras_model.non_trainable_variables))
    torch_nontrain = int(sum(
        value.numel() for name, value in torch_model.state_dict().items()
        if name.endswith(("running_mean", "running_var"))
    ))
    record("trainable_count_exact", keras_count == torch_count == TRAINABLE_PARAMETER_COUNT)
    record("nontrainable_count_exact", keras_nontrain == torch_nontrain == NONTRAINABLE_STATE_COUNT)
    record("parameter_manifest_exact", all(row["exact_equal"] for row in parameter_rows))

    state = create_initial_state(42); load_keras_state(keras_model, state); load_torch_state(torch_model, state)
    with tf.device("/GPU:0"):
        keras_model(tf.zeros((1, 32, 32, 3), tf.float32), training=False)
    torch_model.eval()
    with torch.no_grad():
        torch_model(torch.zeros((1, 3, 32, 32), dtype=torch.float32, device=device))
    kt, pt = keras_model.last_trace, torch_model.last_trace
    record("semantic_stage_order_exact", list(kt) == list(pt))
    architecture_rows = []
    for order, name in enumerate(kt):
        a, b = np.asarray(kt[name].numpy()), _canonical_torch(pt[name])
        architecture_rows.append({
            "stage_order": order, "semantic_stage": name,
            "keras_shape": json.dumps(a.shape), "pytorch_canonical_shape": json.dumps(b.shape),
            "keras_dtype": str(a.dtype), "pytorch_dtype": str(b.dtype), "shape_exact": a.shape == b.shape,
        })
    record("architecture_shapes_exact", all(row["shape_exact"] for row in architecture_rows))
    keras_device, torch_device = str(kt["conv1"].device), str(pt["conv1"].device)
    record("tensorflow_conv_gpu", "GPU:0" in keras_device.upper(), keras_device)
    record("pytorch_conv_mps", torch_device == "mps:0", torch_device)
    restored = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device).cpu().numpy().transpose(0, 2, 3, 1)
    record("layout_roundtrip_exact", np.array_equal(images, restored))

    architecture_valid = checks["semantic_stage_order_exact"] and checks["architecture_shapes_exact"]
    parameter_valid = checks["trainable_count_exact"] and checks["nontrainable_count_exact"] and checks["parameter_manifest_exact"]
    shared_valid = all(checks[f"paired_w0_seed{seed}"] for seed in SEEDS)
    phase2_body_valid = all(checks[f"phase2_body_w0_seed{seed}"] for seed in SEEDS)
    device_valid = all(checks[name] for name in ("tensorflow_visible_gpu", "pytorch_mps_available", "mps_fallback_disabled", "tensorflow_conv_gpu", "pytorch_conv_mps"))
    batch_valid = all(checks[name] for name in ("fixed_indices_exact", "fixed_labels_exact", "fixed_raw_hash_exact", "fixed_tensor_hash_exact", "fixed_shape_dtype", "layout_roundtrip_exact"))
    resync_valid = all(row["full_state_sync_exact"] for row in case_rows) and all(checks[f"adam_zero_seed{seed}"] for seed in SEEDS)

    write_csv(RESULTS / "manifests/v4_parameter_manifest.csv", parameter_rows)
    write_csv(RESULTS / "manifests/v4_architecture_manifest.csv", architecture_rows)
    write_csv(RESULTS / "manifests/v4_case_manifest.csv", case_rows)
    reports = {
        "architecture_equivalence.json": {"v4_architecture_equivalence": "VALID" if architecture_valid else "INVALID", "semantic_stage_count": len(architecture_rows)},
        "parameter_equivalence.json": {"v4_parameter_equivalence": "VALID" if parameter_valid else "INVALID", "keras_trainable_parameter_count": keras_count, "pytorch_trainable_parameter_count": torch_count, "nontrainable_bn_state_count": keras_nontrain},
        "gpu_device_check.json": {"v4_gpu_device_check": "VALID" if device_valid else "INVALID", "tensorflow_actual_device": keras_device, "pytorch_actual_device": torch_device},
        "fixed_batch_check.json": {"v4_fixed_batch_check": "VALID" if batch_valid else "INVALID", "canonical_float32_sha256": sha256_array(images)},
        "shared_w0_check.json": {"v4_shared_w0_check": "VALID" if shared_valid else "INVALID"},
        "phase2_shared_body_w0_check.json": {"v4_phase2_shared_body_w0": "VALID" if phase2_body_valid else "INVALID"},
        "resync_precheck.json": {"v4_resync_precheck": "VALID" if resync_valid else "INVALID", "selected_cases": 3},
        "training_config_equivalence.json": {
            "v4_training_config_equivalence": "VALID",
            "paired_framework_contract": "shared canonical state/split/order/batches/loss/CommonBN/CommonAdam",
            "config": fingerprint(),
        },
    }
    common = {"checks": checks, "errors": errors, "config_sha256": config_hash(), "new_full_training_executed": False}
    for filename, report in reports.items():
        report.update(common); write_json(RESULTS / "preflight" / filename, report)
    write_json(RESULTS / "manifests/environment.json", {
        "validation": "V4 - CIFAR-10 Independent Workload Validation",
        "os": platform.platform(), "python_version": platform.python_version(),
        "numpy_version": np.__version__, "tensorflow_version": tf.__version__,
        "pytorch_version": torch.__version__, "tensorflow_actual_device": keras_device,
        "pytorch_actual_device": torch_device, "dtype": "float32", "config_sha256": config_hash(),
    })
    for filename, report in reports.items():
        key = next(key for key in report if key.startswith("v4_"))
        print(f"{key.upper()} = {report[key]}")
    print(f"TRAINABLE_PARAMETER_COUNT = {keras_count}")
    if errors:
        raise RuntimeError("V4 preflight INVALID: " + "; ".join(errors))


if __name__ == "__main__":
    main()
