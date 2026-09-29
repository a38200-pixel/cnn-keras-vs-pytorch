"""V2 architecture, parameter, GPU, batch, and initial-state gates."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP07 = ROOT / "experiments/07_multistep_state_resynchronization"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP07))
sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from common.controlled_data import tensor_hash
from common.reference_adam import CommonAdam
from common.state_resynchronization import states_exact
from common.trace_utils import write_csv, write_json
from probe_utils import fixed_probe_batch, require_preflight
from resnet18_canonical_state import (
    BN_BETA_NAMES, BN_GAMMA_NAMES, BN_MEAN_NAMES, BN_VARIANCE_NAMES, SPECS,
    TRAINABLE_NAMES, TRAINABLE_PARAMETER_COUNT, NON_TRAINABLE_STATE_COUNT,
    array_hash, create_initial_state, export_keras_state, export_torch_state,
    load_keras_state, load_torch_state, trainable_values,
)
from resnet18_keras import build_keras_resnet18
from resnet18_pytorch import build_torch_resnet18
from v2_config import EXP07_RESULTS, RESULTS, RUN_FULL_TRAINING, cases, config_hash


def _manifest_signature(row: dict[str, object]) -> tuple[object, ...]:
    flip = row["flip"] if isinstance(row["flip"], bool) else str(row["flip"]).lower() == "true"
    return (
        str(row["sample_id"]), int(row["label"]), str(row["tensor_hash"]),
        str(row["batch_tensor_hash"]), bool(flip), float(row["rotation_degrees"]),
        int(row["augmentation_seed"]), int(row["augmentation_epoch"]),
        str(row["dtype"]), str(row["shape"]),
    )


def _canonical_torch(value) -> np.ndarray:
    array = value.detach().cpu().numpy().copy()
    return array.transpose(0, 2, 3, 1) if array.ndim == 4 else array


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate() -> dict[str, object]:
    if RUN_FULL_TRAINING:
        raise RuntimeError("V2 safety guard violated: RUN_FULL_TRAINING must remain False")
    import tensorflow as tf
    import torch

    errors: list[str] = []
    checks: dict[str, bool] = {}

    def record(name: str, condition: bool, detail: object = "") -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {detail}")

    parent = require_preflight()
    record("experiment07_preflight_valid", parent.get("resync_precheck") == "VALID")

    images, labels, reconstructed = fixed_probe_batch(write_manifest=False)
    batch_hash = tensor_hash(images)
    with (EXP07_RESULTS / "probe_batch_manifest.csv").open(encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    record("fixed_batch_row_count", len(persisted) == len(reconstructed) == 32)
    record(
        "fixed_batch_manifest_exact",
        [_manifest_signature(row) for row in persisted]
        == [_manifest_signature(row) for row in reconstructed],
    )
    record("fixed_batch_hash_exact", {row["batch_tensor_hash"] for row in persisted} == {batch_hash})
    record("fixed_batch_shape", images.shape == (32, 128, 128, 3), images.shape)
    record("fixed_batch_dtype", images.dtype == np.float32, images.dtype)
    record("fixed_labels_shape", labels.shape == (32,), labels.shape)

    physical_gpus = tf.config.list_physical_devices("GPU")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    record("tensorflow_visible_gpu", len(physical_gpus) > 0, physical_gpus)
    record("pytorch_mps_available", device.type == "mps", device)

    with tf.device("/GPU:0"):
        keras_model = build_keras_resnet18()
    torch_model = build_torch_resnet18().to(device)

    parameter_rows = []
    case_rows = []
    for case in cases():
        seed = int(case["seed"])
        canonical = create_initial_state(seed)
        load_keras_state(keras_model, canonical)
        load_torch_state(torch_model, canonical)
        keras_state = export_keras_state(keras_model)
        torch_state = export_torch_state(torch_model)
        record(f"semantic_state_keys_seed{seed}", set(keras_state) == set(torch_state) == set(canonical))
        record(f"canonical_w0_keras_seed{seed}", states_exact(canonical, keras_state))
        record(f"canonical_w0_pytorch_seed{seed}", states_exact(canonical, torch_state))
        record(f"paired_w0_seed{seed}", states_exact(keras_state, torch_state))
        record(f"bn_gamma_seed{seed}", states_exact(keras_state, torch_state, BN_GAMMA_NAMES))
        record(f"bn_beta_seed{seed}", states_exact(keras_state, torch_state, BN_BETA_NAMES))
        record(f"bn_mean_seed{seed}", states_exact(keras_state, torch_state, BN_MEAN_NAMES))
        record(f"bn_variance_seed{seed}", states_exact(keras_state, torch_state, BN_VARIANCE_NAMES))
        keras_adam = CommonAdam(trainable_values(canonical))
        torch_adam = CommonAdam(trainable_values(canonical))
        record(f"adam_m_seed{seed}", states_exact(keras_adam.m, torch_adam.m, TRAINABLE_NAMES))
        record(f"adam_v_seed{seed}", states_exact(keras_adam.v, torch_adam.v, TRAINABLE_NAMES))
        record(f"adam_step_seed{seed}", keras_adam.step == torch_adam.step == 0)
        for spec in SPECS:
            name = spec.semantic_name
            parameter_rows.append({
                "seed": seed, "semantic_name": name,
                "keras_name": spec.keras_name, "pytorch_name": spec.pytorch_name,
                "canonical_shape": json.dumps(spec.shape),
                "keras_shape": json.dumps(keras_state[name].shape),
                "pytorch_shape": json.dumps(torch_state[name].shape),
                "dtype": str(canonical[name].dtype), "trainable": spec.trainable,
                "canonical_hash": array_hash(canonical[name]),
                "keras_canonical_hash": array_hash(keras_state[name]),
                "pytorch_canonical_hash": array_hash(torch_state[name]),
                "exact_equal": bool(
                    np.array_equal(canonical[name], keras_state[name])
                    and np.array_equal(canonical[name], torch_state[name])
                ),
            })
        case_rows.append({
            **case, "canonical_state_hash": hashlib_state(canonical),
            "probe_batch_hash": batch_hash, "config_sha256": config_hash(),
            "keras_device": "GPU:0", "pytorch_device": str(device),
            "full_state_sync_exact": bool(
                states_exact(keras_state, torch_state)
                and states_exact(keras_adam.m, torch_adam.m)
                and states_exact(keras_adam.v, torch_adam.v)
                and keras_adam.step == torch_adam.step == 0
            ),
        })

    keras_trainable_count = int(sum(np.prod(variable.shape) for variable in keras_model.trainable_variables))
    torch_trainable_count = int(sum(parameter.numel() for parameter in torch_model.parameters()))
    keras_nontrainable_count = int(sum(np.prod(variable.shape) for variable in keras_model.non_trainable_variables))
    torch_nontrainable_count = int(sum(
        tensor.numel() for name, tensor in torch_model.state_dict().items()
        if name.endswith("running_mean") or name.endswith("running_var")
    ))
    record(
        "trainable_parameter_count_exact",
        keras_trainable_count == torch_trainable_count == TRAINABLE_PARAMETER_COUNT,
        (keras_trainable_count, torch_trainable_count, TRAINABLE_PARAMETER_COUNT),
    )
    record(
        "nontrainable_bn_state_count_exact",
        keras_nontrainable_count == torch_nontrainable_count == NON_TRAINABLE_STATE_COUNT,
        (keras_nontrainable_count, torch_nontrainable_count, NON_TRAINABLE_STATE_COUNT),
    )
    record("parameter_manifest_all_exact", all(row["exact_equal"] for row in parameter_rows))

    canonical42 = create_initial_state(42)
    load_keras_state(keras_model, canonical42)
    load_torch_state(torch_model, canonical42)
    with tf.device("/GPU:0"):
        keras_model(tf.zeros((1, 128, 128, 3), tf.float32), training=False)
    torch_model.eval()
    with torch.no_grad():
        torch_model(torch.zeros((1, 3, 128, 128), dtype=torch.float32, device=device))
    keras_trace = keras_model.last_trace
    torch_trace = torch_model.last_trace
    record("semantic_trace_keys_exact", set(keras_trace) == set(torch_trace))
    architecture_rows = []
    for order, name in enumerate(keras_trace):
        keras_value = np.asarray(keras_trace[name].numpy())
        torch_value = _canonical_torch(torch_trace[name])
        architecture_rows.append({
            "stage_order": order, "semantic_stage": name,
            "keras_shape": json.dumps(keras_value.shape),
            "pytorch_canonical_shape": json.dumps(torch_value.shape),
            "keras_dtype": str(keras_value.dtype), "pytorch_dtype": str(torch_value.dtype),
            "shape_exact": keras_value.shape == torch_value.shape,
        })
    record("all_architecture_shapes_exact", all(row["shape_exact"] for row in architecture_rows))
    required_downsample = {
        f"stage{stage}.block1.{branch}"
        for stage in (2, 3, 4)
        for branch in ("main", "shortcut.output", "add")
    }
    record("downsample_branch_coverage", required_downsample <= set(keras_trace))

    keras_conv_device = str(keras_model.last_trace["stem.conv"].device)
    torch_conv_device = str(torch_model.last_trace["stem.conv"].device)
    record("tensorflow_resnet_conv_on_gpu", "GPU:0" in keras_conv_device.upper(), keras_conv_device)
    record("pytorch_resnet_conv_on_mps", torch_conv_device == "mps:0", torch_conv_device)

    px = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device)
    restored = px.detach().cpu().numpy().transpose(0, 2, 3, 1)
    record("nchw_roundtrip_exact", np.array_equal(images, restored))
    record("label_exact", np.array_equal(labels, labels.astype(np.int64).astype(np.int32)))

    architecture_valid = all(checks[name] for name in (
        "semantic_trace_keys_exact", "all_architecture_shapes_exact", "downsample_branch_coverage",
    ))
    parameter_valid = all(
        value for name, value in checks.items()
        if name.startswith(("semantic_state_keys_", "canonical_w0_", "paired_w0_", "bn_", "adam_"))
    ) and checks["trainable_parameter_count_exact"] and checks["nontrainable_bn_state_count_exact"] \
        and checks["parameter_manifest_all_exact"]
    device_valid = all(checks[name] for name in (
        "tensorflow_visible_gpu", "pytorch_mps_available",
        "tensorflow_resnet_conv_on_gpu", "pytorch_resnet_conv_on_mps",
    ))
    batch_valid = all(checks[name] for name in (
        "fixed_batch_row_count", "fixed_batch_manifest_exact", "fixed_batch_hash_exact",
        "fixed_batch_shape", "fixed_batch_dtype", "fixed_labels_shape",
        "nchw_roundtrip_exact", "label_exact",
    ))
    resync_valid = all(row["full_state_sync_exact"] for row in case_rows)

    write_csv(RESULTS / "manifests/resnet18_parameter_manifest.csv", parameter_rows)
    write_csv(RESULTS / "manifests/resnet18_architecture_manifest.csv", architecture_rows)
    write_csv(RESULTS / "manifests/v2_case_manifest.csv", case_rows)

    try:
        metal_version = importlib.metadata.version("tensorflow-metal")
    except importlib.metadata.PackageNotFoundError:
        metal_version = None
    environment = {
        "validation": "V2 - ResNet18 + BatchNorm Architecture Validation",
        "os": platform.platform(), "architecture": platform.machine(),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "tensorflow_version": tf.__version__, "tensorflow_metal_version": metal_version,
        "tensorflow_visible_gpus": [item.name for item in physical_gpus],
        "tensorflow_actual_resnet_conv_device": keras_conv_device,
        "pytorch_version": torch.__version__, "pytorch_mps_available": torch.backends.mps.is_available(),
        "pytorch_actual_resnet_conv_device": torch_conv_device,
        "pytorch_mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "dtype": "float32", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit(), "config_sha256": config_hash(),
        "pretrained": False, "run_full_training": False,
    }
    write_json(RESULTS / "manifests/environment.json", environment)

    reports = {
        "architecture_equivalence.json": {
            "resnet18_architecture_equivalence": "VALID" if architecture_valid else "INVALID",
            "checks": {name: value for name, value in checks.items() if name in {
                "semantic_trace_keys_exact", "all_architecture_shapes_exact", "downsample_branch_coverage",
            }}, "semantic_stage_count": len(architecture_rows), "errors": errors,
            "config_sha256": config_hash(), "new_full_training_executed": False,
        },
        "parameter_equivalence.json": {
            "resnet18_parameter_equivalence": "VALID" if parameter_valid else "INVALID",
            "keras_trainable_parameter_count": keras_trainable_count,
            "pytorch_trainable_parameter_count": torch_trainable_count,
            "canonical_trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "keras_nontrainable_bn_state_count": keras_nontrainable_count,
            "pytorch_nontrainable_bn_state_count": torch_nontrainable_count,
            "semantic_state_arrays_per_seed": len(SPECS), "all_w0_exact": checks["parameter_manifest_all_exact"],
            "config_sha256": config_hash(), "errors": errors, "new_full_training_executed": False,
        },
        "gpu_device_check.json": {
            "v2_gpu_device_check": "VALID" if device_valid else "INVALID",
            "tensorflow_actual_device": keras_conv_device, "pytorch_actual_device": torch_conv_device,
            "tensorflow_visible_gpu_count": len(physical_gpus),
            "pytorch_mps_available": torch.backends.mps.is_available(),
            "silent_mps_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            "config_sha256": config_hash(), "errors": errors, "new_full_training_executed": False,
        },
        "resync_precheck.json": {
            "v2_fixed_batch_check": "VALID" if batch_valid else "INVALID",
            "v2_resync_precheck": "VALID" if resync_valid else "INVALID",
            "fixed_probe_batch_hash": batch_hash, "selected_cases": len(case_rows),
            "checks": checks, "errors": errors, "config_sha256": config_hash(),
            "new_full_training_executed": False, "parent_assets_modified": False,
        },
    }
    for filename, report in reports.items():
        write_json(RESULTS / "preflight" / filename, report)

    print(f"RESNET18_ARCHITECTURE_EQUIVALENCE = {reports['architecture_equivalence.json']['resnet18_architecture_equivalence']}")
    print(f"RESNET18_PARAMETER_EQUIVALENCE = {reports['parameter_equivalence.json']['resnet18_parameter_equivalence']}")
    print(f"V2_GPU_DEVICE_CHECK = {reports['gpu_device_check.json']['v2_gpu_device_check']}")
    print(f"V2_FIXED_BATCH_CHECK = {reports['resync_precheck.json']['v2_fixed_batch_check']}")
    print(f"V2_RESYNC_PRECHECK = {reports['resync_precheck.json']['v2_resync_precheck']}")
    print(f"TRAINABLE_PARAMETER_COUNT = {keras_trainable_count}")
    print(f"FIXED_BATCH_HASH = {batch_hash}")
    for error in errors:
        print(f"INVALID: {error}")
    return reports["resync_precheck.json"]


def hashlib_state(state: dict[str, np.ndarray]) -> str:
    digest = __import__("hashlib").sha256()
    for name in sorted(state):
        digest.update(name.encode() + b"\0")
        digest.update(np.ascontiguousarray(state[name]).tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    validate()
