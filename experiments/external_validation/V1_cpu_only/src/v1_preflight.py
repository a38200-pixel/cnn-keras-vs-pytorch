"""CPU device, fixed-batch, and exact state-sync gate for V1."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP07 = ROOT / "experiments/07_multistep_state_resynchronization"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP07))
sys.path.insert(0, str(HERE))

from common.controlled_data import tensor_hash
from common.trace_utils import write_csv, write_json
from cpu_trace_utils import CpuLayerTraceRunner, actual_device_check, environment_manifest
from probe_utils import fixed_probe_batch, load_checkpoint, require_preflight
from v1_config import (
    EXP07_RESULTS, EXP08_RESULTS, RESULTS, config_hash, selected_cases,
)


def _manifest_signature(row: dict[str, object]) -> tuple[object, ...]:
    flip = row["flip"] if isinstance(row["flip"], bool) else str(row["flip"]).lower() == "true"
    return (
        str(row["sample_id"]), int(row["label"]), str(row["tensor_hash"]),
        str(row["batch_tensor_hash"]), bool(flip), float(row["rotation_degrees"]),
        int(row["augmentation_seed"]), int(row["augmentation_epoch"]),
        str(row["dtype"]), str(row["shape"]),
    )


def validate() -> dict[str, object]:
    errors: list[str] = []
    checks: dict[str, bool] = {}

    def record(name: str, condition: bool, detail: object = "") -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {detail}")

    if config_hash() is None:
        raise RuntimeError("Missing V1 config hash")
    parent07 = require_preflight()
    parent08 = json.loads((EXP08_RESULTS / "summaries/diagnostic_summary.json").read_text(encoding="utf-8"))
    record("experiment07_valid", parent07.get("resync_precheck") == "VALID")
    record("experiment08_valid", parent08.get("status") == "VALID")

    devices = actual_device_check()
    record("tensorflow_actual_cpu", bool(devices["tensorflow_cpu_actual"]), devices)
    record("tensorflow_no_visible_gpu", devices["tensorflow_visible_gpu_count"] == 0, devices)
    record("pytorch_actual_cpu", bool(devices["pytorch_cpu_actual"]), devices)
    record("pytorch_mps_not_used", not bool(devices["pytorch_mps_tensor_used"]), devices)

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
    record("fixed_batch_float32", images.dtype == np.float32, images.dtype)
    record("fixed_batch_shape", images.shape == (32, 128, 128, 3), images.shape)
    record("fixed_labels_shape", labels.shape == (32,), labels.shape)

    runner = CpuLayerTraceRunner(images, labels)
    record("canonical_nhwc_nchw_roundtrip_exact", runner.input_exact)
    record("label_conversion_exact", runner.label_exact)

    v1_cases = selected_cases()
    exp08_manifest = list(csv.DictReader(
        (EXP08_RESULTS / "manifests/trace_case_manifest.csv").open(encoding="utf-8")
    ))
    record("selected_cases_exactly_reused", [case["case_id"] for case in v1_cases] == [row["case_id"] for row in exp08_manifest])
    case_rows = []
    for case in v1_cases:
        state, optimizer, step, metadata = load_checkpoint(
            int(case["seed"]), str(case["source_framework"]), str(case["checkpoint"]),
        )
        exact = runner.validate_resync(state, optimizer, step)
        record(f"full_state_sync_{case['case_id']}", bool(exact["all_exact"]))
        record(f"optimizer_step_{case['case_id']}", step == int(case["optimizer_step"]), step)
        case_rows.append({
            "case_id": case["case_id"], "seed": case["seed"],
            "checkpoint": case["checkpoint"], "optimizer_step": step,
            "anchor": case["anchor"], "source_framework": case["source_framework"],
            "source_checkpoint": metadata["checkpoint_name"],
            "source_state_hash": exact["model_state_sha256"] + ":" + exact["optimizer_state_sha256"],
            "experiment08_source_state_hash": case["experiment08_source_state_hash"],
            "probe_batch_hash": batch_hash, "config_sha256": config_hash(),
            "keras_device": devices["tensorflow_actual_device"],
            "pytorch_device": devices["pytorch_actual_device"],
            "full_state_sync_exact": exact["all_exact"],
        })
    record("selected_case_count", len(case_rows) == 9, len(case_rows))

    env_path = RESULTS / "manifests/cpu_environment.json"
    environment = environment_manifest()
    if env_path.exists():
        previous = json.loads(env_path.read_text(encoding="utf-8"))
        environment["timestamp_utc"] = previous["timestamp_utc"]
    write_json(env_path, environment)
    write_csv(RESULTS / "manifests/v1_case_manifest.csv", case_rows)

    device_valid = all(checks[name] for name in (
        "tensorflow_actual_cpu", "tensorflow_no_visible_gpu",
        "pytorch_actual_cpu", "pytorch_mps_not_used",
    ))
    # The resync gate covers the entire inherited-case contract, not merely the
    # model variables: parent validity, batch identity, canonical conversion,
    # case selection, state synchronization, and optimizer step must all pass.
    resync_valid = not errors
    status = "VALID" if not errors else "INVALID"
    device_report = {
        "validation": "V1_cpu_only", "cpu_only_device_check": "VALID" if device_valid else "INVALID",
        "devices": devices, "thread_settings": environment["thread_settings"],
        "config_sha256": config_hash(), "errors": [error for error in errors if "cpu" in error.lower() or "device" in error.lower()],
        "new_full_training_executed": False,
    }
    resync_report = {
        "validation": "V1_cpu_only", "v1_resync_precheck": "VALID" if resync_valid else "INVALID",
        "layer_trace_precheck": status, "checks": checks, "errors": errors,
        "selected_cases": len(case_rows), "fixed_probe_batch_hash": batch_hash,
        "input_exact": runner.input_exact, "label_exact": runner.label_exact,
        "config_sha256": config_hash(), "new_full_training_executed": False,
        "parent_checkpoints_modified": False,
    }
    write_json(RESULTS / "preflight/cpu_device_check.json", device_report)
    write_json(RESULTS / "preflight/resync_precheck.json", resync_report)
    print(f"CPU_ONLY_DEVICE_CHECK = {device_report['cpu_only_device_check']}")
    print(f"V1_RESYNC_PRECHECK = {resync_report['v1_resync_precheck']}")
    print(f"V1_SELECTED_CASES_READY = {str(not errors).upper()}")
    print(f"FIXED_BATCH_HASH = {batch_hash}")
    for error in errors:
        print(f"INVALID: {error}")
    return resync_report


if __name__ == "__main__":
    validate()
