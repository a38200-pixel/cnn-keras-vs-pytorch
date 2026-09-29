"""Validate the nine Experiment 08 cases without running a training step."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP07 = ROOT / "experiments/07_multistep_state_resynchronization"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP07))

from common.controlled_data import tensor_hash
from common.trace_utils import write_csv, write_json
from layer_config import (
    PARENT07_RESULTS, RESULTS, config_hash, select_epoch1_control, selected_cases,
)
from probe_utils import ProbeRunner, fixed_probe_batch, load_checkpoint, require_preflight


def validate() -> dict[str, object]:
    errors: list[str] = []
    checks: dict[str, bool] = {}

    def record(name: str, condition: bool, detail: object = "") -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {detail}")

    parent = require_preflight()
    record("experiment07_preflight_valid", parent.get("resync_precheck") == "VALID")

    images, labels, generated_manifest = fixed_probe_batch(write_manifest=False)
    batch_hash = tensor_hash(images)
    with (PARENT07_RESULTS / "probe_batch_manifest.csv").open(encoding="utf-8") as handle:
        persisted_manifest = list(csv.DictReader(handle))
    persisted_hashes = {row["batch_tensor_hash"] for row in persisted_manifest}
    record("experiment07_fixed_batch_hash_exact", persisted_hashes == {batch_hash}, persisted_hashes)
    record(
        "experiment07_sample_order_exact",
        [row["sample_id"] for row in persisted_manifest]
        == [str(row["sample_id"]) for row in generated_manifest],
    )
    record("fixed_batch_shape", images.shape == (32, 128, 128, 3), images.shape)
    record("fixed_batch_dtype", images.dtype == np.float32, images.dtype)
    record("fixed_labels_shape", labels.shape == (32,), labels.shape)

    runner = ProbeRunner(images, labels)
    manifest_rows = []
    for case in selected_cases():
        state, optimizer, step, metadata = load_checkpoint(
            int(case["seed"]), str(case["source_framework"]), str(case["checkpoint"]),
        )
        exact = runner.validate_resync(state, optimizer, step)
        record(f"full_state_sync_{case['case_id']}", bool(exact["all_exact"]))
        manifest_rows.append({
            "case_id": case["case_id"],
            "seed": case["seed"],
            "checkpoint": case["checkpoint"],
            "optimizer_step": step,
            "anchor": case["anchor"],
            "source_checkpoint": metadata["checkpoint_name"],
            "source_state_hash": exact["model_state_sha256"] + ":" + exact["optimizer_state_sha256"],
            "probe_batch_hash": batch_hash,
            "config_sha256": config_hash(),
            "keras_device": "GPU:0",
            "pytorch_device": str(runner.device),
        })

    low_seed, low_scores = select_epoch1_control()
    record("selected_case_count", len(manifest_rows) == 9, len(manifest_rows))
    record("epoch1_low_divergence_seed_is_data_driven", low_seed == min(low_scores, key=low_scores.get))
    write_csv(RESULTS / "manifests/trace_case_manifest.csv", manifest_rows)

    status = "VALID" if not errors else "INVALID"
    report = {
        "experiment": "08_layer_by_layer_trajectory",
        "layer_trace_precheck": status,
        "selected_cases_ready": not errors,
        "config_sha256": config_hash(),
        "checks": checks,
        "errors": errors,
        "selected_case_count": len(manifest_rows),
        "epoch1_control": {
            "selection_rule": "minimum mean K/P-anchor synchronized post_weight_relative_l2",
            "selected_seed": low_seed,
            "scores": {str(key): value for key, value in low_scores.items()},
        },
        "fixed_probe_batch_hash": batch_hash,
        "experiment07_fixed_probe_batch_hash": next(iter(persisted_hashes)) if len(persisted_hashes) == 1 else None,
        "full_state_sync_exact": False,
        "new_full_training_executed": False,
        "parent_checkpoints_modified": False,
    }
    report["full_state_sync_exact"] = all(
        value for name, value in checks.items() if name.startswith("full_state_sync_")
    )
    write_json(RESULTS / "preflight/layer_trace_precheck.json", report)
    print(f"LAYER_TRACE_PRECHECK = {status}")
    print(f"SELECTED_CASES_READY = {str(not errors).upper()}")
    print(f"EPOCH1_CONTROL_SEED = {low_seed}")
    print(f"FIXED_BATCH_HASH = {batch_hash}")
    for error in errors:
        print(f"INVALID: {error}")
    return report


if __name__ == "__main__":
    validate()
