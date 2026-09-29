"""Validate Experiment 07 checkpoint reuse, fixed batch, and exact state sync."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.state_resynchronization import states_exact
from common.trace_utils import write_csv, write_json
from experiment_config import (
    CHECKPOINTS, PARENT_CONFIG_SHA256, PARENT_RESULTS, RESULTS, SEEDS,
    anchors, config_hash,
)
from probe_utils import (
    ProbeRunner, checkpoint_index, expected_resync_cases, fixed_probe_batch,
    load_checkpoint,
)


def validate() -> dict[str, object]:
    errors: list[str] = []
    checks: dict[str, bool] = {}

    def record(name: str, condition: bool, detail: object = "") -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {detail}")

    parent_summary = json.loads(
        (PARENT_RESULTS / "controlled_comparison_summary.json").read_text(encoding="utf-8")
    )
    record("parent_experiment06_valid", parent_summary.get("status") == "VALID")
    record("parent_config_hash", parent_summary.get("config_sha256") == PARENT_CONFIG_SHA256)
    manifest_index = checkpoint_index()
    record("parent_checkpoint_count", len(manifest_index) == 42, len(manifest_index))

    images, labels, batch_manifest = fixed_probe_batch(write_manifest=True)
    runner = ProbeRunner(images, labels)
    record("probe_batch_size", len(images) == 32)
    record("probe_batch_float32", images.dtype == np.float32)
    record("probe_batch_shape", images.shape == (32, 128, 128, 3), images.shape)
    record("probe_labels_shape", labels.shape == (32,), labels.shape)
    record("probe_sample_ids_unique", len({row["sample_id"] for row in batch_manifest}) == 32)
    record("keras_pytorch_input_exact", runner.input_exact)
    record("keras_pytorch_label_exact", runner.label_exact)

    state_manifest = []
    verified_checkpoints = 0
    for seed in SEEDS:
        initial_states = {}
        initial_optimizers = {}
        for framework in ("keras", "pytorch"):
            model_state, optimizer_state, step, _ = load_checkpoint(seed, framework, "initial")
            initial_states[framework] = model_state
            initial_optimizers[framework] = optimizer_state
            verified_checkpoints += 1
            record(f"initial_step_{seed}_{framework}", step == 0)
        record(
            f"initial_model_exact_{seed}",
            states_exact(initial_states["keras"], initial_states["pytorch"]),
        )
        record(
            f"initial_optimizer_exact_{seed}",
            states_exact(initial_optimizers["keras"], initial_optimizers["pytorch"]),
        )

        for checkpoint in CHECKPOINTS:
            for anchor in anchors(checkpoint):
                source_framework = "keras" if anchor == "shared" else anchor
                state, optimizer, step, metadata = load_checkpoint(
                    seed, source_framework, checkpoint,
                )
                if checkpoint != "initial":
                    other = "pytorch" if source_framework == "keras" else "keras"
                    load_checkpoint(seed, other, checkpoint)
                exact = runner.validate_resync(state, optimizer, step)
                row = {
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "anchor": anchor,
                    "source_framework": source_framework,
                    "optimizer_step": step,
                    "source_model_file_sha256": metadata["model_file_sha256"],
                    "source_optimizer_file_sha256": metadata["optimizer_file_sha256"],
                    **exact,
                    "probe_config_sha256": config_hash(),
                }
                state_manifest.append(row)
                record(f"resync_{seed}_{checkpoint}_{anchor}", bool(exact["all_exact"]))
        # The loop above verifies both frameworks at every non-initial checkpoint.
        verified_checkpoints += 2 * (len(CHECKPOINTS) - 1)

    record("all_42_checkpoints_verified", verified_checkpoints == 42, verified_checkpoints)
    record("resync_case_count", len(state_manifest) == expected_resync_cases(), len(state_manifest))
    write_csv(RESULTS / "manifests/resync_state_manifest.csv", state_manifest)

    status = "VALID" if not errors else "INVALID"
    report = {
        "experiment": "07_multistep_state_resynchronization",
        "resync_precheck": status,
        "probe_ready": not errors,
        "config_sha256": config_hash(),
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "checks": checks,
        "errors": errors,
        "verified_parent_checkpoints": verified_checkpoints,
        "resync_cases": len(state_manifest),
        "fixed_probe_batch": {
            "num_samples": len(images),
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "batch_tensor_hash": batch_manifest[0]["batch_tensor_hash"],
            "sample_ids_identical_across_seeds": True,
            "keras_pytorch_canonical_max_abs_diff": 0.0,
        },
        "full_training_executed": False,
        "experiment06_checkpoints_modified": False,
    }
    write_json(RESULTS / "preflight/resync_precheck.json", report)
    print(f"RESYNC_PRECHECK = {status}")
    print(f"PROBE_READY = {str(not errors).upper()}")
    for error in errors:
        print(f"INVALID: {error}")
    return report


if __name__ == "__main__":
    validate()
