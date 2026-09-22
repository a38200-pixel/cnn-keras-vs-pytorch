"""Read-only structural gate for Common Adam controlled training."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_checkpoint import verify_checkpoint
from common.controlled_data import canonical_rows
from common.controlled_initialization import (
    build_keras_model, build_torch_model, export_keras_weights, export_torch_weights,
    load_keras_weights, load_torch_weights,
)
from experiment_config import DIAGNOSTIC_STEPS, RESULTS, SEEDS, TRAINABLE_PARAMETERS, config_hash, fingerprint


def validate() -> dict[str, object]:
    import tensorflow as tf
    import torch

    errors = []; checks = {}
    def record(name, condition, detail=""):
        checks[name] = bool(condition)
        if not condition: errors.append(f"{name}: {detail}")

    parent = json.loads((ROOT / "experiments/04_common_initialization_controlled_training/results/preflight/preflight_summary.json").read_text(encoding="utf-8"))
    record("parent_controlled_precheck", parent.get("controlled_precheck") == "VALID")
    counts = {split: len(canonical_rows(split)) for split in ("train", "val", "test")}
    record("dataset", counts == {"train": 10251, "val": 2194, "test": 2203}, counts)
    record("fixed_schedule", fingerprint()["epochs"] == 30 and fingerprint()["batch_size"] == 32)
    record("optimizer_only_change", fingerprint()["parent_experiment"] == "04_common_initialization_controlled_training"
           and fingerprint()["optimizer_implementation"] == "common_reference_adam_float32_v1")
    record("tensorflow_gpu", bool(tf.config.list_physical_devices("GPU")))
    record("pytorch_mps", torch.backends.mps.is_available())
    km, pm = build_keras_model(), build_torch_model()
    kc = sum(int(np.prod(v.shape)) for v in km.trainable_variables)
    pc = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    record("parameter_count", kc == pc == TRAINABLE_PARAMETERS, f"{kc}/{pc}")

    roundtrip = []
    for seed in SEEDS:
        summary_path = RESULTS / "first_step" / f"seed{seed}_summary.json"
        early_path = RESULTS / "early_steps" / f"seed{seed}_step_summary.csv"
        if not summary_path.exists() or not early_path.exists():
            errors.append(f"Missing diagnostic seed={seed}"); continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record(f"first_step_seed{seed}", all((summary["w0_exact"], summary["input_exact"], summary["conv1_exact"],
              summary["same_common_adam_implementation"], summary["common_adam_step"] == 1,
              not summary["nan_or_inf"], summary["config_sha256"] == config_hash())))
        with early_path.open(encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
        record(f"early_steps_seed{seed}", [int(row["step"]) for row in rows] == list(DIAGNOSTIC_STEPS)
               and float(rows[0]["global_weight_relative_l2"]) == 0
               and all(row["config_sha256"] == config_hash() for row in rows))
        for step in (0, 1):
            folder = RESULTS / "early_steps/checkpoints" / f"seed{seed}" / f"step{step:03d}"
            for framework in ("keras", "pytorch"):
                state, optimizer = verify_checkpoint(
                    folder / f"{framework}_state.npz", folder / f"{framework}_optimizer.npz",
                    folder / f"{framework}_metadata.json", expected_config_hash=config_hash(),
                )
                if framework == "keras":
                    tf.keras.backend.clear_session(); model = build_keras_model(); load_keras_weights(model, state); restored = export_keras_weights(model)
                else:
                    model = build_torch_model(); load_torch_weights(model, state); restored = export_torch_weights(model)
                exact = set(state) == set(restored) and all(np.array_equal(state[name], restored[name]) for name in state)
                # The actual mapping is validated by first-step/early-step execution; only exact checkpoint round-trip gates here.
                record(f"roundtrip_{framework}_{seed}_{step}", exact and (not optimizer if step == 0 else len(optimizer) == 32))
                roundtrip.append({"seed": seed, "step": step, "framework": framework, "model_exact": exact, "optimizer_keys": len(optimizer)})
    record("no_full_training_results", not any((RESULTS / f"{fw}_common_adam_results.csv").exists() for fw in ("keras", "pytorch")))
    status = "VALID" if not errors else "NOT_READY"
    report = {
        "experiment": "05_gradient_optimizer_divergence", "common_adam_precheck": status,
        "full_training_ready": not errors, "config_sha256": config_hash(),
        "checks": checks, "errors": errors, "dataset_counts": counts,
        "checkpoint_roundtrip": roundtrip, "full_training_executed": False,
    }
    path = RESULTS / "preflight/preflight_summary.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"COMMON_ADAM_PRECHECK = {status}")
    print(f"FULL_TRAINING_READY = {str(not errors).upper()}")
    if errors:
        for error in errors: print(f"NOT READY: {error}")
    return report


if __name__ == "__main__":
    validate()
