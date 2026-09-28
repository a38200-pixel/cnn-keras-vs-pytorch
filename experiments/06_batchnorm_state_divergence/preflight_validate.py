"""Structural and diagnostic gate for Experiment 06 Common BN training."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_batchnorm import (
    COMMON_BN_SEMANTICS_ID, build_keras_common_bn_model, build_torch_common_bn_model,
    keras_uses_native_batchnorm, torch_uses_native_batchnorm,
)
from common.controlled_checkpoint import verify_checkpoint
from common.controlled_data import canonical_rows
from common.controlled_initialization import (
    export_keras_weights, export_torch_weights, load_keras_weights, load_torch_weights,
)
from experiment_config import DIAGNOSTIC_STEPS, RESULTS, SEEDS, TRAINABLE_PARAMETERS, config_hash, fingerprint


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate() -> dict[str, object]:
    import tensorflow as tf
    import torch

    errors, checks = [], {}
    def record(name, condition, detail=""):
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {detail}")

    parent_results = ROOT / "experiments/05_gradient_optimizer_divergence/results"
    parent_summary = json.loads((parent_results / "controlled_comparison_summary.json").read_text(encoding="utf-8"))
    record("parent_experiment05_valid", parent_summary.get("status") == "VALID")
    for framework in ("keras", "pytorch"):
        rows = _read_csv(parent_results / f"{framework}_common_adam_results.csv")
        record(
            f"parent_{framework}_completed",
            sorted(int(row["seed"]) for row in rows) == list(SEEDS)
            and all(int(row["epochs_trained"]) == 30 and int(row["optimizer_steps"]) == 9630 for row in rows),
        )
    counts = {split: len(canonical_rows(split)) for split in ("train", "val", "test")}
    record("dataset", counts == {"train": 10251, "val": 2194, "test": 2203}, counts)
    config = fingerprint()
    parent_config_path = ROOT / "experiments/05_gradient_optimizer_divergence/experiment_config.py"
    parent_spec = importlib.util.spec_from_file_location("experiment05_config_for_bn_preflight", parent_config_path)
    if parent_spec is None or parent_spec.loader is None:
        raise RuntimeError("Cannot load Experiment 05 config")
    parent_module = importlib.util.module_from_spec(parent_spec)
    parent_spec.loader.exec_module(parent_module)
    parent_config = parent_module.fingerprint()
    excluded = {"experiment", "parent_experiment", "batchnorm"}
    unchanged_05 = {key: value for key, value in config.items() if key not in excluded}
    unchanged_parent = {key: value for key, value in parent_config.items() if key not in excluded}
    record("fixed_schedule", config["epochs"] == 30 and config["batch_size"] == 32)
    record(
        "batchnorm_only_change",
        config["parent_experiment"] == "05_gradient_optimizer_divergence"
        and config["optimizer_implementation"] == "common_reference_adam_float32_v1"
        and config["batchnorm"]["implementation"] == COMMON_BN_SEMANTICS_ID,
    )
    record("all_non_bn_controls_match_experiment05", unchanged_05 == unchanged_parent)
    record("tensorflow_gpu", bool(tf.config.list_physical_devices("GPU")))
    record("pytorch_mps", torch.backends.mps.is_available())
    km, pm = build_keras_common_bn_model(), build_torch_common_bn_model()
    kc = sum(int(np.prod(variable.shape)) for variable in km.trainable_variables)
    pc = sum(parameter.numel() for parameter in pm.parameters() if parameter.requires_grad)
    record("parameter_count", kc == pc == TRAINABLE_PARAMETERS, f"{kc}/{pc}")
    record("keras_native_batchnorm_absent", not keras_uses_native_batchnorm(km))
    record("pytorch_native_batchnorm_absent", not torch_uses_native_batchnorm(pm))

    roundtrip = []
    for seed in SEEDS:
        first_path = RESULTS / "first_step" / f"seed{seed}_summary.json"
        early_path = RESULTS / "early_steps" / f"seed{seed}_step_summary.csv"
        if not first_path.exists() or not early_path.exists():
            errors.append(f"Missing diagnostic seed={seed}")
            continue
        first = json.loads(first_path.read_text(encoding="utf-8"))
        record(
            f"first_step_seed{seed}",
            all((
                first["w0_exact"], first["input_exact"], first["labels_exact"],
                first["bn_initial_state_exact"], first["same_common_bn_semantics"],
                first["native_batchnorm_absent"], first["same_common_adam_implementation"],
                first["common_adam_step"] == 1, first["gradient_flow_valid"],
                not first["nan_or_inf"], first["config_sha256"] == config_hash(),
            )),
        )
        early = _read_csv(early_path)
        record(
            f"early_steps_seed{seed}",
            [int(row["step"]) for row in early] == list(DIAGNOSTIC_STEPS)
            and float(early[0]["global_weight_relative_l2"]) == 0
            and all(row["config_sha256"] == config_hash() for row in early),
        )
        for step in (0, 1):
            folder = RESULTS / "early_steps/checkpoints" / f"seed{seed}" / f"step{step:03d}"
            for framework in ("keras", "pytorch"):
                state, optimizer = verify_checkpoint(
                    folder / f"{framework}_state.npz",
                    folder / f"{framework}_optimizer.npz",
                    folder / f"{framework}_metadata.json",
                    expected_config_hash=config_hash(),
                )
                if framework == "keras":
                    tf.keras.backend.clear_session()
                    model = build_keras_common_bn_model()
                    load_keras_weights(model, state)
                    restored = export_keras_weights(model)
                else:
                    model = build_torch_common_bn_model()
                    load_torch_weights(model, state)
                    restored = export_torch_weights(model)
                exact = set(state) == set(restored) and all(np.array_equal(state[name], restored[name]) for name in state)
                valid_optimizer = not optimizer if step == 0 else len(optimizer) == 32
                record(f"roundtrip_{framework}_{seed}_{step}", exact and valid_optimizer)
                roundtrip.append({
                    "seed": seed, "step": step, "framework": framework,
                    "model_exact": exact, "optimizer_keys": len(optimizer),
                })
    record(
        "no_full_training_results",
        not any((RESULTS / f"{framework}_common_bn_results.csv").exists() for framework in ("keras", "pytorch")),
    )
    status = "VALID" if not errors else "NOT_READY"
    report = {
        "experiment": "06_batchnorm_state_divergence", "common_bn_precheck": status,
        "full_training_ready": not errors, "config_sha256": config_hash(),
        "checks": checks, "errors": errors, "dataset_counts": counts,
        "checkpoint_roundtrip": roundtrip, "full_training_executed": False,
    }
    path = RESULTS / "preflight/preflight_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"COMMON_BN_PRECHECK = {status}")
    print(f"FULL_TRAINING_READY = {str(not errors).upper()}")
    for error in errors:
        print(f"NOT READY: {error}")
    return report


if __name__ == "__main__":
    validate()
