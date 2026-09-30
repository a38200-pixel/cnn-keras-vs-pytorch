"""Post-process completed V4 Stage B artifacts; never starts training."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from common.trace_utils import compare, write_csv, write_json
from cifar10_canonical_state import TRAINABLE_NAMES
from v4_config import (
    EPOCH_SNAPSHOT_EPOCHS, PHASE2_RESULTS, RESULTS, SEEDS,
    STEP_SNAPSHOT_STEPS, TOTAL_STEPS, config_hash,
)


def _read_csv(path):
    with path.open() as handle: return list(csv.DictReader(handle))


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive: return {name: archive[name].copy() for name in archive.files}


def _relative(left, right, names):
    a = np.concatenate([left[name].ravel() for name in names]); b = np.concatenate([right[name].ravel() for name in names])
    return compare(a, b)["relative_l2_error"]


def _distance_row(seed, checkpoint, global_step, ks, ps):
    row = {"seed": seed, "checkpoint": checkpoint, "global_step": global_step,
           "v4_keras_pytorch_global_weight_rel_l2": _relative(ks, ps, TRAINABLE_NAMES)}
    for index in range(1, 5):
        row[f"v4_conv{index}_rel_l2"] = _relative(ks, ps, [f"conv{index}/kernel"])
        row[f"v4_bn{index}_rel_l2"] = _relative(ks, ps, [f"bn{index}/gamma", f"bn{index}/beta"])
        row[f"v4_bn{index}_running_mean_rel_l2"] = _relative(ks, ps, [f"bn{index}/mean"])
        row[f"v4_bn{index}_running_variance_rel_l2"] = _relative(ks, ps, [f"bn{index}/variance"])
    row["v4_fc128_rel_l2"] = _relative(ks, ps, ["fc128/kernel", "fc128/bias"])
    row["v4_classifier_rel_l2"] = _relative(ks, ps, ["classifier/kernel", "classifier/bias"])
    return row


def _snapshot(framework, seed, name):
    return _read_npz(RESULTS / "snapshots" / framework / f"seed{seed}" / name / "model.npz")


def _validate_snapshot(framework, seed, name, checkpoint_type, epoch, global_step):
    folder = RESULTS / "snapshots" / framework / f"seed{seed}" / name
    metadata = json.loads((folder / "metadata.json").read_text())
    expected = {
        "framework": framework, "seed": seed, "checkpoint_name": name,
        "checkpoint_type": checkpoint_type, "epoch": epoch,
        "global_step": global_step, "common_adam_step": global_step,
        "config_sha256": config_hash(),
    }
    mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatched:
        raise RuntimeError(f"Invalid V4 snapshot metadata {framework} seed{seed} {name}: {mismatched}")
    return _snapshot(framework, seed, name)


def main() -> None:
    histories, tests = {}, {}
    for seed in SEEDS:
        for framework in ("keras", "pytorch"):
            folder = RESULTS / "training" / framework / f"seed{seed}"
            history = _read_csv(folder / "history.csv")
            if len(history) != 30 or int(history[-1]["global_step_end"]) != TOTAL_STEPS:
                raise RuntimeError(f"Incomplete V4 training run: {framework} seed{seed}")
            expected_steps = [epoch * 1_407 for epoch in range(1, 31)]
            if [int(row["global_step_end"]) for row in history] != expected_steps:
                raise RuntimeError(f"Invalid epoch/global-step history: {framework} seed{seed}")
            if any(str(row["nan_inf"]).lower() not in ("false", "0") for row in history):
                raise RuntimeError(f"NaN/Inf flag in V4 history: {framework} seed{seed}")
            histories[(seed, framework)] = history
            tests[(seed, framework)] = json.loads((folder / "test_metrics.json").read_text())
        manifest = json.loads((RESULTS / "manifests" / f"epoch_order_manifest_seed{seed}.json").read_text())
        expected_hashes = [row["permutation_sha256"] for row in manifest["orders"]]
        for framework in ("keras", "pytorch"):
            actual_hashes = [row["epoch_order_hash"] for row in histories[(seed, framework)]]
            if actual_hashes != expected_hashes:
                raise RuntimeError(f"Epoch-order mismatch: {framework} seed{seed}")

    curves = []
    for seed in SEEDS:
        for epoch in range(1, 31):
            k, p = histories[(seed, "keras")][epoch - 1], histories[(seed, "pytorch")][epoch - 1]
            curves.append({"seed": seed, "epoch": epoch, "global_step_end": k["global_step_end"],
                           "keras_train_loss": k["train_loss"], "pytorch_train_loss": p["train_loss"],
                           "keras_train_accuracy": k["train_accuracy"], "pytorch_train_accuracy": p["train_accuracy"],
                           "keras_val_loss": k["validation_loss"], "pytorch_val_loss": p["validation_loss"],
                           "keras_val_accuracy": k["validation_accuracy"], "pytorch_val_accuracy": p["validation_accuracy"]})
    write_csv(RESULTS / "summaries/training_curve_comparison.csv", curves)

    epoch_rows = []
    for seed in SEEDS:
        for epoch in EPOCH_SNAPSHOT_EPOCHS:
            name = "initial_step00000" if epoch == 0 else f"epoch{epoch:03d}"
            step = 0 if epoch == 0 else epoch * 1_407
            checkpoint_type = "initial" if epoch == 0 else "epoch"
            ks = _validate_snapshot("keras", seed, name, checkpoint_type, epoch, step)
            ps = _validate_snapshot("pytorch", seed, name, checkpoint_type, epoch, step)
            row = _distance_row(seed, f"epoch{epoch:03d}", step, ks, ps)
            row["comparison_note"] = "same number of V4 workload passes; update count differs from Phase 2"
            epoch_rows.append(row)
    write_csv(RESULTS / "summaries/epoch_matched_trajectory_summary.csv", epoch_rows)

    phase2 = {}
    for seed in SEEDS:
        for row in _read_csv(PHASE2_RESULTS / "trajectory" / f"seed{seed}_weight_trajectory.csv"):
            phase2[(seed, int(row["optimizer_step"]))] = row
    step_rows = []
    for seed in SEEDS:
        for step in STEP_SNAPSHOT_STEPS:
            name = "initial_step00000" if step == 0 else f"step{step:05d}"
            checkpoint_type = "initial" if step == 0 else "step"
            epoch = 0 if step == 0 else (step - 1) // 1_407 + 1
            ks = _validate_snapshot("keras", seed, name, checkpoint_type, epoch, step)
            ps = _validate_snapshot("pytorch", seed, name, checkpoint_type, epoch, step)
            row = _distance_row(seed, name, step, ks, ps)
            parent = phase2[(seed, step)]
            row["phase2_global_weight_rel_l2"] = parent["global_weight_relative_l2"]
            for index in range(1, 5): row[f"phase2_conv{index}_rel_l2"] = parent[f"conv{index}_kernel_relative_l2"]
            row["comparison_note"] = "same optimizer-update budget; workloads remain non-equivalent; classifier excluded cross-workload"
            step_rows.append(row)
    write_csv(RESULTS / "summaries/step_matched_trajectory_summary.csv", step_rows)

    for checkpoint, filename in (("final", "test_final_summary.csv"), ("best_val", "test_best_val_summary.csv")):
        rows = []
        for seed in SEEDS:
            k, p = tests[(seed, "keras")][checkpoint], tests[(seed, "pytorch")][checkpoint]
            for framework, metric in (("keras", k), ("pytorch", p)):
                rows.append({"row_type": "seed", "seed": seed, "framework": framework, "epoch": metric["epoch"], "loss": metric["loss"], "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"], "paired_gap_p_minus_k": ""})
            rows.append({"row_type": "paired_gap", "seed": seed, "framework": "pytorch_minus_keras", "epoch": "", "loss": float(p["loss"]) - float(k["loss"]), "accuracy": float(p["accuracy"]) - float(k["accuracy"]), "macro_f1": float(p["macro_f1"]) - float(k["macro_f1"]), "paired_gap_p_minus_k": True})
        for framework in ("keras", "pytorch"):
            selected = [tests[(seed, framework)][checkpoint] for seed in SEEDS]
            rows.append({"row_type": "aggregate_mean_std", "seed": "all", "framework": framework, "epoch": "",
                         "loss": f"{np.mean([x['loss'] for x in selected])} ± {np.std([x['loss'] for x in selected], ddof=1)}",
                         "accuracy": f"{np.mean([x['accuracy'] for x in selected])} ± {np.std([x['accuracy'] for x in selected], ddof=1)}",
                         "macro_f1": f"{np.mean([x['macro_f1'] for x in selected])} ± {np.std([x['macro_f1'] for x in selected], ddof=1)}", "paired_gap_p_minus_k": ""})
        gaps = {
            metric: [
                float(tests[(seed, "pytorch")][checkpoint][metric])
                - float(tests[(seed, "keras")][checkpoint][metric])
                for seed in SEEDS
            ] for metric in ("loss", "accuracy", "macro_f1")
        }
        rows.append({
            "row_type": "paired_gap_aggregate", "seed": "all", "framework": "pytorch_minus_keras",
            "epoch": "", "loss": f"signed_mean={np.mean(gaps['loss'])}; mean_abs={np.mean(np.abs(gaps['loss']))}",
            "accuracy": f"signed_mean={np.mean(gaps['accuracy'])}; mean_abs={np.mean(np.abs(gaps['accuracy']))}",
            "macro_f1": f"signed_mean={np.mean(gaps['macro_f1'])}; mean_abs={np.mean(np.abs(gaps['macro_f1']))}",
            "paired_gap_p_minus_k": True,
        })
        write_csv(RESULTS / "summaries" / filename, rows)

    write_json(RESULTS / "preflight/training_validity.json", {
        "v4_training_config_equivalence": "VALID", "v4_epoch_order_check": "VALID",
        "v4_step_checkpoint_check": "VALID", "v4_epoch_checkpoint_check": "VALID",
        "v4_training_runs_complete": True, "completed_runs": 6,
        "v4_selected_checkpoints_complete": True, "v4_test_evaluation_complete": True,
        "nan_inf_found": False, "config_sha256": config_hash(),
    })
    print("V4_TRAINING_RUNS_COMPLETE = TRUE (6/6)")
    print("V4_STEP_CHECKPOINT_CHECK = VALID")
    print("V4_EPOCH_CHECKPOINT_CHECK = VALID")
    print("V4_TEST_EVALUATION_COMPLETE = TRUE")


if __name__ == "__main__": main()
