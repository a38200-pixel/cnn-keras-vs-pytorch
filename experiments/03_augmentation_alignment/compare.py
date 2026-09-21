"""Validate Experiment 03 runtime augmentation, then compare with Baseline 00."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cnn-comparison-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.augmentation_runtime import (
    RUNTIME_AUG_FIELDS,
    expected_records,
    parameter_hash,
    runtime_summary_path,
    runtime_values_path,
    sample_identity,
)
from common.dataset_utils import build_split_manifest, find_dataset_root

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
HISTORY_DIR = RESULTS_DIR / "history"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNTIME_AUG_DIR = RESULTS_DIR / "runtime_augmentation"
BASELINE_DIR = ROOT / "experiments" / "00_baseline_final_cnn_3seed" / "results"
SEEDS = [42, 123, 2026]


def validate_runtime_augmentation() -> dict[str, object]:
    """Validate actual __getitem__ traces against schedule and across frameworks."""
    identities = sorted(
        sample_identity(row["path"], find_dataset_root())
        for row in build_split_manifest()["train"]
    )
    errors: list[str] = []
    missing: list[str] = []
    details: dict[str, object] = {}
    for seed in SEEDS:
        framework_rows: dict[str, list[dict[str, str]]] = {}
        seed_detail: dict[str, object] = {}
        for framework in ("keras", "pytorch"):
            summary_path = runtime_summary_path(RUNTIME_AUG_DIR, framework, seed)
            values_path = runtime_values_path(RUNTIME_AUG_DIR, framework, seed)
            if not summary_path.exists():
                missing.append(str(summary_path))
                continue
            if not values_path.exists():
                missing.append(str(values_path))
                continue
            with summary_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                if reader.fieldnames != RUNTIME_AUG_FIELDS:
                    errors.append(f"Unexpected runtime schema: {summary_path}")
            if not rows:
                errors.append(f"Empty runtime augmentation trace: {summary_path}")
                continue
            try:
                with np.load(values_path) as archive:
                    stored_identities = archive["identities"].astype(str).tolist()
                    epoch_indices = archive["epoch_indices"].astype(int).tolist()
                    flips = archive["flips"]
                    angles = archive["angles"]
            except Exception as error:
                errors.append(f"Unreadable runtime NPZ: {values_path}: {error}")
                continue
            if stored_identities != identities:
                errors.append(f"Identity mismatch: {framework} seed={seed}")
                continue
            if len(rows) != len(epoch_indices) or flips.shape != angles.shape:
                errors.append(f"Runtime array shape mismatch: {framework} seed={seed}")
                continue
            if flips.shape != (len(rows), len(identities)):
                errors.append(f"Runtime sample shape mismatch: {framework} seed={seed} {flips.shape}")
                continue
            observed_epochs = [int(row["epoch"]) for row in rows]
            if observed_epochs != list(range(1, len(rows) + 1)):
                errors.append(f"Non-contiguous epochs: {framework} seed={seed} {observed_epochs}")
            self_match = True
            for row_index, row in enumerate(rows):
                epoch = int(row["epoch"])
                schedule_index = int(row["schedule_index"])
                if schedule_index != epoch - 1 or epoch_indices[row_index] != schedule_index:
                    errors.append(f"Epoch/index mismatch: {framework} seed={seed} epoch={epoch}")
                    self_match = False
                    continue
                records = {
                    identity: (bool(flips[row_index, index]), float(angles[row_index, index]))
                    for index, identity in enumerate(identities)
                }
                expected = expected_records(identities, seed, schedule_index)
                observed_hash = parameter_hash(records)
                expected_hash = parameter_hash(expected)
                if records != expected or row["parameter_hash"] != observed_hash or observed_hash != expected_hash:
                    errors.append(f"Runtime parameter mismatch: {framework} seed={seed} epoch={epoch}")
                    self_match = False
                if int(row["num_samples"]) != len(identities):
                    errors.append(f"Sample-count mismatch: {framework} seed={seed} epoch={epoch}")
                    self_match = False
                if (
                    row["framework"] != framework
                    or int(row["seed"]) != seed
                    or row["mode"] != "strict_sample_id"
                    or row["interpolation"] != "bilinear"
                    or row["fill_mode"] != "constant_0"
                ):
                    errors.append(f"Unexpected runtime metadata: {framework} seed={seed} epoch={epoch}")
                    self_match = False
            framework_rows[framework] = rows
            seed_detail[framework] = {
                "epochs_recorded": len(rows),
                "first_hash": rows[0]["parameter_hash"],
                "last_hash": rows[-1]["parameter_hash"],
                "self_schedule_match": self_match,
            }

        if set(framework_rows) == {"keras", "pytorch"}:
            common_epochs = min(len(framework_rows["keras"]), len(framework_rows["pytorch"]))
            mismatches = [
                index + 1 for index in range(common_epochs)
                if framework_rows["keras"][index]["parameter_hash"]
                != framework_rows["pytorch"][index]["parameter_hash"]
            ]
            if mismatches:
                errors.append(f"Framework augmentation hash mismatch: seed={seed} epochs={mismatches}")
            seed_detail["common_epochs_compared"] = common_epochs
            seed_detail["framework_hash_exact_match"] = not mismatches
        details[str(seed)] = seed_detail

    status = "PENDING" if missing else ("INVALID" if errors else "VALID")
    return {
        "status": status,
        "mode": "strict_sample_id",
        "all_common_epoch_hashes_match": status == "VALID",
        "details": details,
        "missing": missing,
        "errors": errors,
    }


def save_runtime_validation(validation: dict[str, object]) -> Path:
    path = RESULTS_DIR / "runtime_augmentation_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_results(directory: Path, framework: str) -> pd.DataFrame | None:
    path = directory / f"{framework}_results.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path).sort_values("seed").reset_index(drop=True)
    return data if data["seed"].tolist() == SEEDS else None


def load_histories(framework: str) -> dict[int, pd.DataFrame] | None:
    histories = {}
    for seed in SEEDS:
        path = HISTORY_DIR / f"{framework}_seed{seed}_history.csv"
        if not path.exists():
            return None
        data = pd.read_csv(path)
        if data.empty:
            return None
        histories[seed] = data
    return histories


def framework_summary(data: pd.DataFrame) -> dict[str, float]:
    return {
        "accuracy_mean": float(data["test_accuracy"].mean()),
        "accuracy_std": float(data["test_accuracy"].std(ddof=1)),
        "macro_f1_mean": float(data["macro_f1"].mean()),
        "macro_f1_std": float(data["macro_f1"].std(ddof=1)),
        "test_loss_mean": float(data["test_loss"].mean()),
        "test_loss_std": float(data["test_loss"].std(ddof=1)),
    }


def metric_gap(keras: pd.DataFrame, torch: pd.DataFrame, metric: str) -> dict[str, object]:
    paired = torch.set_index("seed")[metric] - keras.set_index("seed")[metric]
    signed = float(paired.mean())
    mean_absolute = float(paired.abs().mean())
    return {
        "definition": "PyTorch - Keras",
        "paired_difference": {str(int(seed)): float(value) for seed, value in paired.items()},
        "signed_mean_gap": signed,
        "mean_absolute_paired_gap": mean_absolute,
        "signed_gap": signed,
        "absolute_gap": abs(signed),  # backward-compatible legacy key
    }


def gap_effect(current: dict[str, object], baseline: dict[str, object]) -> dict[str, float | None]:
    baseline_signed = abs(float(baseline["signed_mean_gap"]))
    current_signed = abs(float(current["signed_mean_gap"]))
    baseline_paired = float(baseline["mean_absolute_paired_gap"])
    current_paired = float(current["mean_absolute_paired_gap"])
    signed_reduction = baseline_signed - current_signed
    paired_reduction = baseline_paired - current_paired
    return {
        "signed_gap_reduction": signed_reduction,
        "signed_gap_reduction_rate": None if baseline_signed < 1e-12 else signed_reduction / baseline_signed,
        "absolute_paired_gap_reduction": paired_reduction,
        "absolute_paired_gap_reduction_rate": None if baseline_paired < 1e-12 else paired_reduction / baseline_paired,
        "mean_absolute_paired_gap_reduction": paired_reduction,
        "mean_absolute_paired_gap_reduction_rate": None if baseline_paired < 1e-12 else paired_reduction / baseline_paired,
    }


def plot_framework(framework: str, histories: dict[int, pd.DataFrame], metric: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for seed, data in histories.items():
        axis.plot(data["epoch"], data[f"train_{metric}"], label=f"Seed {seed} Train")
        axis.plot(data["epoch"], data[f"val_{metric}"], "--", label=f"Seed {seed} Val")
    axis.set_title(f"Experiment 03 - {framework.title()} 3-Seed {metric.title()}")
    axis.set(xlabel="Epoch", ylabel=metric.title())
    axis.grid(alpha=.25)
    axis.legend(ncol=2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined(keras_histories, torch_histories, metric: str, path: Path) -> None:
    colors = {42: "#4C78A8", 123: "#F58518", 2026: "#54A24B"}
    fig, axis = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    for seed in SEEDS:
        axis.plot(keras_histories[seed]["epoch"], keras_histories[seed][f"val_{metric}"],
                  color=colors[seed], label=f"Keras Seed {seed}")
        axis.plot(torch_histories[seed]["epoch"], torch_histories[seed][f"val_{metric}"],
                  color=colors[seed], linestyle="--", label=f"PyTorch Seed {seed}")
    axis.set_title(f"Experiment 03 - Validation {metric.title()} Comparison")
    axis.set(xlabel="Epoch", ylabel=f"Validation {metric.title()}")
    axis.grid(alpha=.25)
    axis.legend(ncol=2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_plots(keras_histories, torch_histories) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        FIGURES_DIR / "keras_3seed_loss.png",
        FIGURES_DIR / "keras_3seed_accuracy.png",
        FIGURES_DIR / "pytorch_3seed_loss.png",
        FIGURES_DIR / "pytorch_3seed_accuracy.png",
        FIGURES_DIR / "validation_loss_3seed_comparison.png",
        FIGURES_DIR / "validation_accuracy_3seed_comparison.png",
    ]
    plot_framework("keras", keras_histories, "loss", outputs[0])
    plot_framework("keras", keras_histories, "accuracy", outputs[1])
    plot_framework("pytorch", torch_histories, "loss", outputs[2])
    plot_framework("pytorch", torch_histories, "accuracy", outputs[3])
    plot_combined(keras_histories, torch_histories, "loss", outputs[4])
    plot_combined(keras_histories, torch_histories, "accuracy", outputs[5])
    return outputs


def main() -> None:
    print("Experiment 03 comparison only: no training or test evaluation is performed.")
    validation = validate_runtime_augmentation()
    validation_path = save_runtime_validation(validation)
    print(f"Experiment Validity: {validation['status']}")
    print(f"Runtime validation report: {validation_path}")
    if validation["status"] != "VALID":
        keras, torch = load_results(RESULTS_DIR, "keras"), load_results(RESULTS_DIR, "pytorch")
        if validation["status"] == "PENDING" and keras is not None and torch is not None:
            validation["status"] = "INVALID"
            validation["errors"].append("Complete results exist but runtime augmentation artifacts are missing.")
            save_runtime_validation(validation)
            print("Experiment Validity: INVALID")
        for item in validation["missing"]:
            print(f"Missing runtime artifact: {item}")
        for item in validation["errors"]:
            print(f"Runtime augmentation error: {item}")
        print("Performance/Gap/OFAT analysis was not generated because the runtime gate did not pass.")
        return

    keras, torch = load_results(RESULTS_DIR, "keras"), load_results(RESULTS_DIR, "pytorch")
    baseline_keras = load_results(BASELINE_DIR, "keras")
    baseline_torch = load_results(BASELINE_DIR, "pytorch")
    keras_histories, torch_histories = load_histories("keras"), load_histories("pytorch")
    if any(value is None for value in (keras, torch, baseline_keras, baseline_torch, keras_histories, torch_histories)):
        print("Complete results and six history files are required after runtime validation.")
        return
    for framework, histories in (("keras", keras_histories), ("pytorch", torch_histories)):
        for seed, history in histories.items():
            recorded = validation["details"][str(seed)][framework]["epochs_recorded"]
            if len(history) != recorded:
                validation["status"] = "INVALID"
                validation["errors"].append(
                    f"Runtime/history count mismatch: {framework} seed={seed} runtime={recorded} history={len(history)}"
                )
    if validation["status"] != "VALID":
        validation["all_common_epoch_hashes_match"] = False
        save_runtime_validation(validation)
        print("Experiment Validity: INVALID")
        return

    metrics = {"accuracy": "test_accuracy", "macro_f1": "macro_f1", "test_loss": "test_loss"}
    baseline_gaps = {name: metric_gap(baseline_keras, baseline_torch, column) for name, column in metrics.items()}
    current_gaps = {name: metric_gap(keras, torch, column) for name, column in metrics.items()}
    summary: dict[str, object] = {
        "experiment_validity": "VALID",
        "runtime_augmentation_validation": validation,
        "comparison_reference": "00_baseline_final_cnn_3seed",
        "gap_definition": "PyTorch - Keras",
        "baseline_00": {
            "keras": framework_summary(baseline_keras), "pytorch": framework_summary(baseline_torch),
            "gaps": baseline_gaps,
        },
        "experiment_03": {
            "keras": framework_summary(keras), "pytorch": framework_summary(torch),
            "gaps": current_gaps,
        },
        "gap_effect": {
            name: gap_effect(current_gaps[name], baseline_gaps[name]) for name in metrics
        },
        "seed_stability_std_change": {
            "keras_accuracy": float(keras["test_accuracy"].std(ddof=1) - baseline_keras["test_accuracy"].std(ddof=1)),
            "keras_macro_f1": float(keras["macro_f1"].std(ddof=1) - baseline_keras["macro_f1"].std(ddof=1)),
            "pytorch_accuracy": float(torch["test_accuracy"].std(ddof=1) - baseline_torch["test_accuracy"].std(ddof=1)),
            "pytorch_macro_f1": float(torch["macro_f1"].std(ddof=1) - baseline_torch["macro_f1"].std(ddof=1)),
        },
    }
    figures = generate_plots(keras_histories, torch_histories)
    summary["figures"] = [str(path.relative_to(EXPERIMENT_DIR)) for path in figures]
    output = RESULTS_DIR / "comparison_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {output}")
    for figure in figures:
        print(f"Saved figure: {figure}")


if __name__ == "__main__":
    main()
