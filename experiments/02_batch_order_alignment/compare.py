"""Compare Experiment 02 directly with Baseline 00 and plot histories. Never trains."""

from __future__ import annotations

import json
import os
import csv
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cnn-comparison-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.batch_order import RUNTIME_ORDER_FIELDS, order_hash

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
HISTORY_DIR = RESULTS_DIR / "history"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNTIME_ORDER_DIR = RESULTS_DIR / "runtime_order"
BATCH_ORDER_DIR = RESULTS_DIR / "batch_order"
BASELINE_DIR = ROOT / "experiments" / "00_baseline_final_cnn_3seed" / "results"
SEEDS = [42, 123, 2026]


def validate_runtime_orders() -> dict[str, object]:
    """Gate performance analysis on orders selected at real epoch boundaries."""
    errors: list[str] = []
    missing: list[str] = []
    details: dict[str, object] = {}
    for seed in SEEDS:
        schedule_path = BATCH_ORDER_DIR / f"seed_{seed}_epoch_orders.npz"
        if not schedule_path.exists():
            missing.append(str(schedule_path))
            continue
        import numpy as np

        with np.load(schedule_path) as archive:
            schedules = archive["orders"]
        framework_rows: dict[str, list[dict[str, str]]] = {}
        seed_detail: dict[str, object] = {}
        for framework in ("keras", "pytorch"):
            path = RUNTIME_ORDER_DIR / f"{framework}_seed{seed}_runtime_order.csv"
            if not path.exists():
                missing.append(str(path))
                continue
            with path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                if reader.fieldnames != RUNTIME_ORDER_FIELDS:
                    errors.append(f"Unexpected runtime schema: {path}")
            if not rows:
                errors.append(f"Empty runtime trace: {path}")
                continue
            framework_rows[framework] = rows
            observed_epochs = [int(row["epoch"]) for row in rows]
            expected_epochs = list(range(1, len(rows) + 1))
            if observed_epochs != expected_epochs:
                errors.append(
                    f"Non-contiguous epochs: {framework} seed={seed} {observed_epochs}"
                )
            for row in rows:
                epoch = int(row["epoch"])
                schedule_index = int(row["schedule_index"])
                if schedule_index != epoch - 1:
                    errors.append(
                        f"Epoch/index mismatch: {framework} seed={seed} "
                        f"epoch={epoch} index={schedule_index}"
                    )
                    continue
                if schedule_index >= len(schedules):
                    errors.append(f"Schedule index out of range: {framework} seed={seed}")
                    continue
                expected_hash = order_hash(schedules[schedule_index])
                if row["order_hash"] != expected_hash:
                    errors.append(
                        f"Schedule hash mismatch: {framework} seed={seed} epoch={epoch}"
                    )
                if int(row["num_samples"]) != schedules.shape[1]:
                    errors.append(
                        f"Sample-count mismatch: {framework} seed={seed} epoch={epoch}"
                    )
            seed_detail[framework] = {
                "epochs_recorded": len(rows),
                "first_hash": rows[0]["order_hash"],
                "last_hash": rows[-1]["order_hash"],
                "self_schedule_match": not any(
                    framework in error and f"seed={seed}" in error for error in errors
                ),
            }

        if set(framework_rows) == {"keras", "pytorch"}:
            common_epochs = min(len(framework_rows["keras"]), len(framework_rows["pytorch"]))
            pair_mismatches = []
            for index in range(common_epochs):
                keras_row = framework_rows["keras"][index]
                torch_row = framework_rows["pytorch"][index]
                if keras_row["order_hash"] != torch_row["order_hash"]:
                    pair_mismatches.append(index + 1)
            if pair_mismatches:
                errors.append(f"Framework hash mismatch: seed={seed} epochs={pair_mismatches}")
            seed_detail["common_epochs_compared"] = common_epochs
            seed_detail["framework_hash_exact_match"] = not pair_mismatches
        details[str(seed)] = seed_detail

    status = "PENDING" if missing else ("INVALID" if errors else "VALID")
    return {
        "status": status,
        "all_common_epoch_hashes_match": status == "VALID",
        "details": details,
        "missing": missing,
        "errors": errors,
    }


def save_runtime_validation(validation: dict[str, object]) -> Path:
    path = RESULTS_DIR / "runtime_order_validation.json"
    path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_debugging_reference(keras: pd.DataFrame | None, torch: pd.DataFrame | None) -> None:
    if keras is None or torch is None:
        return
    print("Performance values below are debugging reference only; no OFAT conclusion is generated.")
    paired = keras.set_index("seed").join(
        torch.set_index("seed"), lsuffix="_keras", rsuffix="_pytorch"
    )
    for seed, row in paired.iterrows():
        print(
            f"Seed {seed}: accuracy K={row['test_accuracy_keras']:.6f} "
            f"P={row['test_accuracy_pytorch']:.6f}; macro_f1 K={row['macro_f1_keras']:.6f} "
            f"P={row['macro_f1_pytorch']:.6f}"
        )


def load_results(directory: Path, framework: str) -> pd.DataFrame | None:
    path = directory / f"{framework}_results.csv"
    if not path.exists():
        print(f"Missing result file: {path}")
        return None
    data = pd.read_csv(path).sort_values("seed").reset_index(drop=True)
    if data["seed"].tolist() != SEEDS:
        print(f"Incomplete {framework} results: {data['seed'].tolist()} (expected {SEEDS})")
        return None
    return data


def load_histories(framework: str) -> dict[int, pd.DataFrame] | None:
    histories = {}
    for seed in SEEDS:
        path = HISTORY_DIR / f"{framework}_seed{seed}_history.csv"
        if not path.exists():
            print(f"Missing history file: {path}")
            return None
        data = pd.read_csv(path)
        if data.empty:
            print(f"Empty history file: {path}")
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
    }


def metric_gap(keras: pd.DataFrame, torch: pd.DataFrame, metric: str) -> dict[str, object]:
    paired = torch.set_index("seed")[metric] - keras.set_index("seed")[metric]
    signed = float(paired.mean())
    return {
        "definition": "PyTorch - Keras",
        "paired_difference": {str(int(seed)): float(value) for seed, value in paired.items()},
        "signed_gap": signed,
        "absolute_gap": abs(signed),
    }


def gap_effect(current: float, baseline: float) -> dict[str, float | None]:
    reduction = baseline - current
    return {
        "gap_reduction": reduction,
        "gap_reduction_rate": None if abs(baseline) < 1e-12 else reduction / baseline,
    }


def plot_framework(framework: str, histories: dict[int, pd.DataFrame], metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for seed, data in histories.items():
        ax.plot(data["epoch"], data[f"train_{metric}"], label=f"Seed {seed} Train")
        ax.plot(data["epoch"], data[f"val_{metric}"], "--", label=f"Seed {seed} Val")
    ax.set_title(f"Experiment 02 - {framework.title()} 3-Seed {metric.title()}")
    ax.set(xlabel="Epoch", ylabel=metric.title())
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    keras_histories: dict[int, pd.DataFrame],
    torch_histories: dict[int, pd.DataFrame],
    metric: str,
    path: Path,
) -> None:
    colors = {42: "#4C78A8", 123: "#F58518", 2026: "#54A24B"}
    fig, ax = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    for seed in SEEDS:
        keras, torch = keras_histories[seed], torch_histories[seed]
        ax.plot(keras["epoch"], keras[f"val_{metric}"], color=colors[seed],
                label=f"Keras Seed {seed}")
        ax.plot(torch["epoch"], torch[f"val_{metric}"], color=colors[seed], linestyle="--",
                label=f"PyTorch Seed {seed}")
    ax.set_title(f"Experiment 02 - Validation {metric.title()} Comparison")
    ax.set(xlabel="Epoch", ylabel=f"Validation {metric.title()}")
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
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


def markdown_preview(summary: dict[str, object]) -> str:
    baseline = summary["baseline_00"]
    current = summary["experiment_02"]
    effect = summary["gap_effect"]
    return f"""
## Experiment 02 Result Preview

| Metric | Baseline Signed Gap | Experiment 02 Signed Gap | Absolute Gap Reduction |
|---|---:|---:|---:|
| Accuracy | {baseline['accuracy_gap']['signed_gap'] * 100:+.2f}%p | {current['accuracy_gap']['signed_gap'] * 100:+.2f}%p | {effect['accuracy']['gap_reduction'] * 100:+.2f}%p |
| Macro F1 | {baseline['macro_f1_gap']['signed_gap'] * 100:+.2f}%p | {current['macro_f1_gap']['signed_gap'] * 100:+.2f}%p | {effect['macro_f1']['gap_reduction'] * 100:+.2f}%p |

Signed Gap = PyTorch - Keras. Experiment 02 is compared directly with Baseline 00, not Experiment 01.
""".strip()


def main() -> None:
    print("Experiment 02 comparison only: no training is performed.")
    validation = validate_runtime_orders()
    validation_path = save_runtime_validation(validation)
    print(f"Experiment Validity: {validation['status']}")
    print(f"Runtime validation report: {validation_path}")
    if validation["status"] != "VALID":
        keras = load_results(RESULTS_DIR, "keras")
        torch = load_results(RESULTS_DIR, "pytorch")
        if validation["status"] == "PENDING" and keras is not None and torch is not None:
            validation["status"] = "INVALID"
            validation["errors"].append(
                "Complete performance results exist but runtime-order artifacts are missing."
            )
            save_runtime_validation(validation)
            print("Experiment Validity: INVALID")
        for item in validation["missing"]:
            print(f"Missing runtime artifact: {item}")
        for item in validation["errors"]:
            print(f"Runtime order error: {item}")
        print_debugging_reference(keras, torch)
        print("Performance/Gap/OFAT analysis was not generated because the runtime gate did not pass.")
        return

    keras, torch = load_results(RESULTS_DIR, "keras"), load_results(RESULTS_DIR, "pytorch")
    if keras is None or torch is None:
        print("Experiment 02 is waiting for complete 3-Seed results.")
        return
    baseline_keras = load_results(BASELINE_DIR, "keras")
    baseline_torch = load_results(BASELINE_DIR, "pytorch")
    if baseline_keras is None or baseline_torch is None:
        print("Complete Baseline 00 results are required.")
        return
    keras_histories, torch_histories = load_histories("keras"), load_histories("pytorch")
    if keras_histories is None or torch_histories is None:
        print("All six history files are required before plotting.")
        return

    history_count_errors = []
    for framework, histories in (("keras", keras_histories), ("pytorch", torch_histories)):
        for seed, history in histories.items():
            recorded = validation["details"][str(seed)][framework]["epochs_recorded"]
            if len(history) != recorded:
                history_count_errors.append(
                    f"{framework} seed={seed}: history={len(history)}, runtime={recorded}"
                )
    if history_count_errors:
        validation["status"] = "INVALID"
        validation["all_common_epoch_hashes_match"] = False
        validation["errors"].extend(history_count_errors)
        save_runtime_validation(validation)
        print("Experiment Validity: INVALID")
        for item in history_count_errors:
            print(f"Runtime/history count error: {item}")
        print_debugging_reference(keras, torch)
        print("Performance/Gap/OFAT analysis was not generated.")
        return

    base_acc = metric_gap(baseline_keras, baseline_torch, "test_accuracy")
    base_f1 = metric_gap(baseline_keras, baseline_torch, "macro_f1")
    current_acc = metric_gap(keras, torch, "test_accuracy")
    current_f1 = metric_gap(keras, torch, "macro_f1")
    summary: dict[str, object] = {
        "experiment_validity": "VALID",
        "runtime_order_validation": validation,
        "comparison_reference": "00_baseline_final_cnn_3seed",
        "gap_definition": "PyTorch - Keras",
        "baseline_00": {
            "keras": framework_summary(baseline_keras), "pytorch": framework_summary(baseline_torch),
            "accuracy_gap": base_acc, "macro_f1_gap": base_f1,
        },
        "experiment_02": {
            "keras": framework_summary(keras), "pytorch": framework_summary(torch),
            "accuracy_gap": current_acc, "macro_f1_gap": current_f1,
        },
        "gap_effect": {
            "accuracy": gap_effect(current_acc["absolute_gap"], base_acc["absolute_gap"]),
            "macro_f1": gap_effect(current_f1["absolute_gap"], base_f1["absolute_gap"]),
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
    print(markdown_preview(summary))
    print(f"Saved summary: {output}")
    for figure in figures:
        print(f"Saved figure: {figure}")


if __name__ == "__main__":
    main()
