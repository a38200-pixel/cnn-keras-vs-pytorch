"""Compare Experiment 01 with Baseline 00 and plot saved histories. Never trains."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cnn-comparison-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
HISTORY_DIR = RESULTS_DIR / "history"
FIGURES_DIR = RESULTS_DIR / "figures"
BASELINE_DIR = ROOT / "experiments" / "00_baseline_final_cnn_3seed" / "results"
SEEDS = [42, 123, 2026]


def load_framework_results(directory: Path, framework: str) -> pd.DataFrame | None:
    path = directory / f"{framework}_results.csv"
    if not path.exists():
        print(f"Missing result file: {path}")
        return None
    data = pd.read_csv(path).sort_values("seed").reset_index(drop=True)
    if data["seed"].tolist() != SEEDS:
        print(f"Incomplete {framework} results: found seeds {data['seed'].tolist()}, expected {SEEDS}")
        return None
    return data


def load_histories(framework: str) -> dict[int, pd.DataFrame] | None:
    histories: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        path = HISTORY_DIR / f"{framework}_seed{seed}_history.csv"
        if not path.exists():
            print(f"Missing history file: {path}")
            return None
        data = pd.read_csv(path)
        if data.empty or data["seed"].astype(int).unique().tolist() != [seed]:
            print(f"Invalid or empty history file: {path}")
            return None
        histories[seed] = data
    return histories


def sample_std(series: pd.Series) -> float:
    return float(series.std(ddof=1))


def framework_summary(data: pd.DataFrame) -> dict[str, float]:
    return {
        "accuracy_mean": float(data["test_accuracy"].mean()),
        "accuracy_std": sample_std(data["test_accuracy"]),
        "macro_f1_mean": float(data["macro_f1"].mean()),
        "macro_f1_std": sample_std(data["macro_f1"]),
        "test_loss_mean": float(data["test_loss"].mean()),
    }


def metric_gap(keras: pd.DataFrame, torch: pd.DataFrame, metric: str) -> dict[str, object]:
    paired = torch.set_index("seed")[metric] - keras.set_index("seed")[metric]
    signed_gap = float(paired.mean())
    return {
        "definition": "PyTorch - Keras",
        "paired_difference": {str(int(seed)): float(value) for seed, value in paired.items()},
        "signed_gap": signed_gap,
        "absolute_gap": abs(signed_gap),
    }


def gap_change(current_abs: float, baseline_abs: float) -> dict[str, float | None]:
    reduction = baseline_abs - current_abs
    rate = None if abs(baseline_abs) < 1e-12 else reduction / baseline_abs
    return {
        "gap_reduction": reduction,
        "gap_reduction_rate": rate,
    }


def plot_framework_history(
    framework: str,
    histories: dict[int, pd.DataFrame],
    metric: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for seed, data in histories.items():
        ax.plot(data["epoch"], data[f"train_{metric}"], label=f"Seed {seed} Train")
        ax.plot(data["epoch"], data[f"val_{metric}"], "--", label=f"Seed {seed} Val")
    ax.set_title(f"Experiment 01 - {framework.title()} 3-Seed {metric.title()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.title())
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined_validation(
    keras_histories: dict[int, pd.DataFrame],
    torch_histories: dict[int, pd.DataFrame],
    metric: str,
    output: Path,
) -> None:
    colors = {42: "#4C78A8", 123: "#F58518", 2026: "#54A24B"}
    fig, ax = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    for seed in SEEDS:
        keras = keras_histories[seed]
        torch = torch_histories[seed]
        ax.plot(
            keras["epoch"], keras[f"val_{metric}"], color=colors[seed], linestyle="-",
            label=f"Keras Seed {seed}",
        )
        ax.plot(
            torch["epoch"], torch[f"val_{metric}"], color=colors[seed], linestyle="--",
            label=f"PyTorch Seed {seed}",
        )
    ax.set_title(f"Experiment 01 - Validation {metric.title()} Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Validation {metric.title()}")
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    keras_histories: dict[int, pd.DataFrame],
    torch_histories: dict[int, pd.DataFrame],
) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        FIGURES_DIR / "keras_3seed_loss.png",
        FIGURES_DIR / "keras_3seed_accuracy.png",
        FIGURES_DIR / "pytorch_3seed_loss.png",
        FIGURES_DIR / "pytorch_3seed_accuracy.png",
        FIGURES_DIR / "validation_loss_3seed_comparison.png",
        FIGURES_DIR / "validation_accuracy_3seed_comparison.png",
    ]
    plot_framework_history("keras", keras_histories, "loss", outputs[0])
    plot_framework_history("keras", keras_histories, "accuracy", outputs[1])
    plot_framework_history("pytorch", torch_histories, "loss", outputs[2])
    plot_framework_history("pytorch", torch_histories, "accuracy", outputs[3])
    plot_combined_validation(keras_histories, torch_histories, "loss", outputs[4])
    plot_combined_validation(keras_histories, torch_histories, "accuracy", outputs[5])
    return outputs


def markdown_preview(summary: dict[str, object]) -> str:
    current = summary["experiment_01"]
    baseline = summary["baseline_00"]
    changes = summary["gap_change"]
    return f"""
## Experiment 01 Result Preview

| Metric | Baseline Signed Gap | Experiment 01 Signed Gap | Absolute Gap Reduction |
|---|---:|---:|---:|
| Accuracy | {baseline['accuracy_gap']['signed_gap'] * 100:+.2f}%p | {current['accuracy_gap']['signed_gap'] * 100:+.2f}%p | {changes['accuracy']['gap_reduction'] * 100:+.2f}%p |
| Macro F1 | {baseline['macro_f1_gap']['signed_gap'] * 100:+.2f}%p | {current['macro_f1_gap']['signed_gap'] * 100:+.2f}%p | {changes['macro_f1']['gap_reduction'] * 100:+.2f}%p |

Signed Gap = PyTorch - Keras. Positive gap reduction means the framework gap decreased.
""".strip()


def main() -> None:
    print("Experiment 01 comparison only: no training is performed.")
    keras = load_framework_results(RESULTS_DIR, "keras")
    torch = load_framework_results(RESULTS_DIR, "pytorch")
    if keras is None or torch is None:
        print("Experiment 01 is still waiting for complete 3-Seed training results.")
        return
    baseline_keras = load_framework_results(BASELINE_DIR, "keras")
    baseline_torch = load_framework_results(BASELINE_DIR, "pytorch")
    if baseline_keras is None or baseline_torch is None:
        print("Complete Baseline 00 results are required.")
        return
    keras_histories = load_histories("keras")
    torch_histories = load_histories("pytorch")
    if keras_histories is None or torch_histories is None:
        print("All six history CSV files are required before plotting.")
        return

    baseline_accuracy_gap = metric_gap(baseline_keras, baseline_torch, "test_accuracy")
    baseline_f1_gap = metric_gap(baseline_keras, baseline_torch, "macro_f1")
    current_accuracy_gap = metric_gap(keras, torch, "test_accuracy")
    current_f1_gap = metric_gap(keras, torch, "macro_f1")
    summary: dict[str, object] = {
        "gap_definition": "PyTorch - Keras",
        "baseline_00": {
            "keras": framework_summary(baseline_keras),
            "pytorch": framework_summary(baseline_torch),
            "accuracy_gap": baseline_accuracy_gap,
            "macro_f1_gap": baseline_f1_gap,
        },
        "experiment_01": {
            "keras": framework_summary(keras),
            "pytorch": framework_summary(torch),
            "accuracy_gap": current_accuracy_gap,
            "macro_f1_gap": current_f1_gap,
        },
        "gap_change": {
            "accuracy": gap_change(current_accuracy_gap["absolute_gap"], baseline_accuracy_gap["absolute_gap"]),
            "macro_f1": gap_change(current_f1_gap["absolute_gap"], baseline_f1_gap["absolute_gap"]),
        },
        "seed_stability_std_change": {
            "keras_accuracy": sample_std(keras["test_accuracy"]) - sample_std(baseline_keras["test_accuracy"]),
            "keras_macro_f1": sample_std(keras["macro_f1"]) - sample_std(baseline_keras["macro_f1"]),
            "pytorch_accuracy": sample_std(torch["test_accuracy"]) - sample_std(baseline_torch["test_accuracy"]),
            "pytorch_macro_f1": sample_std(torch["macro_f1"]) - sample_std(baseline_torch["macro_f1"]),
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
