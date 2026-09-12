"""Compare paired 3-seed framework results without fabricating missing data."""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


def read_rows(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(results_dir.glob("*_results.csv")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def summarize(results_dir: Path) -> dict[str, object] | None:
    rows = read_rows(results_dir)
    if not rows:
        print("Experiment results were not found.\nRun the training scripts first.")
        return None
    by = {(r["framework"].lower(), int(r["seed"])): r for r in rows}
    seeds = sorted(set(s for f, s in by if ("keras", s) in by and ("pytorch", s) in by))
    if not seeds:
        print("Paired Keras/PyTorch seed results were not found.")
        return None
    gaps = [float(by[("keras", s)]["macro_f1"]) - float(by[("pytorch", s)]["macro_f1"]) for s in seeds]
    summary = {
        "seeds": seeds,
        "keras_macro_f1_mean": statistics.mean(float(by[("keras", s)]["macro_f1"]) for s in seeds),
        "pytorch_macro_f1_mean": statistics.mean(float(by[("pytorch", s)]["macro_f1"]) for s in seeds),
        "mean_gap": statistics.mean(gaps),
        "mean_absolute_gap": statistics.mean(abs(x) for x in gaps),
        "gap_std": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
    }
    numeric = ["test_accuracy", "macro_f1", "test_loss", "best_epoch", "epochs_trained",
               "train_accuracy", "validation_accuracy", "train_loss", "validation_loss",
               "train_val_gap", "training_time_seconds"]
    for framework in ("keras", "pytorch"):
        framework_rows = [r for r in rows if r["framework"].lower() == framework]
        for metric in numeric:
            values = [float(r[metric]) for r in framework_rows]
            if values:
                summary[f"{framework}_{metric}_mean"] = statistics.mean(values)
                summary[f"{framework}_{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def gap_effect(current_abs_gap: float, baseline_abs_gap: float) -> dict[str, float]:
    change = current_abs_gap - baseline_abs_gap
    return {"gap_change": change, "gap_reduction": -change,
            "gap_reduction_rate": math.nan if abs(baseline_abs_gap) < 1e-12 else -change / baseline_abs_gap}
