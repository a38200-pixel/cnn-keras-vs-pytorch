"""Read-only Experiment 04 vs 05 diagnostic/training comparison."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_config import SEEDS
from experiment_config import RESULTS, config_hash


def read_csv(path):
    with path.open(encoding="utf-8") as handle: return list(csv.DictReader(handle))


def main() -> None:
    preflight = json.loads((RESULTS / "preflight/preflight_summary.json").read_text(encoding="utf-8"))
    if preflight.get("common_adam_precheck") != "VALID": raise RuntimeError("Common Adam preflight is not VALID")
    diagnostics = {}
    for seed in SEEDS:
        first = json.loads((RESULTS / "first_step" / f"seed{seed}_summary.json").read_text(encoding="utf-8"))
        early = {int(row["step"]): row for row in read_csv(RESULTS / "early_steps" / f"seed{seed}_step_summary.csv")}
        diagnostics[str(seed)] = {
            "gradient_relative_l2": first["gradient_relative_l2"],
            "experiment04_native_update_relative_l2": first["experiment04_native_update_relative_l2"],
            "experiment05_common_update_relative_l2": first["update_relative_l2"],
            "common_vs_native_update_ratio": first["common_vs_native_update_ratio"],
            "step100_weight_relative_l2": float(early[100]["global_weight_relative_l2"]),
        }
    diagnostic_output = {
        "status": "VALID", "config_sha256": config_hash(),
        "interpretation": "Only native Adam implementation was replaced; gradient, BN, and backend differences remain.",
        "by_seed": diagnostics,
    }
    path = RESULTS / "diagnostic_comparison_04_vs_05.json"
    path.write_text(json.dumps(diagnostic_output, indent=2) + "\n", encoding="utf-8")
    result_paths = {fw: RESULTS / f"{fw}_common_adam_results.csv" for fw in ("keras", "pytorch")}
    if not all(path.exists() for path in result_paths.values()):
        print(f"Saved diagnostic comparison: {path}")
        print("Training Pending: no 30-epoch performance comparison generated."); return
    results = {fw: {int(row["seed"]): row for row in read_csv(path)} for fw, path in result_paths.items()}
    metrics = {}
    for metric in ("final_test_accuracy", "final_macro_f1", "final_test_loss", "best_test_accuracy", "best_macro_f1", "best_test_loss"):
        k = [float(results["keras"][seed][metric]) for seed in SEEDS]
        p = [float(results["pytorch"][seed][metric]) for seed in SEEDS]
        gaps = [right - left for left, right in zip(k, p)]
        metrics[metric] = {"keras_mean": statistics.mean(k), "pytorch_mean": statistics.mean(p),
                           "signed_mean_gap": statistics.mean(gaps),
                           "mean_absolute_paired_gap": statistics.mean(abs(value) for value in gaps)}
    output = RESULTS / "controlled_comparison_summary.json"
    output.write_text(json.dumps({"status": "VALID", "config_sha256": config_hash(), "metrics": metrics, "diagnostics": diagnostics}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved completed comparison: {output}")


if __name__ == "__main__": main()
