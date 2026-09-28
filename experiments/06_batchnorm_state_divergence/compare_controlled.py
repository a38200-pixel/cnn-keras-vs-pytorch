"""Read-only Experiment 05 native-BN vs Experiment 06 Common-BN comparison."""

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

PARENT_RESULTS = ROOT / "experiments/05_gradient_optimizer_divergence/results"


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    preflight = json.loads((RESULTS / "preflight/preflight_summary.json").read_text(encoding="utf-8"))
    if preflight.get("common_bn_precheck") != "VALID":
        raise RuntimeError("Common BN preflight is not VALID")
    diagnostics = {}
    for seed in SEEDS:
        first = json.loads((RESULTS / "first_step" / f"seed{seed}_summary.json").read_text(encoding="utf-8"))
        early = {int(row["step"]): row for row in read_csv(RESULTS / "early_steps" / f"seed{seed}_step_summary.csv")}
        parent = first["experiment05_native_bn"]
        diagnostics[str(seed)] = {
            "experiment05_bn1_output_relative_l2": parent["bn1_output_relative_l2"],
            "experiment06_bn1_output_relative_l2": first["bn1_output_relative_l2"],
            "experiment05_bn_running_mean_relative_l2": parent["bn_running_mean_relative_l2"],
            "experiment06_bn_running_mean_relative_l2": first["bn_running_mean_relative_l2"],
            "experiment05_bn_running_variance_relative_l2": parent["bn_running_variance_relative_l2"],
            "experiment06_bn_running_variance_relative_l2": first["bn_running_variance_relative_l2"],
            "experiment05_gradient_relative_l2": parent["gradient_relative_l2"],
            "experiment06_gradient_relative_l2": first["gradient_relative_l2"],
            "experiment05_update_relative_l2": parent["update_relative_l2"],
            "experiment06_update_relative_l2": first["update_relative_l2"],
            "experiment05_step100_weight_relative_l2": float(
                next(row for row in read_csv(PARENT_RESULTS / "early_steps" / f"seed{seed}_step_summary.csv") if int(row["step"]) == 100)["global_weight_relative_l2"]
            ),
            "experiment06_step100_weight_relative_l2": float(early[100]["global_weight_relative_l2"]),
        }
    diagnostic_output = {
        "status": "VALID", "config_sha256": config_hash(),
        "interpretation": "Only native BatchNorm semantics were replaced; backend reduction/autograd differences remain.",
        "experiment05_bn1_forward_source": "Experiment 04 first-step trace; Experiment 05 has the identical pre-update native-BN forward path, W0, and input",
        "by_seed": diagnostics, "full_training_executed": False,
    }
    path = RESULTS / "diagnostic_comparison_05_vs_06.json"
    path.write_text(json.dumps(diagnostic_output, indent=2) + "\n", encoding="utf-8")
    result_paths = {framework: RESULTS / f"{framework}_common_bn_results.csv" for framework in ("keras", "pytorch")}
    if not all(path.exists() for path in result_paths.values()):
        print(f"Saved diagnostic comparison: {path}")
        print("Full Training Pending: no 30-epoch performance comparison generated.")
        return
    from trajectory_from_checkpoints import generate_trajectories
    trajectory_files = [str(path.relative_to(RESULTS)) for path in generate_trajectories()]
    results = {
        framework: {int(row["seed"]): row for row in read_csv(path)}
        for framework, path in result_paths.items()
    }
    metrics = {}
    for metric in ("final_test_accuracy", "final_macro_f1", "final_test_loss", "best_test_accuracy", "best_macro_f1", "best_test_loss"):
        keras = [float(results["keras"][seed][metric]) for seed in SEEDS]
        pytorch = [float(results["pytorch"][seed][metric]) for seed in SEEDS]
        gaps = [right - left for left, right in zip(keras, pytorch)]
        metrics[metric] = {
            "keras_mean": statistics.mean(keras), "pytorch_mean": statistics.mean(pytorch),
            "signed_mean_gap": statistics.mean(gaps),
            "mean_absolute_paired_gap": statistics.mean(abs(value) for value in gaps),
        }
    output = RESULTS / "controlled_comparison_summary.json"
    output.write_text(json.dumps({
        "status": "VALID", "config_sha256": config_hash(),
        "metrics": metrics, "diagnostics": diagnostics,
        "trajectory_files": trajectory_files,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved completed comparison: {output}")


if __name__ == "__main__":
    main()
