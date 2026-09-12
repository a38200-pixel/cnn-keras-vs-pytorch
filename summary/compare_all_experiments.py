"""Create the cross-experiment table from real result CSV files only."""
from __future__ import annotations
import csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.comparison_utils import gap_effect, summarize

EXPERIMENTS = [
    ("00_baseline_final_cnn_3seed", "None"), ("01_input_tensor_alignment", "Input Tensor"),
    ("02_batch_order_alignment", "Batch Order"), ("03_augmentation_alignment", "Augmentation"),
    ("04_initial_weight_alignment", "Initial Weight"), ("05_output_loss_alignment", "Output / Loss"),
    ("06_adam_alignment", "Adam"), ("07_batchnorm_alignment", "BatchNorm"),
    ("08_earlystop_lr_scheduler_alignment", "ES / LR Scheduler"),
]


def collect() -> list[dict[str, object]]:
    rows, baseline = [], None
    for name, variable in EXPERIMENTS:
        result = summarize(ROOT / "experiments" / name / "results")
        if result is None: continue
        if name.startswith("00_"): baseline = result
        effect = gap_effect(float(result["mean_absolute_gap"]), float(baseline["mean_absolute_gap"])) if baseline else {
            "gap_change": 0.0, "gap_reduction": 0.0, "gap_reduction_rate": 0.0}
        rows.append({"experiment": name, "aligned_variable": variable,
            "keras_macro_f1_mean": result["keras_macro_f1_mean"],
            "pytorch_macro_f1_mean": result["pytorch_macro_f1_mean"], "mean_gap": result["mean_gap"],
            "mean_absolute_gap": result["mean_absolute_gap"], **effect})
    return rows


def main() -> None:
    rows = collect()
    output = Path(__file__).parent / "experiment_summary.csv"
    fields = ["experiment", "aligned_variable", "keras_macro_f1_mean", "pytorch_macro_f1_mean",
              "mean_gap", "mean_absolute_gap", "gap_change", "gap_reduction", "gap_reduction_rate"]
    if not rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
        print("Experiment results were not found.\nRun the training scripts first."); return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    print(f"Saved: {output}")


if __name__ == "__main__": main()
