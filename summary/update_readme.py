"""Update only the Phase 1 summary table; preserve curated results and Phase 2 notes."""
from __future__ import annotations
import csv, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.readme_utils import replace_marker


def pct(value: str | float) -> str:
    return f"{100 * float(value):.2f}%"


def main() -> None:
    summary_path = Path(__file__).parent / "experiment_summary.csv"
    subprocess.run([sys.executable, str(Path(__file__).parent / "compare_all_experiments.py")], check=True)
    if not summary_path.exists():
        print("Experiment results were not found.\nRun the training scripts first."); return
    with summary_path.open(encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    if not rows: print("Experiment results were not found.\nRun the training scripts first."); return
    lines = ["| Experiment | Aligned Variable | Keras F1 | PyTorch F1 | Signed Mean Gap | Mean Absolute Paired Gap | Status |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        experiment_id = row["experiment"][:2]
        label = {"00": "00 Baseline", "01": "01 Input", "02": "02 Batch", "03": "03 Augmentation"}[experiment_id]
        status = {"00": "Completed", "01": "Completed", "02": "Attempt 2 VALID", "03": "VALID"}[experiment_id]
        lines.append(f"| {label} | {row['aligned_variable']} | {pct(row['keras_macro_f1_mean'])} | "
                     f"{pct(row['pytorch_macro_f1_mean'])} | {pct(row['mean_gap'])} | "
                     f"{pct(row['mean_absolute_gap'])} | {status} |")
    replace_marker(ROOT / "README.md", "ABLATION_RESULTS", "\n".join(lines))
    print("Phase 1 ablation table updated; curated baseline, hypotheses, and Phase 2 notes preserved.")


if __name__ == "__main__": main()
