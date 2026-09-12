"""Update generated marker regions only; never rebuild the hand-written README."""
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
    baseline = next((r for r in rows if r["experiment"].startswith("00_")), None)
    if baseline:
        baseline_text = ("| Keras Macro F1 mean | PyTorch Macro F1 mean | Mean Gap | Mean Absolute Gap |\n"
                         "|---:|---:|---:|---:|\n" +
                         f"| {pct(baseline['keras_macro_f1_mean'])} | {pct(baseline['pytorch_macro_f1_mean'])} | "
                         f"{pct(baseline['mean_gap'])} | {pct(baseline['mean_absolute_gap'])} |")
        replace_marker(ROOT / "README.md", "BASELINE_RESULTS", baseline_text)
    lines = ["| Experiment | Aligned Variable | Keras F1 | PyTorch F1 | Abs. Gap | Gap Reduction vs Baseline |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        reduction = "0" if row["experiment"].startswith("00_") else pct(row.get("gap_reduction", "nan"))
        lines.append(f"| {row['experiment']} | {row['aligned_variable']} | {pct(row['keras_macro_f1_mean'])} | "
                     f"{pct(row['pytorch_macro_f1_mean'])} | {pct(row['mean_absolute_gap'])} | {reduction} |")
    replace_marker(ROOT / "README.md", "ABLATION_RESULTS", "\n".join(lines))
    layer_path = ROOT / "experiments" / "09_layer_by_layer" / "results" / "layer_3seed_summary.csv"
    if layer_path.exists():
        with layer_path.open(encoding="utf-8") as handle: layer_rows = list(csv.DictReader(handle))
        layer_lines = ["| Stage | Layer | MAE mean±std | Cosine mean±std |", "|---|---|---:|---:|"]
        for row in layer_rows:
            layer_lines.append(f"| {row['stage']} | {row['layer']} | {float(row['mae_mean']):.3e} ± {float(row['mae_std']):.3e} | "
                               f"{float(row['cosine_similarity_mean']):.8f} ± {float(row['cosine_similarity_std']):.3e} |")
        replace_marker(ROOT / "README.md", "LAYER_RESULTS", "\n".join(layer_lines))
    # Decisions remain conservative: the script reports evidence, not causality.
    directions = [] if not baseline else [float(baseline["mean_gap"])]
    # Statistical/causal decisions are deliberately not automated from a mean
    # alone. The generated table exposes evidence and requests interpretation.
    h0 = "Pending" if len(rows) < 9 else "Review required"
    h1 = "Pending" if not directions else "Review required"
    h2 = "Pending" if len(rows) < 9 else "Review required"
    h3 = "Pending" if not layer_path.exists() else "Review required"
    hypothesis = ("| Hypothesis | Result | Evidence |\n|---|---|---|\n"
                  f"| H0 | {h0} | Seed별 방향과 변동성 확인 필요 |\n"
                  f"| H1 | {h1} | Baseline Mean Gap={pct(directions[0]) if directions else '-'} |\n"
                  f"| H2 | {h2} | branch별 gap reduction 표 참조 |\n"
                  f"| H3 | {h3} | layer 표 참조 |")
    replace_marker(ROOT / "README.md", "HYPOTHESIS_RESULTS", hypothesis)
    print("README marker regions updated; all other prose preserved.")


if __name__ == "__main__": main()
