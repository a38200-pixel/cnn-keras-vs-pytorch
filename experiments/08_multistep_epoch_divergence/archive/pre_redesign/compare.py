"""Summarize this experiment and, for ablations, quantify change from baseline."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.comparison_utils import gap_effect, summarize


def main() -> None:
    current = summarize(Path(__file__).parent / "results")
    if current is None:
        return
    baseline_dir = ROOT / "experiments" / "00_baseline_final_cnn_3seed" / "results"
    if Path(__file__).parent.name != "00_baseline_final_cnn_3seed":
        baseline = summarize(baseline_dir)
        if baseline is None:
            print("Baseline results are required for gap-reduction analysis.")
        else:
            current.update(gap_effect(float(current["mean_absolute_gap"]), float(baseline["mean_absolute_gap"])))
    output = Path(__file__).parent / "results" / "comparison_summary.json"
    output.write_text(json.dumps(current, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
