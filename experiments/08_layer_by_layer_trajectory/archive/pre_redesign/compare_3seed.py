"""Aggregate per-layer metrics as mean ± sample standard deviation."""
from __future__ import annotations
import csv, json, statistics
from collections import defaultdict
from compare_initial import RESULTS


def main() -> None:
    files = sorted(RESULTS.glob("*_seed*.json"))
    if not files:
        print("Experiment results were not found.\nRun the diagnostic scripts first."); return
    grouped = defaultdict(list)
    for path in files:
        for row in json.loads(path.read_text(encoding="utf-8")): grouped[(row["stage"], row["layer"])].append(row)
    metrics = ["mae", "max_absolute_error", "mse", "cosine_similarity", "keras_mean", "pytorch_mean", "keras_std", "pytorch_std"]
    output = RESULTS / "layer_3seed_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["stage", "layer", "n_seeds"] + [f"{m}_{stat}" for m in metrics for stat in ("mean", "std")]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for (stage, layer), rows in grouped.items():
            out = {"stage": stage, "layer": layer, "n_seeds": len(rows)}
            for metric in metrics:
                values = [float(r[metric]) for r in rows]
                out[f"{metric}_mean"] = statistics.mean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            writer.writerow(out)
    print(f"Saved: {output}")


if __name__ == "__main__": main()
