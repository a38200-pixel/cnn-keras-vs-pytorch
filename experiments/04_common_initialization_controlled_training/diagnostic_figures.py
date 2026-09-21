"""Generate compact read-only plots from saved diagnostic CSVs."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cnn-controlled-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.controlled_config import RESULTS, SEEDS


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(path: Path, fig) -> None:
    if path.exists():
        raise FileExistsError(f"Diagnostic figure exists; overwrite disabled: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    output = RESULTS / "figures" / "diagnostic"
    for seed in SEEDS:
        for kind, source, key in (
            ("forward", "forward_distribution", "layer"),
            ("gradient", "gradient_distribution", "parameter"),
            ("update", "update_distribution", "parameter"),
        ):
            data = rows(RESULTS / "first_step" / f"seed{seed}_{source}.csv")
            labels = [row[key] for row in data]
            values = [float(row["relative_l2_error"]) for row in data]
            fig, ax = plt.subplots(figsize=(max(9, len(labels) * .45), 4.8), constrained_layout=True)
            ax.bar(range(len(labels)), values)
            ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=8)
            ax.set_ylabel("Relative L2 error")
            ax.set_title(f"Seed {seed} first-step {kind} difference")
            ax.grid(axis="y", alpha=.25)
            save(output / f"first_step_{kind}_diff_seed{seed}.png", fig)
        summary = rows(RESULTS / "early_steps" / f"seed{seed}_step_summary.csv")
        layers = rows(RESULTS / "early_steps" / f"seed{seed}_layer_trajectory.csv")
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        ax.plot([int(row["step"]) for row in summary],
                [float(row["global_weight_relative_l2"]) for row in summary],
                marker="o", label="global")
        for name in ("conv1/kernel", "conv4/kernel", "fc128/kernel", "logits/kernel"):
            selected = [row for row in layers if row["layer"] == name]
            ax.plot([int(row["step"]) for row in selected],
                    [float(row["weight_relative_l2"]) for row in selected],
                    marker=".", label=name)
        ax.set(xlabel="Optimizer step", ylabel="Relative L2 distance",
               title=f"Seed {seed} isolated early-step weight trajectory")
        ax.grid(alpha=.25)
        ax.legend()
        save(output / f"early_step_weight_divergence_seed{seed}.png", fig)
    print("Diagnostic figures generated from saved traces; no training was run.")


if __name__ == "__main__":
    main()
