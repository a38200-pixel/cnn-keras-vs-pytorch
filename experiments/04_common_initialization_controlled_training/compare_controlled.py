"""Read-only comparison of completed Phase-2 training, never trains."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cnn-controlled-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.controlled_config import MAX_EPOCHS, RESULTS, SEEDS, config_hash
from common.controlled_checkpoint import checkpoint_step, verify_checkpoint
from common.trace_utils import compare as compare_arrays, write_csv as write_protected_csv
from common.controlled_training_utils import require_valid_preflight


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    require_valid_preflight()
    traces = {}
    for seed in SEEDS:
        path = RESULTS / "first_step" / f"seed{seed}_summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing first-step trace: {path}")
        traces[str(seed)] = json.loads(path.read_text(encoding="utf-8"))
    results = {}
    histories = {}
    for framework in ("keras", "pytorch"):
        path = RESULTS / f"{framework}_controlled_results.csv"
        if not path.exists():
            print(f"Training pending: {path}; no performance summary generated.")
            return
        rows = read_csv(path)
        if sorted(int(row["seed"]) for row in rows) != list(SEEDS):
            raise RuntimeError(f"Incomplete 3-Seed controlled result: {path}")
        results[framework] = {int(row["seed"]): row for row in rows}
        histories[framework] = {}
        for seed in SEEDS:
            row = results[framework][seed]
            if row.get("config_sha256") != config_hash():
                raise RuntimeError(f"Controlled result config hash mismatch: {framework} seed={seed}")
            history_path = RESULTS / "history" / f"{framework}_seed{seed}_history.csv"
            history = read_csv(history_path)
            if int(row["epochs_trained"]) != MAX_EPOCHS or len(history) != MAX_EPOCHS:
                raise RuntimeError(f"Fixed epoch count mismatch: {framework} seed={seed}")
            if int(row["optimizer_steps"]) != 9630 or int(history[-1]["num_optimizer_steps"]) != 9630:
                raise RuntimeError(f"Optimizer step count mismatch: {framework} seed={seed}")
            if any(float(item["learning_rate"]) != 0.001 for item in history):
                raise RuntimeError(f"Nonconstant LR: {framework} seed={seed}")
            if any(item.get("config_sha256") != config_hash() for item in history):
                raise RuntimeError(f"History config hash mismatch: {framework} seed={seed}")
            if [int(item["num_optimizer_steps"]) for item in history] != [321 * epoch for epoch in range(1, MAX_EPOCHS + 1)]:
                raise RuntimeError(f"History global-step schedule mismatch: {framework} seed={seed}")
            histories[framework][seed] = history
    checkpoint_names = ["initial", "after_first_step", "epoch_001", "epoch_005", "epoch_010", "epoch_020", "epoch_030"]
    manifest_path = RESULTS / "checkpoints" / "checkpoint_manifest.csv"
    manifest_rows = read_csv(manifest_path)
    expected_entries = {(str(seed), framework, name) for seed in SEEDS for framework in ("keras", "pytorch") for name in checkpoint_names}
    if {(row["seed"], row["framework"], row["checkpoint_name"]) for row in manifest_rows} != expected_entries:
        raise RuntimeError("Controlled checkpoint manifest is incomplete or contains unexpected entries")
    if any(row["config_sha256"] != config_hash() for row in manifest_rows):
        raise RuntimeError("Checkpoint manifest config hash mismatch")
    trajectory_files = []
    for seed in SEEDS:
        trajectory_rows = []
        for name in checkpoint_names:
            states = {}
            for framework in ("keras", "pytorch"):
                folder = RESULTS / "checkpoints" / f"seed{seed}" / framework
                model, optimizer = verify_checkpoint(
                    folder / f"{name}.npz", folder / f"{name}_optimizer.npz",
                    folder / f"{name}_metadata.json",
                )
                meta = json.loads((folder / f"{name}_metadata.json").read_text(encoding="utf-8"))
                if int(meta["optimizer_step"]) != checkpoint_step(name) or int(meta["seed"]) != seed:
                    raise RuntimeError(f"Checkpoint step/seed mismatch: {framework} {seed} {name}")
                states[framework] = model
            ka, pa = states["keras"], states["pytorch"]
            if set(ka) != set(pa):
                raise RuntimeError(f"Checkpoint key mismatch: seed={seed} {name}")
            keys = sorted(ka)
            trainable = [key for key in keys if not key.endswith(("/mean", "/variance"))]
            global_stats = compare_arrays(
                np.concatenate([ka[key].ravel() for key in trainable]),
                np.concatenate([pa[key].ravel() for key in trainable]),
            )
            row = {"seed": seed, "checkpoint": name, "optimizer_step": checkpoint_step(name),
                   "config_sha256": config_hash(), "global_weight_relative_l2": global_stats["relative_l2_error"]}
            for key in keys:
                row[f"{key.replace('/', '_')}_relative_l2"] = compare_arrays(ka[key], pa[key])["relative_l2_error"]
            trajectory_rows.append(row)
        destination = RESULTS / "trajectory" / f"seed{seed}_weight_trajectory.csv"
        write_protected_csv(destination, trajectory_rows)
        trajectory_files.append(str(destination.relative_to(RESULTS)))
    paired = {}
    for metric in ("final_test_accuracy", "final_macro_f1", "final_test_loss", "best_test_accuracy", "best_macro_f1", "best_test_loss"):
        keras = np.asarray([float(results["keras"][seed][metric]) for seed in SEEDS])
        pytorch = np.asarray([float(results["pytorch"][seed][metric]) for seed in SEEDS])
        difference = pytorch - keras
        paired[metric] = {
            "keras_mean": float(keras.mean()), "keras_sample_std": float(keras.std(ddof=1)),
            "pytorch_mean": float(pytorch.mean()), "pytorch_sample_std": float(pytorch.std(ddof=1)),
            "paired_by_seed": dict(zip(map(str, SEEDS), map(float, difference))),
            "signed_mean_gap": float(difference.mean()),
            "mean_absolute_paired_gap": float(np.abs(difference).mean()),
        }
    figures = []
    figure_dir = RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("loss", "accuracy"):
        fig, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
        for framework in ("keras", "pytorch"):
            for seed in SEEDS:
                history = histories[framework][seed]
                axis.plot(
                    [int(item["epoch"]) for item in history],
                    [float(item[f"val_{metric}"]) for item in history],
                    label=f"{framework} seed {seed}",
                    linestyle="-" if framework == "keras" else "--",
                )
        axis.set(xlabel="Epoch", ylabel=f"Validation {metric}", title=f"Controlled Epoch-30 Validation {metric}")
        axis.grid(alpha=.25)
        axis.legend(ncol=2)
        output = figure_dir / f"controlled_validation_{metric}.png"
        fig.savefig(output, dpi=180)
        plt.close(fig)
        figures.append(str(output.relative_to(RESULTS)))
    summary = {
        "phase": "strict_controlled_comparison", "experiment_validity": "VALID",
        "primary_checkpoint": "fixed_epoch_30", "secondary_checkpoint": "best_validation_loss",
        "first_step_diagnostic": {
            seed: {"first_nonzero_forward_layer": trace["first_nonzero_forward_layer"]}
            for seed, trace in traces.items()
        },
        "metrics": paired, "figures": figures, "trajectory_files": trajectory_files,
        "config_sha256": config_hash(),
        "interpretation_note": "Phase 2 is not an OFAT gap-reduction continuation of 00–03.",
    }
    output = RESULTS / "controlled_comparison_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved controlled comparison: {output}")


if __name__ == "__main__":
    main()
