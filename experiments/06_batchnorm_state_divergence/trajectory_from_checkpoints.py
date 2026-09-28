"""Generate Experiment 06 epoch trajectories from completed checkpoints only."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_checkpoint import checkpoint_step, verify_checkpoint
from common.trace_utils import compare as compare_arrays, write_csv as write_protected_csv
from experiment_config import RESULTS, SEEDS, config_hash

CHECKPOINT_NAMES = [
    "initial", "after_first_step", "epoch_001", "epoch_005",
    "epoch_010", "epoch_020", "epoch_030",
]
EXPECTED_KEYS = {
    *(f"conv{index}/kernel" for index in range(1, 5)),
    *(f"bn{index}/{field}" for index in range(1, 5) for field in ("gamma", "beta", "mean", "variance")),
    "fc128/kernel", "fc128/bias", "logits/kernel", "logits/bias",
}


def _manifest():
    path = RESULTS / "checkpoints/checkpoint_manifest.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        (seed, framework, name)
        for seed in SEEDS for framework in ("keras", "pytorch") for name in CHECKPOINT_NAMES
    }
    indexed = {(int(row["seed"]), row["framework"], row["checkpoint_name"]): row for row in rows}
    if set(indexed) != expected or len(rows) != len(expected):
        raise RuntimeError("Experiment 06 checkpoint manifest is incomplete or unexpected")
    if any(row["config_sha256"] != config_hash() for row in rows):
        raise RuntimeError("Experiment 06 checkpoint config hash mismatch")
    return indexed


def generate_trajectories() -> list[Path]:
    manifest = _manifest()
    outputs = []
    for seed in SEEDS:
        trajectory = []
        for name in CHECKPOINT_NAMES:
            states = {}
            for framework in ("keras", "pytorch"):
                row = manifest[(seed, framework, name)]
                model, _optimizer = verify_checkpoint(
                    RESULTS / row["model_file"], RESULTS / row["optimizer_file"],
                    RESULTS / row["metadata_file"], expected_config_hash=config_hash(),
                )
                metadata = json.loads((RESULTS / row["metadata_file"]).read_text(encoding="utf-8"))
                if int(metadata["optimizer_step"]) != checkpoint_step(name):
                    raise RuntimeError(f"Checkpoint step mismatch: {framework} seed={seed} {name}")
                if set(model) != EXPECTED_KEYS:
                    raise RuntimeError(f"Canonical state mismatch: {framework} seed={seed} {name}")
                states[framework] = model
            keras, pytorch = states["keras"], states["pytorch"]
            keys = sorted(keras)
            trainable = [key for key in keys if not key.endswith(("/mean", "/variance"))]
            global_stats = compare_arrays(
                np.concatenate([keras[key].ravel() for key in trainable]),
                np.concatenate([pytorch[key].ravel() for key in trainable]),
            )
            row = {
                "seed": seed, "checkpoint": name, "optimizer_step": checkpoint_step(name),
                "config_sha256": config_hash(),
                "global_weight_relative_l2": global_stats["relative_l2_error"],
            }
            for key in keys:
                row[f"{key.replace('/', '_')}_relative_l2"] = compare_arrays(
                    keras[key], pytorch[key]
                )["relative_l2_error"]
            trajectory.append(row)
        destination = RESULTS / "trajectory" / f"seed{seed}_weight_trajectory.csv"
        write_protected_csv(destination, trajectory)
        outputs.append(destination)
    return outputs


if __name__ == "__main__":
    generate_trajectories()
