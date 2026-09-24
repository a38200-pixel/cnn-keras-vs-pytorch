"""Generate epoch-level Keras/PyTorch trajectories from saved checkpoints only."""

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
EXPECTED_STEPS = [0, 1, 321, 1605, 3210, 6420, 9630]
EXPECTED_KEYS = {
    *(f"conv{index}/kernel" for index in range(1, 5)),
    *(f"bn{index}/{field}" for index in range(1, 5)
      for field in ("gamma", "beta", "mean", "variance")),
    "fc128/kernel", "fc128/bias", "logits/kernel", "logits/bias",
}


def _read_manifest() -> dict[tuple[int, str, str], dict[str, str]]:
    path = RESULTS / "checkpoints" / "checkpoint_manifest.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        (seed, framework, name)
        for seed in SEEDS
        for framework in ("keras", "pytorch")
        for name in CHECKPOINT_NAMES
    }
    indexed = {(int(row["seed"]), row["framework"], row["checkpoint_name"]): row for row in rows}
    if set(indexed) != expected or len(rows) != len(expected):
        raise RuntimeError("Experiment 05 checkpoint manifest is incomplete or contains unexpected entries")
    if any(row["config_sha256"] != config_hash() for row in rows):
        raise RuntimeError("Experiment 05 checkpoint manifest config hash mismatch")
    return indexed


def _load_state(row: dict[str, str], seed: int, framework: str, name: str) -> dict[str, np.ndarray]:
    model_path = RESULTS / row["model_file"]
    optimizer_path = RESULTS / row["optimizer_file"]
    metadata_path = RESULTS / row["metadata_file"]
    model, _optimizer = verify_checkpoint(
        model_path, optimizer_path, metadata_path,
        expected_config_hash=config_hash(),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_step = checkpoint_step(name)
    if int(row["optimizer_step"]) != expected_step or int(metadata["optimizer_step"]) != expected_step:
        raise RuntimeError(f"Checkpoint step mismatch: {framework} seed={seed} {name}")
    if int(metadata["seed"]) != seed or metadata["framework"] != framework:
        raise RuntimeError(f"Checkpoint identity mismatch: {framework} seed={seed} {name}")
    if set(model) != EXPECTED_KEYS:
        raise RuntimeError(f"Canonical parameter key mismatch: {framework} seed={seed} {name}")
    if any(not np.isfinite(value).all() for value in model.values()):
        raise RuntimeError(f"NaN/Inf in checkpoint: {framework} seed={seed} {name}")
    return model


def generate_trajectories() -> list[Path]:
    manifest = _read_manifest()
    outputs = []
    for seed in SEEDS:
        trajectory_rows = []
        for name in CHECKPOINT_NAMES:
            states = {
                framework: _load_state(manifest[(seed, framework, name)], seed, framework, name)
                for framework in ("keras", "pytorch")
            }
            keras_state, pytorch_state = states["keras"], states["pytorch"]
            keys = sorted(keras_state)
            trainable = [key for key in keys if not key.endswith(("/mean", "/variance"))]
            global_stats = compare_arrays(
                np.concatenate([keras_state[key].ravel() for key in trainable]),
                np.concatenate([pytorch_state[key].ravel() for key in trainable]),
            )
            row = {
                "seed": seed,
                "checkpoint": name,
                "optimizer_step": checkpoint_step(name),
                "config_sha256": config_hash(),
                "global_weight_relative_l2": global_stats["relative_l2_error"],
            }
            for key in keys:
                row[f"{key.replace('/', '_')}_relative_l2"] = compare_arrays(
                    keras_state[key], pytorch_state[key]
                )["relative_l2_error"]
            trajectory_rows.append(row)

        if [int(row["optimizer_step"]) for row in trajectory_rows] != EXPECTED_STEPS:
            raise RuntimeError(f"Unexpected trajectory steps: seed={seed}")
        initial_metrics = [
            float(value) for key, value in trajectory_rows[0].items()
            if key.endswith("_relative_l2")
        ]
        if any(value != 0.0 for value in initial_metrics):
            raise RuntimeError(f"Initial Keras/PyTorch state is not exact: seed={seed}")
        numeric_metrics = [
            float(value) for row in trajectory_rows for key, value in row.items()
            if key.endswith("_relative_l2")
        ]
        if not np.isfinite(np.asarray(numeric_metrics, dtype=np.float64)).all():
            raise RuntimeError(f"NaN/Inf in trajectory output: seed={seed}")

        destination = RESULTS / "trajectory" / f"seed{seed}_weight_trajectory.csv"
        write_protected_csv(destination, trajectory_rows)
        outputs.append(destination)
        print(
            f"seed={seed}: initial=0, steps={EXPECTED_STEPS}, finite=True -> {destination}",
            flush=True,
        )
    return outputs


if __name__ == "__main__":
    generate_trajectories()
    print("Experiment 05 trajectory post-processing complete; no training was run.")
