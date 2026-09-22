"""Canonical, framework-neutral model/Adam checkpoint format and integrity checks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from common.controlled_config import (
    AUGMENTATION_VERSION, BATCH_SCHEDULE_VERSION, RESULTS, config_hash,
)
from common.trace_utils import assert_finite, state_hash

MANIFEST_FIELDS = [
    "seed", "framework", "checkpoint_name", "epoch", "optimizer_step",
    "model_file", "optimizer_file", "metadata_file", "model_sha256",
    "optimizer_sha256", "config_sha256",
]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def save_canonical_checkpoint(
    framework: str, seed: int, name: str, model_state: dict[str, np.ndarray],
    optimizer_state: dict[str, np.ndarray], *, epoch: int, optimizer_step: int,
    diagnostic: bool = False, results_root: Path = RESULTS,
    config_sha256: str | None = None,
) -> tuple[Path, Path, Path]:
    """Save copies only; no framework forward/backward/RNG calls are made."""
    assert_finite(model_state, "checkpoint model")
    assert_finite(optimizer_state, "checkpoint optimizer")
    if not model_state or not optimizer_state and optimizer_step:
        raise ValueError("Missing checkpoint state")
    if optimizer_step == 0 and optimizer_state:
        raise ValueError("Initial checkpoint must have empty Adam state")
    expected_config_hash = config_sha256 or config_hash()
    base = results_root / ("early_steps/checkpoints" if diagnostic else "checkpoints") / f"seed{seed}"
    folder = base / (f"step{optimizer_step:03d}" if diagnostic else framework)
    stem = framework if diagnostic else name
    model_path = folder / f"{stem}_state.npz" if diagnostic else folder / f"{stem}.npz"
    optimizer_path = folder / f"{stem}_optimizer.npz"
    metadata_path = folder / f"{stem}_metadata.json"
    targets = (model_path, optimizer_path, metadata_path)
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Checkpoint exists; overwrite disabled: {targets}")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, **model_state)
    np.savez_compressed(optimizer_path, **optimizer_state)
    metadata = {
        "schema": "canonical_model_adam_v1", "seed": seed, "framework": framework,
        "checkpoint_name": name, "epoch": epoch, "optimizer_step": optimizer_step,
        "batch_schedule_version": BATCH_SCHEDULE_VERSION,
        "augmentation_version": AUGMENTATION_VERSION,
        "config_sha256": expected_config_hash,
        "model_state_sha256": state_hash(model_state),
        "optimizer_state_sha256": state_hash(optimizer_state),
        "model_file_sha256": _file_hash(model_path),
        "optimizer_file_sha256": _file_hash(optimizer_path),
        "model_keys": sorted(model_state), "optimizer_keys": sorted(optimizer_state),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    verify_checkpoint(model_path, optimizer_path, metadata_path, expected_config_hash=expected_config_hash)
    if not diagnostic:
        _append_manifest(metadata, model_path, optimizer_path, metadata_path, results_root)
    return targets


def verify_checkpoint(
    model_path: Path, optimizer_path: Path, metadata_path: Path,
    *, expected_config_hash: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_state, optimizer_state = read_npz(model_path), read_npz(optimizer_path)
    checks = (
        metadata["config_sha256"] == (expected_config_hash or config_hash()),
        metadata["model_state_sha256"] == state_hash(model_state),
        metadata["optimizer_state_sha256"] == state_hash(optimizer_state),
        metadata["model_file_sha256"] == _file_hash(model_path),
        metadata["optimizer_file_sha256"] == _file_hash(optimizer_path),
        metadata["model_keys"] == sorted(model_state),
        metadata["optimizer_keys"] == sorted(optimizer_state),
    )
    if not all(checks):
        raise RuntimeError(f"Checkpoint integrity failure: {model_path}")
    return model_state, optimizer_state


def _append_manifest(
    metadata: dict[str, object], model_path: Path, optimizer_path: Path,
    metadata_path: Path, results_root: Path = RESULTS,
) -> None:
    path = results_root / "checkpoints" / "checkpoint_manifest.csv"
    row = {
        "seed": metadata["seed"], "framework": metadata["framework"],
        "checkpoint_name": metadata["checkpoint_name"], "epoch": metadata["epoch"],
        "optimizer_step": metadata["optimizer_step"],
        "model_file": str(model_path.relative_to(results_root)),
        "optimizer_file": str(optimizer_path.relative_to(results_root)),
        "metadata_file": str(metadata_path.relative_to(results_root)),
        "model_sha256": metadata["model_file_sha256"],
        "optimizer_sha256": metadata["optimizer_file_sha256"],
        "config_sha256": metadata["config_sha256"],
    }
    existing = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    if any(item["seed"] == str(row["seed"]) and item["framework"] == row["framework"] and item["checkpoint_name"] == row["checkpoint_name"] for item in existing):
        raise FileExistsError(f"Checkpoint manifest entry exists: {row}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)


def checkpoint_step(name: str, batches_per_epoch: int = 321) -> int:
    if name == "initial":
        return 0
    if name == "after_first_step":
        return 1
    if name.startswith("epoch_"):
        return int(name.split("_")[1]) * batches_per_epoch
    raise ValueError(name)
