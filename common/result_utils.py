"""Canonical result schema and idempotent CSV saving."""
from __future__ import annotations

import csv
import platform
import sys
from pathlib import Path

RESULT_FIELDS = [
    "experiment", "framework", "seed", "test_accuracy", "macro_f1", "test_loss",
    "best_epoch", "epochs_trained", "train_accuracy", "validation_accuracy",
    "train_loss", "validation_loss", "train_val_gap", "training_time_seconds",
    "python_version", "framework_version", "device", "architecture",
]


def save_result(path: Path, row: dict[str, object]) -> None:
    row = dict(row)
    framework = str(row.get("framework", "")).lower()
    if framework == "keras":
        import tensorflow as tf
        framework_version = tf.__version__
        device = "tensorflow-metal GPU:0" if tf.config.list_physical_devices("GPU") else "cpu"
    elif framework == "pytorch":
        import torch
        from common.environment_utils import select_torch_device
        framework_version = torch.__version__
        device = str(select_torch_device())
    else:
        framework_version, device = "unknown", "unknown"
    row.setdefault("python_version", sys.version.split()[0])
    row.setdefault("framework_version", framework_version)
    row.setdefault("device", device)
    row.setdefault("architecture", platform.machine())
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(RESULT_FIELDS) - set(row)
    if missing:
        raise ValueError(f"Missing result fields: {sorted(missing)}")
    existing: list[dict[str, object]] = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    # Re-running one seed replaces that seed instead of biasing the 3-seed mean.
    existing = [old for old in existing if not (
        str(old.get("experiment")) == str(row["experiment"])
        and str(old.get("framework")) == str(row["framework"])
        and str(old.get("seed")) == str(row["seed"])
    )]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow({key: row[key] for key in RESULT_FIELDS})
