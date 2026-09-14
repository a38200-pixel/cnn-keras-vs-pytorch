"""Shared per-epoch history schema for Keras and PyTorch experiments."""

from __future__ import annotations

import csv
from pathlib import Path


HISTORY_FIELDS = [
    "framework",
    "experiment",
    "seed",
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "learning_rate",
    "elapsed_epoch_sec",
]


def save_history(path: Path, rows: list[dict[str, object]]) -> None:
    """Save one Seed immediately after training using the common schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in HISTORY_FIELDS} for row in rows)
