"""CIFAR-10 local-only loading, deterministic split, order, and batch utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from v4_config import (
    BATCH_SIZE, EPOCHS, NUM_CLASSES, SEEDS, SPLIT_SEED, TEST_SIZE,
    TRAIN_SIZE, VAL_SIZE,
)

MANUAL_DOWNLOAD = (
    "source .venv-metal/bin/activate\n"
    "python -u experiments/external_validation/V4_cifar10/src/download_cifar10.py "
    "--download --data-root data/cifar10"
)


def require_local_cifar10(data_root: Path):
    from torchvision.datasets import CIFAR10

    try:
        train = CIFAR10(root=str(data_root), train=True, download=False)
        test = CIFAR10(root=str(data_root), train=False, download=False)
    except (RuntimeError, FileNotFoundError) as error:
        raise SystemExit(f"CIFAR-10 is not available locally. Run:\n{MANUAL_DOWNLOAD}") from error
    if len(train) != 50_000 or len(test) != TEST_SIZE:
        raise RuntimeError("Unexpected CIFAR-10 official split size")
    return train, test


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def stratified_split(targets) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([SPLIT_SEED, 0xC1FA10]))
    train_parts, val_parts = [], []
    for class_index in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == class_index)
        if indices.size != 5_000:
            raise RuntimeError(f"Unexpected CIFAR-10 class count: {class_index}")
        shuffled = indices[rng.permutation(indices.size)]
        train_parts.append(shuffled[:4_500])
        val_parts.append(shuffled[4_500:])
    train_indices = np.sort(np.concatenate(train_parts)).astype(np.int64)
    val_indices = np.sort(np.concatenate(val_parts)).astype(np.int64)
    if train_indices.size != TRAIN_SIZE or val_indices.size != VAL_SIZE:
        raise RuntimeError("V4 split size mismatch")
    if np.intersect1d(train_indices, val_indices).size:
        raise RuntimeError("V4 split overlap")
    return train_indices, val_indices


def epoch_order(train_indices: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    if seed not in SEEDS or not 1 <= epoch <= EPOCHS:
        raise ValueError("Non-canonical V4 seed/epoch")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0x0D3E]))
    order = train_indices[rng.permutation(train_indices.size)]
    if order.size != TRAIN_SIZE or np.unique(order).size != TRAIN_SIZE:
        raise RuntimeError("V4 epoch order is not a complete permutation")
    return order


def order_manifest(train_indices: np.ndarray, seed: int) -> dict[str, object]:
    rows = []
    for epoch in range(1, EPOCHS + 1):
        order = epoch_order(train_indices, seed, epoch)
        rows.append({
            "epoch": epoch, "sample_count": int(order.size),
            "unique_count": int(np.unique(order).size),
            "permutation_sha256": sha256_array(order),
        })
    payload = {
        "seed": seed, "epochs": EPOCHS, "sample_count_per_epoch": TRAIN_SIZE,
        "orders": rows,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def canonical_images(dataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.ascontiguousarray(np.asarray(dataset.data)[indices], dtype=np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.int64)[indices]
    images = np.ascontiguousarray(raw.astype(np.float32) / np.float32(255.0))
    return raw, images, labels


def batches(order: np.ndarray):
    for start in range(0, order.size, BATCH_SIZE):
        yield order[start:start + BATCH_SIZE]
