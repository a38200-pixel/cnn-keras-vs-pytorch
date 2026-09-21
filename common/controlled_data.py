"""Canonical sample metadata, batch order, and pixel-identical input batches."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common.augmentation_runtime import augmentation_parameters
from common.controlled_config import (
    BATCH_SIZE, IMAGE_SIZE, MAX_EPOCHS, RESULTS, ROTATION_DEGREES,
)
from common.dataset_utils import CLASSES, build_split_manifest, find_dataset_root


def canonical_rows(split: str) -> list[dict[str, object]]:
    root = find_dataset_root()
    rows = build_split_manifest()[split]
    result = [{
        "sample_id": Path(str(row["path"])).relative_to(root).as_posix(),
        "path": str(row["path"]), "label": int(row["label"]),
    } for row in rows]
    result.sort(key=lambda row: (row["label"], row["sample_id"]))
    if len({row["sample_id"] for row in result}) != len(result):
        raise ValueError(f"Duplicate sample identity in {split}.")
    if sorted({row["label"] for row in result}) != list(range(len(CLASSES))):
        raise ValueError(f"Class mapping mismatch in {split}.")
    return result


def save_sample_manifest(rows: list[dict[str, object]]) -> Path:
    path = RESULTS / "control" / "canonical_train_samples.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "relative_path", "label"])
        writer.writeheader()
        writer.writerows({
            "sample_id": index, "relative_path": row["sample_id"], "label": row["label"],
        } for index, row in enumerate(rows))
    return path


def order_path(seed: int) -> Path:
    return RESULTS / "control" / "batch_order" / f"seed{seed}_epoch_permutations.npz"


def create_orders(seed: int, length: int, save: bool = True) -> np.ndarray:
    # Namespaced RNG: changing weight initialization cannot perturb batch order.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x0A11CE]))
    orders = np.stack([rng.permutation(length) for _ in range(MAX_EPOCHS)]).astype(np.int32)
    path = order_path(seed)
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path) as archive:
                previous = archive["orders"]
            if not np.array_equal(previous, orders):
                raise RuntimeError(f"Persisted order differs from deterministic schedule: {path}")
        else:
            np.savez_compressed(path, orders=orders)
    return orders


def load_orders(seed: int, length: int) -> np.ndarray:
    path = order_path(seed)
    with np.load(path) as archive:
        observed = archive["orders"]
    expected = create_orders(seed, length, save=False)
    if not np.array_equal(observed, expected):
        raise RuntimeError(f"Order validation failed: {path}")
    return observed


def iter_batches(order: np.ndarray):
    for start in range(0, len(order), BATCH_SIZE):
        yield order[start:start + BATCH_SIZE]


def canonical_image(row: dict[str, object], seed: int, epoch: int, training: bool) -> np.ndarray:
    with Image.open(str(row["path"])) as source:
        image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    if training:
        flip, angle = augmentation_parameters(str(row["sample_id"]), seed, epoch)
        if flip:
            image = ImageOps.mirror(image)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def canonical_batch(rows: list[dict[str, object]], indices, seed: int, epoch: int, training: bool):
    chosen = [rows[int(index)] for index in indices]
    images = np.stack([canonical_image(row, seed, epoch, training) for row in chosen])
    labels = np.asarray([row["label"] for row in chosen], dtype=np.int32)
    return np.ascontiguousarray(images), labels


def tensor_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def numpy_reference_ce(logits: np.ndarray, labels: np.ndarray) -> float:
    x = logits.astype(np.float64)
    shifted = x - x.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(logsumexp - shifted[np.arange(len(labels)), labels]))
