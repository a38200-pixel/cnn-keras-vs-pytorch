"""Deterministic augmentation parameters and actual-runtime trace utilities."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


RUNTIME_AUG_FIELDS = [
    "framework", "seed", "epoch", "schedule_index", "mode", "num_samples",
    "parameter_hash", "interpolation", "fill_mode",
]


def sample_identity(path: str | Path, dataset_root: str | Path) -> str:
    """Return a machine-independent identity such as train/anger/image.jpg."""
    return Path(path).resolve().relative_to(Path(dataset_root).resolve()).as_posix()


def augmentation_parameters(identity: str, seed: int, epoch_index: int) -> tuple[bool, float]:
    """Map (seed, epoch, sample identity) to flip and Uniform[-5, 5] angle."""
    digest = hashlib.sha256(f"{seed}:{epoch_index}:{identity}".encode("utf-8")).digest()
    flip = digest[0] < 128
    unit = int.from_bytes(digest[1:9], "big") / (2**64 - 1)
    return flip, -5.0 + 10.0 * unit


def parameter_hash(records: dict[str, tuple[bool, float]]) -> str:
    """Hash identity + the exact binary64 augmentation parameters, order-independently."""
    digest = hashlib.sha256()
    for identity in sorted(records):
        flip, angle = records[identity]
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"1" if flip else b"0")
        digest.update(b"\0")
        digest.update(float(angle).hex().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def expected_records(identities: list[str], seed: int, epoch_index: int) -> dict[str, tuple[bool, float]]:
    return {
        identity: augmentation_parameters(identity, seed, epoch_index)
        for identity in identities
    }


def runtime_summary_path(directory: Path, framework: str, seed: int) -> Path:
    return directory / f"{framework}_seed{seed}_runtime_aug.csv"


def runtime_values_path(directory: Path, framework: str, seed: int) -> Path:
    return directory / f"{framework}_seed{seed}_runtime_aug.npz"


class AugmentationRuntimeRecorder:
    """Capture parameters from Dataset/Sequence __getitem__, not from a simulation."""

    def __init__(
        self,
        framework: str,
        seed: int,
        identities: list[str],
        output_dir: Path | None,
    ) -> None:
        if len(identities) != len(set(identities)):
            raise ValueError("Augmentation identities must be unique.")
        self.framework = framework
        self.seed = seed
        self.identities = sorted(identities)
        self.identity_set = set(identities)
        self.output_dir = output_dir
        self.active_epoch: int | None = None
        self.active: dict[str, tuple[bool, float]] = {}
        self.completed: list[tuple[int, dict[str, tuple[bool, float]]]] = []

    def begin_epoch(self, epoch_index: int) -> None:
        if self.active_epoch is not None:
            raise RuntimeError(f"Epoch {self.active_epoch} was not finalized.")
        self.active_epoch = epoch_index
        self.active = {}

    def record(self, identity: str, flip: bool, angle: float) -> None:
        if self.active_epoch is None:
            return  # Keras may inspect a batch before the real epoch begins.
        if identity not in self.identity_set:
            raise RuntimeError(f"Unexpected augmentation identity: {identity}")
        value = (bool(flip), float(angle))
        previous = self.active.get(identity)
        if previous is not None and previous != value:
            raise RuntimeError(f"Inconsistent repeated augmentation parameters: {identity}")
        self.active[identity] = value

    def finalize_epoch(self) -> dict[str, object]:
        if self.active_epoch is None:
            raise RuntimeError("No active augmentation epoch to finalize.")
        missing = self.identity_set - set(self.active)
        if missing:
            preview = sorted(missing)[:3]
            raise RuntimeError(
                f"Runtime augmentation did not observe {len(missing)} samples; first={preview}"
            )
        expected = expected_records(self.identities, self.seed, self.active_epoch)
        if self.active != expected:
            mismatches = [key for key in self.identities if self.active[key] != expected[key]]
            raise RuntimeError(f"Runtime augmentation schedule mismatch: {mismatches[:3]}")
        row = {
            "framework": self.framework,
            "seed": self.seed,
            "epoch": self.active_epoch + 1,
            "schedule_index": self.active_epoch,
            "mode": "strict_sample_id",
            "num_samples": len(self.active),
            "parameter_hash": parameter_hash(self.active),
            "interpolation": "bilinear",
            "fill_mode": "constant_0",
        }
        self.completed.append((self.active_epoch, dict(self.active)))
        self.active_epoch = None
        self.active = {}
        if self.output_dir is not None:
            self._save()
        return row

    def _save(self) -> None:
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        flips, angles, epochs = [], [], []
        for epoch_index, records in self.completed:
            rows.append({
                "framework": self.framework,
                "seed": self.seed,
                "epoch": epoch_index + 1,
                "schedule_index": epoch_index,
                "mode": "strict_sample_id",
                "num_samples": len(records),
                "parameter_hash": parameter_hash(records),
                "interpolation": "bilinear",
                "fill_mode": "constant_0",
            })
            epochs.append(epoch_index)
            flips.append([records[key][0] for key in self.identities])
            angles.append([records[key][1] for key in self.identities])
        summary = runtime_summary_path(self.output_dir, self.framework, self.seed)
        with summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RUNTIME_AUG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(
            runtime_values_path(self.output_dir, self.framework, self.seed),
            identities=np.asarray(self.identities),
            epoch_indices=np.asarray(epochs, dtype=np.int16),
            flips=np.asarray(flips, dtype=np.uint8),
            angles=np.asarray(angles, dtype=np.float64),
        )


def apply_tensorflow_augmentation(image, flip: bool, angle_deg: float):
    """Apply explicit flip then CCW rotation to an HWC TensorFlow image."""
    import math
    import tensorflow as tf

    image = tf.cast(image, tf.float32)
    if flip:
        image = tf.image.flip_left_right(image)
    angle = math.radians(float(angle_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    center = (float(image.shape[1]) - 1.0) / 2.0
    # ImageProjectiveTransform maps output coordinates back to input coordinates.
    transform = [[
        cosine, -sine, center - cosine * center + sine * center,
        sine, cosine, center - sine * center - cosine * center,
        0.0, 0.0,
    ]]
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.constant(transform, dtype=tf.float32),
        output_shape=tf.constant([int(image.shape[0]), int(image.shape[1])]),
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=tf.constant(0.0, dtype=tf.float32),
    )
    return rotated[0]


def apply_torchvision_augmentation(image, flip: bool, angle_deg: float):
    """Apply the same semantics with torchvision's native PIL operators."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as functional

    if flip:
        image = functional.hflip(image)
    return functional.rotate(
        image, float(angle_deg), interpolation=InterpolationMode.BILINEAR, fill=0,
    )


def numerical_operator_diagnostic(paths: list[str], seed: int, epoch_index: int, dataset_root: Path) -> dict[str, object]:
    """Compare operators on common diagnostic pixels without changing training input."""
    from PIL import Image

    per_sample = []
    for path in paths:
        identity = sample_identity(path, dataset_root)
        flip, angle = augmentation_parameters(identity, seed, epoch_index)
        canonical = Image.open(path).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        keras = apply_tensorflow_augmentation(np.asarray(canonical, dtype=np.float32), flip, angle).numpy()
        torch = np.asarray(apply_torchvision_augmentation(canonical, flip, angle), dtype=np.float32)
        difference = keras - torch
        per_sample.append({
            "sample_identity": identity,
            "flip_applied": flip,
            "rotation_angle_deg": angle,
            "mae": float(np.mean(np.abs(difference))),
            "mse": float(np.mean(np.square(difference))),
            "max_absolute_difference": float(np.max(np.abs(difference))),
        })
    return {
        "scope": "diagnostic-only common PIL-resized pixels; training pipelines unchanged",
        "seed": seed,
        "epoch_index": epoch_index,
        "num_samples": len(per_sample),
        "mae_mean": float(np.mean([row["mae"] for row in per_sample])),
        "mse_mean": float(np.mean([row["mse"] for row in per_sample])),
        "max_absolute_difference": float(max(row["max_absolute_difference"] for row in per_sample)),
        "samples": per_sample,
    }


def save_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
