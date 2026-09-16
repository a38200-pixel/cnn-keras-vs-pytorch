"""Shared, persisted batch-order schedule for Experiment 02."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np


RUNTIME_ORDER_FIELDS = [
    "framework", "seed", "epoch", "schedule_index", "num_samples", "order_hash",
]


def canonical_train_rows(rows: list[dict[str, object]], dataset_root: Path) -> list[dict[str, object]]:
    """Sort by class index and dataset-relative path to create one canonical list."""
    return sorted(
        rows,
        key=lambda row: (
            int(row["label"]),
            Path(str(row["path"])).relative_to(dataset_root).as_posix(),
        ),
    )


def generate_epoch_orders(length: int, seed: int, epochs: int) -> np.ndarray:
    """Generate all epoch permutations from one framework-neutral NumPy RNG stream."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(length) for _ in range(epochs)]).astype(np.int64)


def order_hash(order: np.ndarray | list[int]) -> str:
    text = ",".join(str(int(index)) for index in order)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def batch_groups(order: np.ndarray | list[int], batch_size: int) -> list[list[int]]:
    values = [int(index) for index in order]
    return [values[start:start + batch_size] for start in range(0, len(values), batch_size)]


def runtime_order_path(output_dir: Path, framework: str, seed: int) -> Path:
    return output_dir / f"{framework}_seed{seed}_runtime_order.csv"


def initialize_runtime_order_log(output_dir: Path, framework: str, seed: int) -> Path:
    """Start a fresh per-seed runtime trace immediately before a training run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_order_path(output_dir, framework, seed)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RUNTIME_ORDER_FIELDS).writeheader()
    return path


def record_runtime_order(
    output_dir: Path,
    framework: str,
    seed: int,
    epoch: int,
    schedule_index: int,
    order: np.ndarray | list[int],
    canonical: list[dict[str, object]],
    dataset_root: Path,
    batch_size: int,
    preview_batches: int = 3,
) -> str:
    """Record the order selected at the real epoch boundary and a short preview."""
    values = [int(index) for index in order]
    if schedule_index != epoch - 1:
        raise RuntimeError(
            f"Runtime schedule mismatch: epoch={epoch}, schedule_index={schedule_index}"
        )
    if len(values) != len(canonical) or sorted(values) != list(range(len(canonical))):
        raise RuntimeError("Runtime order is not a complete canonical-sample permutation.")

    path = runtime_order_path(output_dir, framework, seed)
    if not path.exists():
        initialize_runtime_order_log(output_dir, framework, seed)
    with path.open(encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))
    if any(int(row["epoch"]) == epoch for row in existing):
        raise RuntimeError(f"Duplicate runtime order record: {framework} seed={seed} epoch={epoch}")

    digest = order_hash(values)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_ORDER_FIELDS)
        writer.writerow({
            "framework": framework,
            "seed": seed,
            "epoch": epoch,
            "schedule_index": schedule_index,
            "num_samples": len(values),
            "order_hash": digest,
        })

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{framework}_seed{seed}_epoch{epoch:02d}_preview.csv"
    preview_fields = [
        "framework", "seed", "epoch", "schedule_index", "batch_index",
        "position_in_batch", "sample_index", "relative_path",
    ]
    with preview_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=preview_fields)
        writer.writeheader()
        for position, sample_index in enumerate(values[:batch_size * preview_batches]):
            writer.writerow({
                "framework": framework,
                "seed": seed,
                "epoch": epoch,
                "schedule_index": schedule_index,
                "batch_index": position // batch_size + 1,
                "position_in_batch": position % batch_size,
                "sample_index": sample_index,
                "relative_path": Path(str(canonical[sample_index]["path"])).relative_to(
                    dataset_root
                ).as_posix(),
            })
    print(
        f"Runtime Order | framework={framework} seed={seed} epoch={epoch:02d} "
        f"schedule_index={schedule_index} samples={len(values)} hash={digest}",
        flush=True,
    )
    return digest


def prepare_batch_order_artifacts(
    rows: list[dict[str, object]],
    dataset_root: Path,
    output_dir: Path,
    seeds: list[int],
    epochs: int,
) -> tuple[list[dict[str, object]], dict[int, np.ndarray]]:
    """Create reproducibility artifacts once, then load and verify those exact files."""
    canonical = canonical_train_rows(rows, dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "canonical_train_samples.csv"
    expected_rows = [
        {
            "sample_index": index,
            "relative_path": Path(str(row["path"])).relative_to(dataset_root).as_posix(),
            "class_name": str(row["class_name"]),
            "class_index": int(row["label"]),
        }
        for index, row in enumerate(canonical)
    ]
    if canonical_path.exists():
        with canonical_path.open(encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        normalized = [
            {
                "sample_index": int(row["sample_index"]),
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "class_index": int(row["class_index"]),
            }
            for row in existing
        ]
        if normalized != expected_rows:
            raise RuntimeError(f"Canonical sample list mismatch: {canonical_path}")
    else:
        with canonical_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=expected_rows[0].keys())
            writer.writeheader()
            writer.writerows(expected_rows)

    schedules: dict[int, np.ndarray] = {}
    for seed in seeds:
        expected = generate_epoch_orders(len(canonical), seed, epochs)
        schedule_path = output_dir / f"seed_{seed}_epoch_orders.npz"
        if not schedule_path.exists():
            np.savez_compressed(schedule_path, orders=expected, seed=seed, epochs=epochs)
        with np.load(schedule_path) as archive:
            loaded = archive["orders"].astype(np.int64, copy=False)
        if loaded.shape != expected.shape or not np.array_equal(loaded, expected):
            raise RuntimeError(f"Batch-order schedule mismatch: {schedule_path}")
        schedules[seed] = loaded

        preview_path = output_dir / f"seed{seed}_epoch1_preview.csv"
        preview_rows = [
            {
                "position": position,
                "sample_index": int(sample_index),
                "relative_path": expected_rows[int(sample_index)]["relative_path"],
                "class_name": expected_rows[int(sample_index)]["class_name"],
                "class_index": expected_rows[int(sample_index)]["class_index"],
            }
            for position, sample_index in enumerate(loaded[0, :64])
        ]
        with preview_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=preview_rows[0].keys())
            writer.writeheader()
            writer.writerows(preview_rows)

    return canonical, schedules
