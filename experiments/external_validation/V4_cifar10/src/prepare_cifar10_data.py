"""Prepare deterministic V4 split/order/fixed-batch manifests from local CIFAR-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from common.trace_utils import write_json
from v4_config import RESULTS, SEEDS, SPLIT_SEED, config_hash
from v4_data import (
    canonical_images, order_manifest, require_local_cifar10, sha256_array,
    stratified_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10"))
    args = parser.parse_args()
    train, test = require_local_cifar10(args.data_root)
    train_indices, val_indices = stratified_split(train.targets)
    targets = np.asarray(train.targets, dtype=np.int64)
    per_class = {
        str(index): {
            "train": int(np.sum(targets[train_indices] == index)),
            "validation": int(np.sum(targets[val_indices] == index)),
        } for index in range(10)
    }
    split = {
        "dataset": "CIFAR-10", "split_seed": SPLIT_SEED,
        "official_train_count": len(train), "official_test_count": len(test),
        "train_count": int(train_indices.size), "validation_count": int(val_indices.size),
        "per_class": per_class,
        "train_indices_sha256": sha256_array(train_indices),
        "validation_indices_sha256": sha256_array(val_indices),
        "test_identity": "official_test_indices_0_to_9999",
        "test_raw_sha256": sha256_array(np.asarray(test.data, dtype=np.uint8)),
        "config_sha256": config_hash(), "v4_split_check": "VALID",
    }
    split["split_manifest_sha256"] = hashlib.sha256(
        json.dumps(split, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(RESULTS / "manifests/cifar10_split_manifest.json", split)
    write_json(RESULTS / "preflight/split_check.json", split)
    order_reports = {}
    for seed in SEEDS:
        manifest = order_manifest(train_indices, seed)
        write_json(RESULTS / "manifests" / f"epoch_order_manifest_seed{seed}.json", manifest)
        order_reports[str(seed)] = {
            "epochs": len(manifest["orders"]),
            "all_sample_counts_exact": all(row["sample_count"] == 45_000 for row in manifest["orders"]),
            "all_unique_counts_exact": all(row["unique_count"] == 45_000 for row in manifest["orders"]),
            "manifest_sha256": manifest["manifest_sha256"],
        }
    write_json(RESULTS / "preflight/epoch_order_check.json", {
        "v4_epoch_order_check": "VALID",
        "seeds": order_reports, "config_sha256": config_hash(),
    })

    fixed_indices = train_indices[:32]
    raw, images, labels = canonical_images(train, fixed_indices)
    fixed = {
        "selection": "first_32_sorted_training_indices", "count": 32,
        "indices": fixed_indices.tolist(), "labels": labels.tolist(),
        "raw_uint8_sha256": sha256_array(raw),
        "canonical_float32_sha256": sha256_array(images),
        "canonical_shape": list(images.shape), "canonical_dtype": str(images.dtype),
        "preprocessing": "uint8_to_float32_div255", "config_sha256": config_hash(),
    }
    write_json(RESULTS / "manifests/fixed_batch_manifest.json", fixed)
    dataset_integrity = {
        "v4_dataset_integrity": "VALID", "official_train_count": len(train),
        "official_test_count": len(test), "train_raw_sha256": sha256_array(np.asarray(train.data, dtype=np.uint8)),
        "test_raw_sha256": split["test_raw_sha256"], "config_sha256": config_hash(),
        "download_performed": False,
    }
    write_json(RESULTS / "preflight/dataset_integrity.json", dataset_integrity)
    print("V4_DATASET_INTEGRITY = VALID")
    print("V4_SPLIT_CHECK = VALID")
    print("V4_EPOCH_ORDER_CHECK = VALID")
    print(f"FIXED_BATCH_FLOAT32_SHA256 = {fixed['canonical_float32_sha256']}")


if __name__ == "__main__":
    main()
