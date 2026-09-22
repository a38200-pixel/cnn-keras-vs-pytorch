"""Artifact, evaluation, and safety helpers for Experiment 05."""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

from common.controlled_checkpoint import save_canonical_checkpoint
from common.controlled_data import canonical_batch, iter_batches
from common.trace_utils import assert_finite
from experiment_config import EXPERIMENT, RESULTS, config_hash

HISTORY_FIELDS = [
    "framework", "experiment", "phase", "seed", "epoch", "train_loss",
    "train_accuracy", "val_loss", "val_accuracy", "learning_rate",
    "num_optimizer_steps", "elapsed_epoch_sec", "config_sha256",
]
RESULT_FIELDS = [
    "experiment", "phase", "framework", "seed", "epochs_trained", "optimizer_steps",
    "final_test_accuracy", "final_macro_f1", "final_test_loss", "best_val_epoch",
    "best_val_loss", "best_test_accuracy", "best_macro_f1", "best_test_loss",
    "training_time_seconds", "python_version", "framework_version", "device",
    "architecture", "config_sha256",
]


def save_checkpoint(framework, seed, name, model_state, adam, epoch, step, diagnostic=False):
    return save_canonical_checkpoint(
        framework, seed, name, model_state, adam.export_state() if step else {},
        epoch=epoch, optimizer_step=step, diagnostic=diagnostic,
        results_root=RESULTS, config_sha256=config_hash(),
    )


def require_preflight() -> dict[str, object]:
    path = RESULTS / "preflight" / "preflight_summary.json"
    if not path.exists():
        raise RuntimeError("Run Experiment 05 preflight_validate.py first")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("common_adam_precheck") != "VALID" or not report.get("full_training_ready"):
        raise RuntimeError(f"Common Adam preflight not VALID: {report.get('errors')}")
    if report.get("config_sha256") != config_hash():
        raise RuntimeError("Experiment 05 config hash changed since preflight")
    return report


def ensure_seed_available(framework: str, seed: int) -> None:
    paths = [
        RESULTS / "history" / f"{framework}_seed{seed}_history.csv",
        RESULTS / "checkpoints" / f"seed{seed}" / framework,
    ]
    existing = [path for path in paths if path.exists()]
    result_path = RESULTS / f"{framework}_common_adam_results.csv"
    if result_path.exists():
        with result_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if any(int(row["seed"]) == seed for row in rows):
            existing.append(result_path)
    if existing:
        raise FileExistsError(f"Refusing overwrite/automatic resume: {existing}")


def write_history(framework: str, seed: int, rows: list[dict[str, object]]) -> None:
    path = RESULTS / "history" / f"{framework}_seed{seed}_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader(); writer.writerows(rows)


def write_result(framework: str, seed: int, row: dict[str, object]) -> None:
    path = RESULTS / f"{framework}_common_adam_results.csv"
    existing = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if any(int(item["seed"]) == seed for item in existing):
            raise FileExistsError(f"Result already exists: {framework} seed={seed}")
    value = dict(row)
    value.update({
        "experiment": EXPERIMENT, "phase": "strict_controlled_common_adam",
        "framework": framework, "seed": seed, "python_version": sys.version.split()[0],
        "architecture": platform.machine(), "config_sha256": config_hash(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader(); writer.writerows(existing); writer.writerow(value)


def evaluate_keras(model, rows, seed: int) -> dict[str, float]:
    import tensorflow as tf
    from sklearn.metrics import f1_score
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_sum = correct = count = 0; truth = []; predicted_all = []
    for indices in iter_batches(np.arange(len(rows))):
        images, labels = canonical_batch(rows, indices, seed, 0, False)
        logits = model(tf.convert_to_tensor(images, dtype=tf.float32), training=False)
        loss = criterion(labels, logits); predicted = logits.numpy().argmax(1)
        loss_sum += float(loss) * len(labels); correct += int((predicted == labels).sum()); count += len(labels)
        truth.extend(labels.tolist()); predicted_all.extend(predicted.tolist())
    return {"loss": loss_sum / count, "accuracy": correct / count,
            "macro_f1": float(f1_score(truth, predicted_all, average="macro"))}


def evaluate_torch(model, rows, seed: int, device) -> dict[str, float]:
    import torch
    from sklearn.metrics import f1_score
    model.eval(); loss_sum = correct = count = 0; truth = []; predicted_all = []
    with torch.no_grad():
        for indices in iter_batches(np.arange(len(rows))):
            images, labels = canonical_batch(rows, indices, seed, 0, False)
            x = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device)
            y = torch.from_numpy(labels.astype(np.int64)).to(device)
            logits = model(x); loss = torch.nn.functional.cross_entropy(logits, y)
            predicted = logits.argmax(1)
            loss_sum += float(loss.item()) * len(labels); correct += int((predicted == y).sum()); count += len(labels)
            truth.extend(labels.tolist()); predicted_all.extend(predicted.cpu().tolist())
    return {"loss": loss_sum / count, "accuracy": correct / count,
            "macro_f1": float(f1_score(truth, predicted_all, average="macro"))}
