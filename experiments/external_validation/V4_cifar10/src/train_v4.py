"""User-invoked V4 Stage B controlled 30-epoch training for one framework/seed."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from common.reference_adam import CommonAdam
from common.trace_utils import state_hash
from cifar10_canonical_state import (
    TRAINABLE_NAMES, create_initial_state, export_keras_state, export_torch_state,
    keras_gradients, load_keras_state, load_torch_state, torch_gradients, trainable_values,
)
from cifar10_keras import build_keras_cifar10
from cifar10_pytorch import build_torch_cifar10
from v4_config import (
    BATCH_SIZE, EPOCHS, EPOCH_SNAPSHOT_EPOCHS, RESULTS, SEEDS,
    STEP_SNAPSHOT_STEPS, STEPS_PER_EPOCH, TOTAL_STEPS, config_hash,
)
from v4_data import batches, canonical_images, epoch_order, require_local_cifar10, stratified_split


def _optimizer_arrays(adam: CommonAdam) -> dict[str, np.ndarray]:
    values = {f"m::{name}": value.copy() for name, value in adam.m.items()}
    values.update({f"v::{name}": value.copy() for name, value in adam.v.items()})
    values["step"] = np.asarray([adam.step], dtype=np.int64)
    return values


def _restore_adam(state, optimizer) -> CommonAdam:
    adam = CommonAdam(trainable_values(state))
    adam.step = int(optimizer["step"][0])
    for name in TRAINABLE_NAMES:
        adam.m[name] = optimizer[f"m::{name}"].astype(np.float32, copy=True)
        adam.v[name] = optimizer[f"v::{name}"].astype(np.float32, copy=True)
    return adam


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _save_npz_exact(path: Path, values: dict[str, np.ndarray]) -> None:
    if path.exists():
        old = _read_npz(path)
        if set(old) == set(values) and all(np.array_equal(old[name], values[name]) for name in values):
            return
        raise FileExistsError(f"Snapshot differs; refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _save_snapshot(framework, seed, name, checkpoint_type, epoch, global_step, state, adam, split_hash, order_hash):
    folder = RESULTS / "snapshots" / framework / f"seed{seed}" / name
    model_path, optimizer_path, metadata_path = folder / "model.npz", folder / "optimizer.npz", folder / "metadata.json"
    _save_npz_exact(model_path, state); _save_npz_exact(optimizer_path, _optimizer_arrays(adam))
    metadata = {
        "framework": framework, "seed": seed, "checkpoint_name": name,
        "checkpoint_type": checkpoint_type, "epoch": epoch, "global_step": global_step,
        "common_adam_step": adam.step, "capture_timing": "immediately_after_update" if global_step else "initial",
        "model_state_sha256": state_hash(state), "split_hash": split_hash,
        "epoch_order_manifest_hash": order_hash, "config_sha256": config_hash(),
    }
    encoded = json.dumps(metadata, indent=2) + "\n"
    if metadata_path.exists() and metadata_path.read_text() != encoded:
        raise FileExistsError(f"Snapshot metadata differs: {metadata_path}")
    metadata_path.parent.mkdir(parents=True, exist_ok=True); metadata_path.write_text(encoded)
    manifest_path = RESULTS / "snapshots" / "checkpoint_manifest.csv"
    fields = [
        "framework", "seed", "checkpoint_name", "checkpoint_type", "epoch",
        "global_step", "common_adam_step", "capture_timing", "model_path",
        "optimizer_path", "model_state_sha256", "split_hash",
        "epoch_order_manifest_hash", "config_sha256",
    ]
    row = dict(metadata)
    row["model_path"] = str(model_path.relative_to(RESULTS))
    row["optimizer_path"] = str(optimizer_path.relative_to(RESULTS))
    existing = []
    if manifest_path.exists():
        with manifest_path.open() as handle:
            existing = list(csv.DictReader(handle))
    identity = (framework, str(seed), name)
    matches = [item for item in existing if (item["framework"], item["seed"], item["checkpoint_name"]) == identity]
    comparable = {field: str(row[field]) for field in fields}
    if matches and matches[0] != comparable:
        raise FileExistsError(f"Snapshot manifest differs: {identity}")
    if not matches:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not existing:
                writer.writeheader()
            writer.writerow(row)


def _save_resume(
    framework, seed, completed_epoch, current_epoch, next_batch_position,
    partial_loss_sum, partial_correct, partial_count,
    global_step, state, adam, split_hash, order_hash,
):
    folder = RESULTS / "resume" / framework / f"seed{seed}"; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "latest_model.npz", **state)
    np.savez_compressed(folder / "latest_optimizer.npz", **_optimizer_arrays(adam))
    (folder / "latest_metadata.json").write_text(json.dumps({
        "completed_epoch": completed_epoch, "current_epoch": current_epoch,
        "next_batch_position": next_batch_position,
        "partial_loss_sum": partial_loss_sum, "partial_correct": partial_correct,
        "partial_count": partial_count, "global_step": global_step,
        "common_adam_step": adam.step, "split_hash": split_hash,
        "epoch_order_manifest_hash": order_hash, "config_sha256": config_hash(),
    }, indent=2) + "\n")


def _macro_f1(labels, predictions, classes=10):
    scores = []
    for label in range(classes):
        tp = np.sum((labels == label) & (predictions == label)); fp = np.sum((labels != label) & (predictions == label)); fn = np.sum((labels == label) & (predictions != label))
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=("keras", "pytorch"))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if (args.epochs, args.batch_size, args.lr) != (EPOCHS, BATCH_SIZE, 0.001):
        parser.error("Research mode requires epochs=30, batch_size=32, lr=0.001")
    train, test = require_local_cifar10(args.data_root); train_indices, val_indices = stratified_split(train.targets)
    split = json.loads((RESULTS / "manifests/cifar10_split_manifest.json").read_text())
    order_manifest = json.loads((RESULTS / "manifests" / f"epoch_order_manifest_seed{args.seed}.json").read_text())
    split_hash, order_hash = split["split_manifest_sha256"], order_manifest["manifest_sha256"]
    if args.framework == "keras":
        import tensorflow as tf
        torch = None
        if not tf.config.list_physical_devices("GPU"): raise RuntimeError("TensorFlow GPU unavailable")
        with tf.device("/GPU:0"): model = build_keras_cifar10()
    else:
        import torch
        tf = None
        if not torch.backends.mps.is_available(): raise RuntimeError("PyTorch MPS unavailable")
        device = torch.device("mps"); model = build_torch_cifar10().to(device)

    state, adam, start_epoch, global_step = create_initial_state(args.seed), None, 1, 0
    resume_batch_position, resume_loss, resume_correct, resume_count = 0, 0.0, 0, 0
    history_path = RESULTS / "training" / args.framework / f"seed{args.seed}" / "history.csv"
    history = []
    best_val, best_state, best_epoch = float("inf"), None, None
    if args.resume:
        folder = RESULTS / "resume" / args.framework / f"seed{args.seed}"
        metadata = json.loads((folder / "latest_metadata.json").read_text())
        if metadata["config_sha256"] != config_hash() or metadata["split_hash"] != split_hash or metadata["epoch_order_manifest_hash"] != order_hash:
            raise RuntimeError("Resume identity mismatch")
        state = _read_npz(folder / "latest_model.npz"); optimizer = _read_npz(folder / "latest_optimizer.npz")
        adam = _restore_adam(state, optimizer); global_step = metadata["global_step"]
        start_epoch = metadata["current_epoch"]
        resume_batch_position = metadata["next_batch_position"]
        resume_loss = float(metadata["partial_loss_sum"])
        resume_correct = int(metadata["partial_correct"])
        resume_count = int(metadata["partial_count"])
        if history_path.exists():
            with history_path.open() as handle: history = list(csv.DictReader(handle))
        expected_resume_step = int(metadata["completed_epoch"]) * STEPS_PER_EPOCH + int(resume_batch_position)
        if global_step != expected_resume_step or len(history) != int(metadata["completed_epoch"]):
            raise RuntimeError("Resume epoch/batch/history/global-step mismatch")
        if history:
            best_row = min(history, key=lambda row: float(row["validation_loss"]))
            best_val, best_epoch = float(best_row["validation_loss"]), int(best_row["epoch"])
            best_state = _read_npz(RESULTS / "runtime" / args.framework / f"seed{args.seed}" / "best_val_model.npz")
    else:
        adam = CommonAdam(trainable_values(state))
        _save_snapshot(args.framework, args.seed, "initial_step00000", "initial", 0, 0, state, adam, split_hash, order_hash)
    if adam.step != global_step: raise RuntimeError("Resume CommonAdam/global-step mismatch")

    def load_state(values):
        (load_keras_state if args.framework == "keras" else load_torch_state)(model, values)
    def export_state():
        return (export_keras_state if args.framework == "keras" else export_torch_state)(model)
    load_state(state)

    def step_batch(indices):
        nonlocal state, global_step
        _, images, labels = canonical_images(train, indices)
        if args.framework == "keras":
            with tf.device("/GPU:0"):
                x, y = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, tf.int32)
                with tf.GradientTape() as tape:
                    logits = model(x, training=True); loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
                gradients = keras_gradients(model, tape.gradient(loss, model.trainable_variables)); predictions = tf.argmax(logits, axis=1).numpy()
        else:
            x = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device); y = torch.from_numpy(labels).to(device)
            model.train(); model.zero_grad(set_to_none=True); logits = model(x); loss = torch.nn.functional.cross_entropy(logits, y); loss.backward()
            gradients = torch_gradients(model); predictions = logits.argmax(1).detach().cpu().numpy()
        after_forward = export_state(); updated, _ = adam.update(trainable_values(after_forward), gradients); state = dict(after_forward); state.update(updated); load_state(state)
        global_step += 1
        if adam.step != global_step: raise RuntimeError("Off-by-one: CommonAdam step != global step after update")
        loss_value = float(loss.numpy() if args.framework == "keras" else loss.detach().cpu())
        if not np.isfinite(loss_value): raise RuntimeError("NaN/Inf in V4 training loss")
        return loss_value, predictions, labels

    def evaluate(dataset, indices):
        losses, correct, count, ys, ps = 0.0, 0, 0, [], []
        for batch_indices in batches(indices):
            _, images, labels = canonical_images(dataset, batch_indices)
            if args.framework == "keras":
                logits = model(tf.convert_to_tensor(images), training=False); loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)); pred = tf.argmax(logits, 1).numpy(); value = float(loss.numpy())
            else:
                model.eval()
                with torch.no_grad():
                    logits = model(torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device)); y = torch.from_numpy(labels).to(device); loss = torch.nn.functional.cross_entropy(logits, y); pred = logits.argmax(1).cpu().numpy(); value = float(loss.cpu())
            losses += value * len(labels); correct += int(np.sum(pred == labels)); count += len(labels); ys.append(labels); ps.append(pred)
        metrics = (losses / count, correct / count, _macro_f1(np.concatenate(ys), np.concatenate(ps)))
        if not np.isfinite(metrics).all(): raise RuntimeError("NaN/Inf in V4 evaluation metrics")
        return metrics

    for epoch in range(start_epoch, EPOCHS + 1):
        order = epoch_order(train_indices, args.seed, epoch)
        if epoch == start_epoch:
            total_loss, correct, count = resume_loss, resume_correct, resume_count
            first_batch_position = resume_batch_position
        else:
            total_loss, correct, count, first_batch_position = 0.0, 0, 0, 0
        for batch_position, batch_indices in enumerate(batches(order)):
            if batch_position < first_batch_position:
                continue
            loss, pred, labels = step_batch(batch_indices); total_loss += loss * len(labels); correct += int(np.sum(pred == labels)); count += len(labels)
            if global_step in STEP_SNAPSHOT_STEPS:
                _save_snapshot(args.framework, args.seed, f"step{global_step:05d}", "step", epoch, global_step, state, adam, split_hash, order_hash)
                _save_resume(
                    args.framework, args.seed, epoch - 1, epoch, batch_position + 1,
                    total_loss, correct, count, global_step, state, adam, split_hash, order_hash,
                )
        expected_step = epoch * STEPS_PER_EPOCH
        if global_step != expected_step: raise RuntimeError(f"Epoch/global-step mismatch: {epoch}, {global_step}")
        val_loss, val_accuracy, _ = evaluate(train, val_indices)
        row = {"epoch": epoch, "global_step_end": global_step, "train_loss": total_loss / count, "train_accuracy": correct / count,
               "validation_loss": val_loss, "validation_accuracy": val_accuracy, "learning_rate": 0.001,
               "epoch_order_hash": order_manifest["orders"][epoch - 1]["permutation_sha256"], "nan_inf": False}
        history.append(row)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        if epoch in EPOCH_SNAPSHOT_EPOCHS:
            _save_snapshot(args.framework, args.seed, f"epoch{epoch:03d}", "epoch", epoch, global_step, state, adam, split_hash, order_hash)
        if val_loss < best_val:
            best_val, best_state, best_epoch = val_loss, {name: value.copy() for name, value in state.items()}, epoch
            runtime_best = RESULTS / "runtime" / args.framework / f"seed{args.seed}" / "best_val_model.npz"
            runtime_best.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(runtime_best, **best_state)
        _save_resume(
            args.framework, args.seed, epoch, epoch + 1, 0, 0.0, 0, 0,
            global_step, state, adam, split_hash, order_hash,
        )
        print(f"V4 {args.framework} seed={args.seed} epoch={epoch:02d}/30 step={global_step} train_acc={correct/count:.4f} val_acc={val_accuracy:.4f}", flush=True)
    if global_step != TOTAL_STEPS: raise RuntimeError("V4 final total-step mismatch")
    final_state = {name: value.copy() for name, value in state.items()}
    _save_npz_exact(RESULTS / "training" / args.framework / f"seed{args.seed}" / "final_model.npz", final_state)
    _save_npz_exact(RESULTS / "training" / args.framework / f"seed{args.seed}" / "best_val_model.npz", best_state)
    test_indices = np.arange(len(test), dtype=np.int64); metrics = {}
    for label, selected in (("final", final_state), ("best_val", best_state)):
        load_state(selected); loss, accuracy, macro_f1 = evaluate(test, test_indices); metrics[label] = {"epoch": 30 if label == "final" else best_epoch, "loss": loss, "accuracy": accuracy, "macro_f1": macro_f1}
    metrics.update({"framework": args.framework, "seed": args.seed, "completed_epochs": EPOCHS, "global_step": global_step, "config_sha256": config_hash()})
    (history_path.parent / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("V4_TRAINING_RUN_COMPLETE = TRUE")


if __name__ == "__main__": main()
