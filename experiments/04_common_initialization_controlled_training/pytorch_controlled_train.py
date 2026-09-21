"""Explicit 30-epoch PyTorch loop; full training is opt-in."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_config import (
    BETA1, BETA2, EPSILON, EXPERIMENT, LEARNING_RATE, MAX_EPOCHS,
    SEEDS, CHECKPOINT_EPOCHS, SAVE_CANONICAL_CHECKPOINT,
    SAVE_FIRST_STEP_CHECKPOINT, SAVE_INITIAL_CHECKPOINT, config_hash,
)
from common.controlled_checkpoint import save_canonical_checkpoint
from common.controlled_optimizer import export_torch_adam
from common.controlled_data import canonical_batch, canonical_rows, iter_batches, load_orders
from common.controlled_initialization import (
    build_torch_model, export_torch_weights, load_initial_weights, load_torch_weights,
)
from common.controlled_training_utils import (
    ensure_seed_available, require_valid_preflight, write_history, write_result,
)
from common.environment_utils import select_torch_device
from common.seed_utils import seed_pytorch

RUN_TRAINING = False


def evaluate(model, rows, seed: int, device):
    import torch
    from sklearn.metrics import f1_score

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    model.eval()
    loss_sum = correct = count = 0
    labels_all, predictions = [], []
    with torch.no_grad():
        for indices in iter_batches(np.arange(len(rows))):
            images, labels = canonical_batch(rows, indices, seed, 0, False)
            x = torch.from_numpy(images.transpose(0, 3, 1).copy()).to(device)
            y = torch.from_numpy(labels.astype(np.int64)).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            predicted = logits.argmax(1)
            loss_sum += float(loss.item()) * len(labels)
            correct += int((predicted == y).sum().item())
            count += len(labels)
            labels_all.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())
    return {
        "loss": loss_sum / count, "accuracy": correct / count,
        "macro_f1": float(f1_score(labels_all, predictions, average="macro")),
    }


def train_seed(seed: int) -> None:
    import torch

    require_valid_preflight()
    ensure_seed_available("pytorch", seed)
    seed_pytorch(seed)
    torch.set_default_dtype(torch.float32)
    device = select_torch_device()
    if device.type != "mps":
        raise RuntimeError("PyTorch MPS is required for controlled training.")
    train, val, test = (canonical_rows(split) for split in ("train", "val", "test"))
    order = load_orders(seed, len(train))
    canonical = load_initial_weights(seed)
    model = build_torch_model().to(device)
    load_torch_weights(model, canonical)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2),
        eps=EPSILON, weight_decay=0.0, amsgrad=False, foreach=False,
    )
    def checkpoint(name: str, epoch_number: int, global_step: int) -> None:
        if global_step:
            slots, optimizer_step = export_torch_adam(model, optimizer)
            if optimizer_step != global_step:
                raise RuntimeError(f"PyTorch checkpoint step mismatch: {optimizer_step}/{global_step}")
        else:
            slots = {}
        save_canonical_checkpoint(
            "pytorch", seed, name, export_torch_weights(model), slots,
            epoch=epoch_number, optimizer_step=global_step,
        )
    if SAVE_CANONICAL_CHECKPOINT and SAVE_INITIAL_CHECKPOINT:
        checkpoint("initial", 0, 0)
    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    steps = 0
    started = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = correct = count = 0
        for indices in iter_batches(order[epoch]):
            images, labels = canonical_batch(train, indices, seed, epoch, True)
            x = torch.from_numpy(images.transpose(0, 3, 1).copy()).to(device)
            y = torch.from_numpy(labels.astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at seed={seed} epoch={epoch + 1}")
            loss.backward()
            optimizer.step()
            steps += 1
            if steps == 1 and SAVE_CANONICAL_CHECKPOINT and SAVE_FIRST_STEP_CHECKPOINT:
                checkpoint("after_first_step", 0, steps)
            predicted = logits.argmax(1)
            loss_sum += float(loss.item()) * len(labels)
            correct += int((predicted == y).sum().item())
            count += len(labels)
        validation = evaluate(model, val, seed, device)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "framework": "pytorch", "experiment": EXPERIMENT,
            "phase": "strict_controlled_comparison", "seed": seed, "epoch": epoch + 1,
            "train_loss": loss_sum / count, "train_accuracy": correct / count,
            "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
            "learning_rate": LEARNING_RATE, "num_optimizer_steps": steps,
            "elapsed_epoch_sec": time.perf_counter() - epoch_started,
            "config_sha256": config_hash(),
        })
        if SAVE_CANONICAL_CHECKPOINT and epoch + 1 in CHECKPOINT_EPOCHS:
            checkpoint(f"epoch_{epoch + 1:03d}", epoch + 1, steps)
        print(
            f"PyTorch seed={seed} epoch={epoch + 1:02d}/{MAX_EPOCHS} "
            f"loss={history[-1]['train_loss']:.5f} acc={history[-1]['train_accuracy']:.4f} "
            f"val_loss={validation['loss']:.5f} val_acc={validation['accuracy']:.4f} "
            f"lr={LEARNING_RATE} steps={steps}", flush=True,
        )
    elapsed = time.perf_counter() - started
    if steps != MAX_EPOCHS * ((len(train) + 31) // 32):
        raise RuntimeError(f"Unexpected optimizer step count: {steps}")
    write_history("pytorch", seed, history)
    final = evaluate(model, test, seed, device)
    best_model = build_torch_model().to(device)
    best_model.load_state_dict(best_state)
    best = evaluate(best_model, test, seed, device)
    write_result("pytorch", seed, {
        "epochs_trained": MAX_EPOCHS, "optimizer_steps": steps,
        "final_test_accuracy": final["accuracy"], "final_macro_f1": final["macro_f1"],
        "final_test_loss": final["loss"], "best_val_epoch": best_epoch,
        "best_val_loss": best_loss, "best_test_accuracy": best["accuracy"],
        "best_macro_f1": best["macro_f1"], "best_test_loss": best["loss"],
        "training_time_seconds": elapsed, "framework_version": torch.__version__,
        "device": str(device),
    })


def main() -> None:
    print(f"{EXPERIMENT}: RUN_TRAINING={RUN_TRAINING}")
    if not RUN_TRAINING:
        print("Full training was not started. Set RUN_TRAINING=True only after VALID preflight.")
        return
    for seed in SEEDS:
        train_seed(seed)


if __name__ == "__main__":
    main()
