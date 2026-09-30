"""PyTorch native backward + CommonAdam + Common BN; full training is opt-in."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.common_adam_control import load_torch_trainable, torch_gradients, trainable_values
from common.controlled_batchnorm import build_torch_common_bn_model, torch_uses_native_batchnorm
from common.controlled_data import canonical_batch, canonical_rows, iter_batches, load_orders
from common.controlled_initialization import export_torch_weights, load_initial_weights, load_torch_weights
from common.environment_utils import select_torch_device
from common.reference_adam import CommonAdam
from common.seed_utils import seed_pytorch
from experiment_config import CHECKPOINT_EPOCHS, LEARNING_RATE, MAX_EPOCHS, SEEDS, config_hash
from experiment_utils import ensure_seed_available, evaluate_torch, require_preflight, save_checkpoint, write_history, write_result

RUN_TRAINING = False


def train_seed(seed: int) -> None:
    import torch

    require_preflight()
    ensure_seed_available("pytorch", seed)
    seed_pytorch(seed)
    torch.set_default_dtype(torch.float32)
    device = select_torch_device()
    if device.type != "mps":
        raise RuntimeError("PyTorch MPS required")
    train, val, test = (canonical_rows(split) for split in ("train", "val", "test"))
    orders = load_orders(seed, len(train))
    w0 = load_initial_weights(seed)
    model = build_torch_common_bn_model().to(device)
    if torch_uses_native_batchnorm(model):
        raise RuntimeError("Native PyTorch BatchNorm2d found in Experiment 06")
    load_torch_weights(model, w0)
    adam = CommonAdam(trainable_values(w0))
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    save_checkpoint("pytorch", seed, "initial", export_torch_weights(model), adam, 0, 0)
    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    steps = 0
    started = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = correct = count = 0
        for indices in iter_batches(orders[epoch]):
            images, labels = canonical_batch(train, indices, seed, epoch, True)
            x = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device)
            y = torch.from_numpy(labels.astype(np.int64)).to(device)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            gradients = torch_gradients(model)
            current = trainable_values(export_torch_weights(model))
            updated, _ = adam.update(current, gradients)
            load_torch_trainable(model, updated)
            steps += 1
            if steps == 1:
                save_checkpoint("pytorch", seed, "after_first_step", export_torch_weights(model), adam, 0, steps)
            predicted = logits.argmax(1)
            loss_sum += float(loss.item()) * len(labels)
            correct += int((predicted == y).sum().item())
            count += len(labels)
        validation = evaluate_torch(model, val, seed, device)
        if validation["loss"] < best_loss:
            best_loss, best_epoch, best_state = validation["loss"], epoch + 1, export_torch_weights(model)
        history.append({
            "framework": "pytorch", "experiment": "06_batchnorm_state_divergence",
            "phase": "strict_controlled_common_adam_common_bn", "seed": seed, "epoch": epoch + 1,
            "train_loss": loss_sum / count, "train_accuracy": correct / count,
            "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
            "learning_rate": LEARNING_RATE, "num_optimizer_steps": steps,
            "elapsed_epoch_sec": time.perf_counter() - epoch_started, "config_sha256": config_hash(),
        })
        if epoch + 1 in CHECKPOINT_EPOCHS:
            save_checkpoint("pytorch", seed, f"epoch_{epoch + 1:03d}", export_torch_weights(model), adam, epoch + 1, steps)
        print(
            f"PyTorch CommonBN seed={seed} epoch={epoch+1:02d}/{MAX_EPOCHS} "
            f"loss={loss_sum/count:.5f} acc={correct/count:.4f} "
            f"val_loss={validation['loss']:.5f} val_acc={validation['accuracy']:.4f} steps={steps}",
            flush=True,
        )
    if steps != 9630:
        raise RuntimeError(f"Unexpected optimizer steps: {steps}")
    write_history("pytorch", seed, history)
    final = evaluate_torch(model, test, seed, device)
    best_model = build_torch_common_bn_model().to(device)
    load_torch_weights(best_model, best_state)
    best = evaluate_torch(best_model, test, seed, device)
    write_result("pytorch", seed, {
        "epochs_trained": MAX_EPOCHS, "optimizer_steps": steps,
        "final_test_accuracy": final["accuracy"], "final_macro_f1": final["macro_f1"], "final_test_loss": final["loss"],
        "best_val_epoch": best_epoch, "best_val_loss": best_loss, "best_test_accuracy": best["accuracy"],
        "best_macro_f1": best["macro_f1"], "best_test_loss": best["loss"],
        "training_time_seconds": time.perf_counter() - started,
        "framework_version": torch.__version__, "device": str(device),
    })


def main() -> None:
    print(f"06_batchnorm_state_divergence: RUN_TRAINING={RUN_TRAINING}")
    if not RUN_TRAINING:
        print("Full training was not started. Set RUN_TRAINING=True only after VALID preflight.")
        return
    for seed in SEEDS:
        train_seed(seed)


if __name__ == "__main__":
    main()
