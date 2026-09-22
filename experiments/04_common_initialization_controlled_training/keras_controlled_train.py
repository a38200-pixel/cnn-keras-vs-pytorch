"""Explicit 30-epoch TensorFlow loop; full training is opt-in."""

from __future__ import annotations

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
from common.controlled_optimizer import export_keras_adam
from common.controlled_data import canonical_batch, canonical_rows, iter_batches, load_orders
from common.controlled_initialization import (
    build_keras_model, export_keras_weights, load_initial_weights, load_keras_weights,
)
from common.controlled_training_utils import (
    ensure_seed_available, require_valid_preflight, write_history, write_result,
)
from common.seed_utils import seed_tensorflow

RUN_TRAINING = True


def evaluate(model, rows, seed: int):
    import tensorflow as tf
    from sklearn.metrics import f1_score

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_sum = correct = count = 0
    labels_all, predictions = [], []
    for indices in iter_batches(np.arange(len(rows))):
        images, labels = canonical_batch(rows, indices, seed, 0, False)
        logits = model(tf.convert_to_tensor(images), training=False)
        batch_loss = criterion(labels, logits)
        predicted = np.argmax(logits.numpy(), axis=1)
        loss_sum += float(batch_loss) * len(labels)
        correct += int(np.sum(predicted == labels))
        count += len(labels)
        labels_all.extend(labels.tolist())
        predictions.extend(predicted.tolist())
    return {
        "loss": loss_sum / count, "accuracy": correct / count,
        "macro_f1": float(f1_score(labels_all, predictions, average="macro")),
    }


def train_seed(seed: int) -> None:
    import tensorflow as tf

    require_valid_preflight()
    reuse_initial_checkpoint = ensure_seed_available("keras", seed)
    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx("float32")
    seed_tensorflow(seed)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("TensorFlow Metal GPU is required for controlled training.")
    train, val, test = (canonical_rows(split) for split in ("train", "val", "test"))
    order = load_orders(seed, len(train))
    canonical = load_initial_weights(seed)
    model = build_keras_model()
    load_keras_weights(model, canonical)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=BETA1, beta_2=BETA2,
        epsilon=EPSILON, weight_decay=0.0, amsgrad=False,
        clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False,
    )
    def checkpoint(name: str, epoch_number: int, global_step: int) -> None:
        if global_step:
            slots, optimizer_step = export_keras_adam(model, optimizer)
            if optimizer_step != global_step:
                raise RuntimeError(f"Keras checkpoint step mismatch: {optimizer_step}/{global_step}")
        else:
            slots = {}
        save_canonical_checkpoint(
            "keras", seed, name, export_keras_weights(model), slots,
            epoch=epoch_number, optimizer_step=global_step,
        )
    if SAVE_CANONICAL_CHECKPOINT and SAVE_INITIAL_CHECKPOINT and not reuse_initial_checkpoint:
        checkpoint("initial", 0, 0)
    history = []
    best_loss, best_epoch, best_weights = float("inf"), 0, None
    steps = 0
    started = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        epoch_started = time.perf_counter()
        loss_sum = correct = count = 0
        for indices in iter_batches(order[epoch]):
            images, labels = canonical_batch(train, indices, seed, epoch, True)
            x = tf.convert_to_tensor(images, dtype=tf.float32)
            y = tf.convert_to_tensor(labels, dtype=tf.int32)
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = criterion(y, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            if any(gradient is None for gradient in gradients) or not np.isfinite(float(loss)):
                raise RuntimeError(f"Non-finite or missing gradient at seed={seed} epoch={epoch + 1}")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            steps += 1
            if steps == 1 and SAVE_CANONICAL_CHECKPOINT and SAVE_FIRST_STEP_CHECKPOINT:
                checkpoint("after_first_step", 0, steps)
            predicted = np.argmax(logits.numpy(), axis=1)
            loss_sum += float(loss) * len(labels)
            correct += int(np.sum(predicted == labels))
            count += len(labels)
        validation = evaluate(model, val, seed)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch + 1
            best_weights = [value.copy() for value in model.get_weights()]
        history.append({
            "framework": "keras", "experiment": EXPERIMENT,
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
            f"Keras seed={seed} epoch={epoch + 1:02d}/{MAX_EPOCHS} "
            f"loss={history[-1]['train_loss']:.5f} acc={history[-1]['train_accuracy']:.4f} "
            f"val_loss={validation['loss']:.5f} val_acc={validation['accuracy']:.4f} "
            f"lr={LEARNING_RATE} steps={steps}", flush=True,
        )
    elapsed = time.perf_counter() - started
    if steps != MAX_EPOCHS * ((len(train) + 31) // 32):
        raise RuntimeError(f"Unexpected optimizer step count: {steps}")
    write_history("keras", seed, history)
    final = evaluate(model, test, seed)
    best_model = build_keras_model()
    best_model.set_weights(best_weights)
    best = evaluate(best_model, test, seed)
    write_result("keras", seed, {
        "epochs_trained": MAX_EPOCHS, "optimizer_steps": steps,
        "final_test_accuracy": final["accuracy"], "final_macro_f1": final["macro_f1"],
        "final_test_loss": final["loss"], "best_val_epoch": best_epoch,
        "best_val_loss": best_loss, "best_test_accuracy": best["accuracy"],
        "best_macro_f1": best["macro_f1"], "best_test_loss": best["loss"],
        "training_time_seconds": elapsed, "framework_version": tf.__version__,
        "device": "tensorflow-metal GPU:0",
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
