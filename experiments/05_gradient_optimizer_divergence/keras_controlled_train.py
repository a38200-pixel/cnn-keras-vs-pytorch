"""Keras native backward + shared CommonAdam; full training is opt-in."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.common_adam_control import keras_gradients, load_keras_trainable, trainable_values
from common.controlled_data import canonical_batch, canonical_rows, iter_batches, load_orders
from common.controlled_initialization import build_keras_model, export_keras_weights, load_initial_weights, load_keras_weights
from common.reference_adam import CommonAdam
from common.seed_utils import seed_tensorflow
from experiment_config import CHECKPOINT_EPOCHS, LEARNING_RATE, MAX_EPOCHS, SEEDS, config_hash
from experiment_utils import ensure_seed_available, evaluate_keras, require_preflight, save_checkpoint, write_history, write_result

RUN_TRAINING = True


def train_seed(seed: int) -> None:
    import tensorflow as tf

    require_preflight(); ensure_seed_available("keras", seed)
    tf.keras.backend.clear_session(); tf.keras.backend.set_floatx("float32"); seed_tensorflow(seed)
    if not tf.config.list_physical_devices("GPU"): raise RuntimeError("TensorFlow Metal GPU required")
    train, val, test = (canonical_rows(split) for split in ("train", "val", "test"))
    orders = load_orders(seed, len(train)); w0 = load_initial_weights(seed)
    model = build_keras_model(); load_keras_weights(model, w0)
    adam = CommonAdam(trainable_values(w0)); criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    save_checkpoint("keras", seed, "initial", export_keras_weights(model), adam, 0, 0)
    history = []; best_loss = float("inf"); best_epoch = 0; best_state = None; steps = 0
    started = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        epoch_started = time.perf_counter(); loss_sum = correct = count = 0
        for indices in iter_batches(orders[epoch]):
            images, labels = canonical_batch(train, indices, seed, epoch, True)
            x = tf.convert_to_tensor(images, dtype=tf.float32); y = tf.convert_to_tensor(labels, dtype=tf.int32)
            with tf.GradientTape() as tape:
                logits = model(x, training=True); loss = criterion(y, logits)
            gradients = keras_gradients(model, tape.gradient(loss, model.trainable_variables))
            current = trainable_values(export_keras_weights(model)); updated, _ = adam.update(current, gradients)
            load_keras_trainable(model, updated); steps += 1
            if steps == 1: save_checkpoint("keras", seed, "after_first_step", export_keras_weights(model), adam, 0, steps)
            predicted = logits.numpy().argmax(1); loss_sum += float(loss) * len(labels)
            correct += int((predicted == labels).sum()); count += len(labels)
        validation = evaluate_keras(model, val, seed)
        if validation["loss"] < best_loss:
            best_loss, best_epoch, best_state = validation["loss"], epoch + 1, export_keras_weights(model)
        history.append({
            "framework": "keras", "experiment": "05_gradient_optimizer_divergence",
            "phase": "strict_controlled_common_adam", "seed": seed, "epoch": epoch + 1,
            "train_loss": loss_sum / count, "train_accuracy": correct / count,
            "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
            "learning_rate": LEARNING_RATE, "num_optimizer_steps": steps,
            "elapsed_epoch_sec": time.perf_counter() - epoch_started, "config_sha256": config_hash(),
        })
        if epoch + 1 in CHECKPOINT_EPOCHS:
            save_checkpoint("keras", seed, f"epoch_{epoch + 1:03d}", export_keras_weights(model), adam, epoch + 1, steps)
        print(f"Keras CommonAdam seed={seed} epoch={epoch+1:02d}/{MAX_EPOCHS} loss={loss_sum/count:.5f} acc={correct/count:.4f} val_loss={validation['loss']:.5f} val_acc={validation['accuracy']:.4f} steps={steps}", flush=True)
    if steps != 9630: raise RuntimeError(f"Unexpected optimizer steps: {steps}")
    write_history("keras", seed, history); final = evaluate_keras(model, test, seed)
    best_model = build_keras_model(); load_keras_weights(best_model, best_state); best = evaluate_keras(best_model, test, seed)
    write_result("keras", seed, {
        "epochs_trained": MAX_EPOCHS, "optimizer_steps": steps,
        "final_test_accuracy": final["accuracy"], "final_macro_f1": final["macro_f1"], "final_test_loss": final["loss"],
        "best_val_epoch": best_epoch, "best_val_loss": best_loss, "best_test_accuracy": best["accuracy"],
        "best_macro_f1": best["macro_f1"], "best_test_loss": best["loss"],
        "training_time_seconds": time.perf_counter() - started, "framework_version": tf.__version__, "device": "tensorflow-metal GPU:0",
    })


def main() -> None:
    print(f"05_gradient_optimizer_divergence: RUN_TRAINING={RUN_TRAINING}")
    if not RUN_TRAINING:
        print("Full training was not started. Set RUN_TRAINING=True only after VALID preflight."); return
    for seed in SEEDS: train_seed(seed)


if __name__ == "__main__": main()
