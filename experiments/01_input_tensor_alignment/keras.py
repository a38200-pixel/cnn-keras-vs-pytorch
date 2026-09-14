"""Experiment 01: Keras baseline with deterministic input preprocessing aligned."""

# ============================================================
# 1. Imports
# ============================================================
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from common.dataset_utils import (
    CLASSES,
    build_split_manifest,
    find_dataset_root,
    is_physically_split,
    split_counts,
)
from common.environment_utils import print_tensorflow_environment, tensorflow_device_name
from common.history_utils import save_history
from common.input_alignment import compare_framework_inputs, load_aligned_rgb
from common.result_utils import save_result
from common.seed_utils import seed_tensorflow


# ============================================================
# 2. Experiment Configuration
# ============================================================
EXPERIMENT = "01_input_tensor_alignment"
ALIGNMENT = "input_tensor"
SEEDS = [42, 123, 2026]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS = 128, 32, 8, 30
LEARNING_RATE, EARLY_PATIENCE, LR_PATIENCE, MIN_DELTA = 0.001, 7, 4, 1e-4
RUN_TRAINING = False
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS = RESULTS_DIR / "keras_results.csv"
HISTORY_DIR = RESULTS_DIR / "history"


# ============================================================
# 3. Random Seed
# ============================================================
def set_seed(seed: int) -> None:
    seed_tensorflow(seed)


# ============================================================
# 4. Dataset / Input Preprocessing
# ============================================================
def load_deterministic_input(path: str) -> np.ndarray:
    """Pillow decode -> RGB -> bilinear 128x128 -> float32 [0, 1]."""
    return load_aligned_rgb(path, IMG_SIZE)


def make_sequence(rows, seed: int, training: bool):
    import tensorflow as tf

    class ImageSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.epoch = 0
            self.indices = list(range(len(rows)))
            self.on_epoch_end(initial=True)

        def __len__(self):
            return (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

        def __getitem__(self, batch_index):
            indices = self.indices[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
            images = [load_deterministic_input(str(rows[index]["path"])) for index in indices]
            labels = [int(rows[index]["label"]) for index in indices]
            return np.stack(images), np.asarray(labels, dtype=np.int32)

        def on_epoch_end(self, initial=False):
            if not initial:
                self.epoch += 1
            if training:
                # Preserve the framework-specific Baseline batch order.
                self.indices = np.random.default_rng(seed + self.epoch).permutation(len(rows)).tolist()

    return ImageSequence()


# ============================================================
# 5. Framework-Specific Baseline Augmentation
# ============================================================
def apply_baseline_augmentation(inputs, seed: int):
    """Keep Keras augmentation unchanged; Experiment 03 aligns augmentation."""
    from tensorflow.keras import layers

    x = layers.RandomFlip("horizontal", seed=seed, name="augmentation_flip")(inputs)
    return layers.RandomRotation(0.014, seed=seed + 1, name="augmentation_rotation")(x)


# ============================================================
# 6. CNN Model
# ============================================================
def build_model(seed: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="input")
    # The common loader already returns [0, 1]. Omitting Rescaling prevents
    # accidental double normalization while keeping the numerical input equal.
    x = apply_baseline_augmentation(inputs, seed)
    for index, filters in enumerate((32, 64, 128, 256), 1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{index}")(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=.99, name=f"bn{index}")(x)
        x = layers.ReLU(name=f"relu{index}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{index}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, name="fc128")(x)
    x = layers.ReLU(name="fc_relu")(x)
    logits = layers.Dense(NUM_CLASSES, name="logits")(x)
    outputs = layers.Softmax(name="softmax")(logits)
    return tf.keras.Model(inputs, outputs, name="final_rgb_cnn")


# ============================================================
# 7. Optimizer / Loss / Callbacks
# ============================================================
def make_optimizer_and_loss():
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=.9,
        beta_2=.999,
        epsilon=1e-7,
        amsgrad=False,
    )
    return optimizer, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def make_callbacks(seed: int, history_rows: list[dict[str, object]]):
    import tensorflow as tf

    class EpochHistory(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            logs["learning_rate"] = learning_rate
            row = {
                "framework": "keras",
                "experiment": EXPERIMENT,
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": float(logs["loss"]),
                "train_accuracy": float(logs["accuracy"]),
                "val_loss": float(logs["val_loss"]),
                "val_accuracy": float(logs["val_accuracy"]),
                "learning_rate": learning_rate,
                "elapsed_epoch_sec": time.perf_counter() - self.epoch_start,
            }
            history_rows.append(row)
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} summary | "
                f"train_loss={row['train_loss']:.6f} train_acc={row['train_accuracy']:.4f} | "
                f"val_loss={row['val_loss']:.6f} val_acc={row['val_accuracy']:.4f} | "
                f"lr={learning_rate:.8f}",
                flush=True,
            )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=EARLY_PATIENCE,
        min_delta=MIN_DELTA, restore_best_weights=True, verbose=1,
    )
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", mode="min", factor=.5, patience=LR_PATIENCE,
        min_delta=MIN_DELTA, min_lr=1e-6, verbose=1,
    )
    return early_stopping, scheduler, EpochHistory()


# ============================================================
# 8. Training / History Saving / Evaluation
# ============================================================
def print_training_header(seed: int) -> None:
    print("=" * 60)
    print("Experiment 01 - Input Tensor Alignment")
    print("Framework : Keras")
    print(f"Seed      : {seed}")
    print(f"Device    : {tensorflow_device_name()}")
    print(f"Epochs    : max {EPOCHS}")
    print(f"Batch     : {BATCH_SIZE}")
    print("=" * 60)


def result_contains_seed(seed: int) -> bool:
    if not RESULTS.exists():
        return False
    with RESULTS.open(encoding="utf-8") as handle:
        return any(int(row["seed"]) == seed for row in csv.DictReader(handle))


def seed_is_complete(seed: int) -> bool:
    return (
        result_contains_seed(seed)
        and (RESULTS_DIR / f"keras_seed{seed}.keras").exists()
        and (HISTORY_DIR / f"keras_seed{seed}_history.csv").exists()
    )


def train_one_seed(seed: int) -> None:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score

    tf.keras.backend.clear_session()
    set_seed(seed)
    print_training_header(seed)
    print_tensorflow_environment(EXPERIMENT, seed)
    manifest = build_split_manifest()
    train, val, test = (
        make_sequence(manifest[split], seed, split == "train")
        for split in ("train", "val", "test")
    )
    model = build_model(seed)
    optimizer, loss = make_optimizer_and_loss()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    history_rows: list[dict[str, object]] = []
    early_stopping, scheduler, recorder = make_callbacks(seed, history_rows)
    start = time.perf_counter()
    history = model.fit(
        train, validation_data=val, epochs=EPOCHS,
        callbacks=[early_stopping, scheduler, recorder], verbose=1,
    )
    elapsed = time.perf_counter() - start

    history_path = HISTORY_DIR / f"keras_seed{seed}_history.csv"
    save_history(history_path, history_rows)
    print(f"Saved history immediately: {history_path}")

    model_path = RESULTS_DIR / f"keras_seed{seed}.keras"
    model.save(model_path)
    test_loss, _ = model.evaluate(test, verbose=0)
    y_true = np.concatenate([test[index][1] for index in range(len(test))])
    predictions = np.argmax(model.predict(test, verbose=0), axis=1)
    best = int(getattr(early_stopping, "best_epoch", np.argmin(history.history["val_loss"])))
    save_result(RESULTS, {
        "experiment": EXPERIMENT, "framework": "keras", "seed": seed,
        "test_accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro"),
        "test_loss": test_loss, "best_epoch": best + 1,
        "epochs_trained": len(history_rows),
        "train_accuracy": history_rows[best]["train_accuracy"],
        "validation_accuracy": history_rows[best]["val_accuracy"],
        "train_loss": history_rows[best]["train_loss"],
        "validation_loss": history_rows[best]["val_loss"],
        "train_val_gap": history_rows[best]["train_accuracy"] - history_rows[best]["val_accuracy"],
        "training_time_seconds": elapsed,
    })
    print(f"Completed and saved Seed {seed}: {model_path}")


def run_3seed_experiment() -> None:
    for seed in SEEDS:
        if seed_is_complete(seed):
            print(f"[SKIP] Seed {seed} already has result, model, and history files.")
            continue
        if any((
            (RESULTS_DIR / f"keras_seed{seed}.keras").exists(),
            (HISTORY_DIR / f"keras_seed{seed}_history.csv").exists(),
            result_contains_seed(seed),
        )):
            print(f"[WARNING] Seed {seed} has partial artifacts and will be rerun.")
        train_one_seed(seed)


# ============================================================
# 9. Numerical Equality / Model Sanity Check
# ============================================================
def run_input_alignment_check(paths: list[str]) -> dict[str, object]:
    metrics = compare_framework_inputs(paths, IMG_SIZE)
    print("Input Alignment Check")
    print(f"Samples : {metrics['samples']}")
    print(f"Keras Shape  : {metrics['keras_shape']}")
    print(f"PyTorch Shape: {metrics['pytorch_shape']}")
    print(f"Dtype   : {metrics['dtype']}")
    print(f"Range   : [{metrics['minimum']:.8f}, {metrics['maximum']:.8f}]")
    print(f"MAE     : {metrics['mae']:.12g}")
    print(f"Max Abs : {metrics['max_abs']:.12g}")
    print(f"MSE     : {metrics['mse']:.12g}")
    print(f"All Close: {metrics['allclose']}")
    assert metrics["allclose"], metrics
    print("Input Alignment Check: PASS")
    return metrics


def run_sanity_check() -> None:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    set_seed(SEEDS[0])
    print_tensorflow_environment(EXPERIMENT, SEEDS[0])
    manifest = build_split_manifest()
    counts = split_counts(manifest)
    assert is_physically_split()
    assert counts == {"train": 10251, "val": 2194, "test": 2203}
    assert all({str(row["class_name"]) for row in rows} == set(CLASSES) for rows in manifest.values())

    fixed_paths = [str(row["path"]) for row in manifest["train"][:8]]
    run_input_alignment_check(fixed_paths)
    sequence = make_sequence(manifest["train"][:BATCH_SIZE], SEEDS[0], True)
    images, labels = sequence[0]
    model = build_model(SEEDS[0])
    output = model(images, training=False).numpy()
    assert images.shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
    assert labels.shape == (BATCH_SIZE,)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES) and np.isfinite(output).all()
    print(f"Dataset: {find_dataset_root()} | {counts}")
    print(f"Classes: {dict((name, index) for index, name in enumerate(CLASSES))}")
    print(f"Model Output: {output.shape} | parameters={model.count_params():,} | finite=True")


# ============================================================
# 10. Entry Point
# ============================================================
def print_experiment_config() -> None:
    print("Experiment: 01 Input Tensor Alignment")
    print("Changed Variable: Input Tensor only")
    print(f"Seeds: {SEEDS}")
    print(f"RUN_TRAINING: {RUN_TRAINING}")


if __name__ == "__main__":
    print_experiment_config()
    run_sanity_check()
    if RUN_TRAINING:
        run_3seed_experiment()
    else:
        print("Training was not started.\nSet RUN_TRAINING = True to start training.")
