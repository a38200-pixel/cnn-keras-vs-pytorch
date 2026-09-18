"""Experiment 02: Keras Baseline with only training batch order aligned."""

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

from common.batch_order import (
    batch_groups,
    initialize_runtime_order_log,
    order_hash,
    prepare_batch_order_artifacts,
    record_runtime_order,
    runtime_order_path,
)
from common.dataset_utils import CLASSES, build_split_manifest, find_dataset_root, is_physically_split, split_counts
from common.environment_utils import print_tensorflow_environment, tensorflow_device_name
from common.history_utils import save_history
from common.result_utils import save_result
from common.seed_utils import seed_tensorflow


# ============================================================
# 1. Configuration
# ============================================================
EXPERIMENT = "02_batch_order_alignment"
ALIGNMENT = "batch_order"
SEEDS = [42, 123, 2026]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS = 128, 32, 8, 30
LEARNING_RATE, EARLY_PATIENCE, LR_PATIENCE, MIN_DELTA = 0.001, 7, 4, 1e-4
RUN_TRAINING = False
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS = RESULTS_DIR / "keras_results.csv"
HISTORY_DIR = RESULTS_DIR / "history"
BATCH_ORDER_DIR = RESULTS_DIR / "batch_order"
RUNTIME_ORDER_DIR = RESULTS_DIR / "runtime_order"


# ============================================================
# 2. Framework Seed (separate from persisted batch-order RNG)
# ============================================================
def set_seed(seed: int) -> None:
    seed_tensorflow(seed)


# ============================================================
# 3. Baseline Keras Input Preprocessing / Dataset
# ============================================================
def load_baseline_image(path: str) -> np.ndarray:
    """Preserve Baseline TensorFlow decode/resize and its [0,255] model input."""
    import tensorflow as tf

    encoded = tf.io.read_file(path)
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    return tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method="bilinear").numpy().astype(np.float32)


def make_sequence(
    rows,
    training: bool,
    epoch_orders: np.ndarray | None = None,
    runtime_seed: int | None = None,
    runtime_canonical=None,
    runtime_dataset_root: Path | None = None,
):
    import tensorflow as tf

    if training and epoch_orders is None:
        raise ValueError("Training requires the persisted common epoch-order schedule.")

    class OrderedImageSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.epoch = 0
            self.next_training_epoch = 0
            self.indices = self._indices_for_epoch()

        def _indices_for_epoch(self) -> list[int]:
            if training:
                return [int(index) for index in epoch_orders[self.epoch]]
            return list(range(len(rows)))

        def __len__(self):
            # drop_last=False: the final 11 samples form the same partial batch.
            return (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

        def __getitem__(self, batch_index):
            indices = self.indices[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
            images = [load_baseline_image(str(rows[index]["path"])) for index in indices]
            labels = [int(rows[index]["label"]) for index in indices]
            return np.stack(images), np.asarray(labels, dtype=np.int32)

        def on_epoch_begin(self):
            # Keras 3 calls adapter.reset() before fit, and reset() invokes
            # on_epoch_end(). Selecting the order here prevents that pre-fit
            # callback from shifting Keras one permutation ahead of PyTorch.
            if training:
                if self.next_training_epoch >= len(epoch_orders):
                    raise RuntimeError("Keras requested more epochs than the persisted schedule contains.")
                self.epoch = self.next_training_epoch
                self.indices = self._indices_for_epoch()
                self.next_training_epoch += 1
                if runtime_seed is not None:
                    if runtime_canonical is None or runtime_dataset_root is None:
                        raise RuntimeError("Runtime order logging requires canonical rows and dataset root.")
                    record_runtime_order(
                        RUNTIME_ORDER_DIR, "keras", runtime_seed,
                        epoch=self.epoch + 1, schedule_index=self.epoch,
                        order=self.indices, canonical=runtime_canonical,
                        dataset_root=runtime_dataset_root, batch_size=BATCH_SIZE,
                    )

        def on_epoch_end(self):
            # Intentionally do not advance here; Keras calls this once before
            # the first real epoch as well as after each completed epoch.
            pass

    return OrderedImageSequence()


# ============================================================
# 4. Baseline Keras Model / Augmentation
# ============================================================
def build_model(seed: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="input")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
    x = layers.RandomFlip("horizontal", seed=seed, name="augmentation_flip")(x)
    x = layers.RandomRotation(0.014, seed=seed + 1, name="augmentation_rotation")(x)
    for index, filters in enumerate((32, 64, 128, 256), 1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{index}")(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=.99, name=f"bn{index}")(x)
        x = layers.ReLU(name=f"relu{index}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{index}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, name="fc128")(x)
    x = layers.ReLU(name="fc_relu")(x)
    logits = layers.Dense(NUM_CLASSES, name="logits")(x)
    return tf.keras.Model(inputs, layers.Softmax(name="softmax")(logits), name="final_rgb_cnn")


def make_optimizer_and_loss():
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=.9, beta_2=.999,
        epsilon=1e-7, amsgrad=False,
    )
    return optimizer, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def make_callbacks(seed: int, history_rows: list[dict[str, object]]):
    import tensorflow as tf

    class EpochHistory(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            row = {
                "framework": "keras", "experiment": EXPERIMENT, "seed": seed, "epoch": epoch + 1,
                "train_loss": float(logs["loss"]), "train_accuracy": float(logs["accuracy"]),
                "val_loss": float(logs["val_loss"]), "val_accuracy": float(logs["val_accuracy"]),
                "learning_rate": learning_rate,
                "elapsed_epoch_sec": time.perf_counter() - self.started,
            }
            history_rows.append(row)
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | train_loss={row['train_loss']:.6f} "
                f"train_acc={row['train_accuracy']:.4f} | val_loss={row['val_loss']:.6f} "
                f"val_acc={row['val_accuracy']:.4f} | lr={learning_rate:.8f}", flush=True,
            )

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=EARLY_PATIENCE,
        min_delta=MIN_DELTA, restore_best_weights=True, verbose=1,
    )
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", mode="min", factor=.5, patience=LR_PATIENCE,
        min_delta=MIN_DELTA, min_lr=1e-6, verbose=1,
    )
    return early, scheduler, EpochHistory()


# ============================================================
# 5. Batch-Order Artifacts / Exact-Match Sanity
# ============================================================
def prepare_orders():
    manifest = build_split_manifest()
    canonical, schedules = prepare_batch_order_artifacts(
        manifest["train"], find_dataset_root(), BATCH_ORDER_DIR, SEEDS, EPOCHS,
    )
    return manifest, canonical, schedules


def run_batch_order_check(canonical, schedules) -> dict[int, str]:
    hashes: dict[int, str] = {}
    first_epoch_orders = []
    for seed in SEEDS:
        keras_sequence = make_sequence(canonical, True, schedules[seed])
        # Reproduce the Keras 3 fit lifecycle: pre-fit reset calls
        # on_epoch_end(), then the actual epoch calls on_epoch_begin().
        keras_sequence.on_epoch_end()
        observed_schedule_indices = []
        selected_hashes = {}
        for expected_index in range(EPOCHS):
            keras_sequence.on_epoch_begin()
            observed_schedule_indices.append(keras_sequence.epoch)
            keras_order_for_epoch = keras_sequence.indices
            pytorch_order_for_epoch = [int(index) for index in schedules[seed][expected_index]]
            assert keras_order_for_epoch == pytorch_order_for_epoch
            assert order_hash(keras_order_for_epoch) == order_hash(pytorch_order_for_epoch)
            if expected_index in {0, 1, EPOCHS - 1}:
                selected_hashes[expected_index + 1] = order_hash(keras_order_for_epoch)
            keras_sequence.on_epoch_end()
        expected_schedule_indices = list(range(EPOCHS))
        assert observed_schedule_indices == expected_schedule_indices

        keras_order = [int(index) for index in schedules[seed][0]]
        pytorch_sampler_order = [int(index) for index in schedules[seed][0]]
        keras_batches = batch_groups(keras_order, BATCH_SIZE)
        pytorch_batches = batch_groups(pytorch_sampler_order, BATCH_SIZE)
        exact = keras_batches == pytorch_batches
        keras_hash = order_hash(keras_order)
        pytorch_hash = order_hash(pytorch_sampler_order)
        assert exact and keras_hash == pytorch_hash
        assert len(keras_batches) == 321 and len(keras_batches[-1]) == 11
        assert not np.array_equal(schedules[seed][0], schedules[seed][1])
        first_epoch_orders.append(schedules[seed][0])
        hashes[seed] = keras_hash
        print("Batch Order Alignment Check")
        print(f"Seed: {seed} | Epoch: 1 | Samples: {len(keras_order)}")
        print(f"First 3 Keras batches : {keras_batches[:3]}")
        print(f"First 3 PyTorch batches: {pytorch_batches[:3]}")
        print(f"Keras order hash : {keras_hash}")
        print(f"PyTorch order hash: {pytorch_hash}")
        print(f"Exact match: {exact}")
        print(f"Expected lifecycle: {expected_schedule_indices}")
        print(f"Observed lifecycle: {observed_schedule_indices}")
        print(f"Lifecycle exact match: {observed_schedule_indices == expected_schedule_indices}")
        print(f"Epoch 1/2/30 shared hashes: {selected_hashes}")
    assert all(
        not np.array_equal(first_epoch_orders[left], first_epoch_orders[right])
        for left in range(len(SEEDS)) for right in range(left + 1, len(SEEDS))
    )
    print("Batch Order exact-match sanity check: PASS")
    return hashes


# ============================================================
# 6. Training / History / Evaluation
# ============================================================
def result_contains_seed(seed: int) -> bool:
    if not RESULTS.exists():
        return False
    with RESULTS.open(encoding="utf-8") as handle:
        return any(int(row["seed"]) == seed for row in csv.DictReader(handle))


def seed_is_complete(seed: int) -> bool:
    return result_contains_seed(seed) and (RESULTS_DIR / f"keras_seed{seed}.keras").exists() and (
        HISTORY_DIR / f"keras_seed{seed}_history.csv"
    ).exists() and runtime_order_path(RUNTIME_ORDER_DIR, "keras", seed).exists()


def seed_has_partial_artifacts(seed: int) -> bool:
    return result_contains_seed(seed) or (RESULTS_DIR / f"keras_seed{seed}.keras").exists() or (
        HISTORY_DIR / f"keras_seed{seed}_history.csv"
    ).exists() or runtime_order_path(RUNTIME_ORDER_DIR, "keras", seed).exists()


def print_training_header(seed: int) -> None:
    print("=" * 60)
    print("Experiment 02 - Batch Order Alignment")
    print("Framework : Keras")
    print(f"Seed      : {seed}")
    print(f"Device    : {tensorflow_device_name()}")
    print(f"Batch     : {BATCH_SIZE}")
    print(f"Max Epoch : {EPOCHS}")
    print("=" * 60)


def train_one_seed(seed: int) -> None:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score

    tf.keras.backend.clear_session()
    set_seed(seed)
    print_training_header(seed)
    print_tensorflow_environment(EXPERIMENT, seed)
    manifest, canonical, schedules = prepare_orders()
    initialize_runtime_order_log(RUNTIME_ORDER_DIR, "keras", seed)
    train = make_sequence(
        canonical, True, schedules[seed], runtime_seed=seed,
        runtime_canonical=canonical, runtime_dataset_root=find_dataset_root(),
    )
    val = make_sequence(manifest["val"], False)
    test = make_sequence(manifest["test"], False)
    model = build_model(seed)
    optimizer, loss = make_optimizer_and_loss()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    history_rows: list[dict[str, object]] = []
    early, scheduler, recorder = make_callbacks(seed, history_rows)
    started = time.perf_counter()
    history = model.fit(
        train, validation_data=val, epochs=EPOCHS,
        callbacks=[early, scheduler, recorder], verbose=1, shuffle=False,
    )
    elapsed = time.perf_counter() - started
    history_path = HISTORY_DIR / f"keras_seed{seed}_history.csv"
    save_history(history_path, history_rows)
    print(f"Saved history immediately: {history_path}")
    model_path = RESULTS_DIR / f"keras_seed{seed}.keras"
    model.save(model_path)
    test_loss, _ = model.evaluate(test, verbose=0)
    y_true = np.concatenate([test[index][1] for index in range(len(test))])
    predictions = np.argmax(model.predict(test, verbose=0), axis=1)
    best = int(getattr(early, "best_epoch", np.argmin(history.history["val_loss"])))
    save_result(RESULTS, {
        "experiment": EXPERIMENT, "framework": "keras", "seed": seed,
        "test_accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro"),
        "test_loss": test_loss, "best_epoch": best + 1, "epochs_trained": len(history_rows),
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
        else:
            if seed_has_partial_artifacts(seed):
                print(f"[WARNING] Seed {seed} has partial artifacts; this run will complete/replace them.")
            train_one_seed(seed)


# ============================================================
# 7. Dataset / Model Sanity and Entry Point
# ============================================================
def run_sanity_check() -> None:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    set_seed(SEEDS[0])
    print_tensorflow_environment(EXPERIMENT, SEEDS[0])
    manifest, canonical, schedules = prepare_orders()
    counts = split_counts(manifest)
    assert is_physically_split() and counts == {"train": 10251, "val": 2194, "test": 2203}
    assert all({str(row["class_name"]) for row in rows} == set(CLASSES) for rows in manifest.values())
    run_batch_order_check(canonical, schedules)
    sequence = make_sequence(canonical[:BATCH_SIZE], False)
    images, labels = sequence[0]
    model = build_model(SEEDS[0])
    output = model(images, training=False).numpy()
    assert images.shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3) and labels.shape == (BATCH_SIZE,)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES) and np.isfinite(output).all()
    print(f"Dataset: {find_dataset_root()} | {counts}")
    print(f"Classes: {dict((name, index) for index, name in enumerate(CLASSES))}")
    print(f"Model Output: {output.shape} | parameters={model.count_params():,} | finite=True")


def print_experiment_config() -> None:
    print("Experiment: 02 Batch Order Alignment")
    print("Changed Variable: Training Batch Order only")
    print(f"Seeds: {SEEDS}")
    print(f"RUN_TRAINING: {RUN_TRAINING}")


if __name__ == "__main__":
    print_experiment_config()
    run_sanity_check()
    if RUN_TRAINING:
        run_3seed_experiment()
    else:
        print("Training was not started.\nSet RUN_TRAINING = True to start training.")
