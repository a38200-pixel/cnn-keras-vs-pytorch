"""Experiment 03: Keras Baseline with only augmentation parameters aligned."""

from __future__ import annotations

import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from common.augmentation_runtime import (
    AugmentationRuntimeRecorder,
    apply_tensorflow_augmentation,
    augmentation_parameters,
    expected_records,
    numerical_operator_diagnostic,
    parameter_hash,
    runtime_summary_path,
    runtime_values_path,
    sample_identity,
    save_json,
)
from common.dataset_utils import (
    CLASSES, build_split_manifest, find_dataset_root, is_physically_split, split_counts,
)
from common.environment_utils import print_tensorflow_environment, tensorflow_device_name
from common.history_utils import save_history
from common.result_utils import save_result
from common.seed_utils import seed_tensorflow


EXPERIMENT = "03_augmentation_alignment"
ALIGNMENT = "augmentation"
SEEDS = [42, 123, 2026]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS = 128, 32, 8, 30
LEARNING_RATE, EARLY_PATIENCE, LR_PATIENCE, MIN_DELTA = 0.001, 7, 4, 1e-4
RUN_TRAINING = False
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS = RESULTS_DIR / "keras_results.csv"
HISTORY_DIR = RESULTS_DIR / "history"
RUNTIME_AUG_DIR = RESULTS_DIR / "runtime_augmentation"
SANITY_DIR = RESULTS_DIR / "sanity"


def set_seed(seed: int) -> None:
    seed_tensorflow(seed)


def configuration_fingerprint() -> dict[str, object]:
    return {
        "branch_source": "00_baseline_final_cnn_3seed",
        "changed_variable": "augmentation_only",
        "model": "Conv(32,64,128,256)-BN-ReLU-MaxPool-GAP-Dense128-Dense8",
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "optimizer": "Adam(beta_1=0.9,beta_2=0.999,epsilon=1e-7,amsgrad=False)",
        "learning_rate": LEARNING_RATE,
        "loss": "SparseCategoricalCrossentropy(from_logits=False)",
        "batchnorm": "epsilon=1e-3,momentum=0.99",
        "early_stopping": f"val_loss,patience={EARLY_PATIENCE},min_delta={MIN_DELTA},restore_best",
        "lr_scheduler": f"val_loss,factor=0.5,patience={LR_PATIENCE},min_lr=1e-6",
        "input": "TensorFlow decode_image RGB -> tf.image.resize bilinear -> model Rescaling(1/255)",
        "shuffle": "Baseline Keras Sequence NumPy permutation(seed + lifecycle_epoch)",
        "augmentation": "strict sample-ID; flip p=0.5 then Uniform[-5,+5] degree rotation; bilinear; constant 0",
        "validation_test_augmentation": False,
    }


def load_baseline_image(path: str) -> np.ndarray:
    """Preserve Baseline TensorFlow decoding, resize, and [0,255] model input."""
    import tensorflow as tf

    encoded = tf.io.read_file(path)
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    return tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method="bilinear").numpy().astype(np.float32)


def make_sequence(rows, seed: int, training: bool, recorder: AugmentationRuntimeRecorder | None = None):
    import tensorflow as tf

    dataset_root = find_dataset_root()

    class AugmentedImageSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.shuffle_epoch = 0
            self.augmentation_epoch = 0
            self.next_augmentation_epoch = 0
            self.runtime_epoch_active = False
            self.indices = list(range(len(rows)))
            self.on_epoch_end(initial=True)

        def __len__(self):
            return (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

        def __getitem__(self, batch_index):
            indices = self.indices[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
            images, labels = [], []
            for index in indices:
                row = rows[index]
                path = str(row["path"])
                image = load_baseline_image(path)
                if training:
                    identity = sample_identity(path, dataset_root)
                    flip, angle = augmentation_parameters(identity, seed, self.augmentation_epoch)
                    image = apply_tensorflow_augmentation(image, flip, angle).numpy()
                    if recorder is not None:
                        recorder.record(identity, flip, angle)
                images.append(image)
                labels.append(int(row["label"]))
            return np.stack(images), np.asarray(labels, dtype=np.int32)

        def on_epoch_begin(self):
            if training:
                if self.next_augmentation_epoch >= EPOCHS:
                    raise RuntimeError("Keras requested an augmentation epoch outside 0..29.")
                self.augmentation_epoch = self.next_augmentation_epoch
                self.next_augmentation_epoch += 1
                if recorder is not None:
                    recorder.begin_epoch(self.augmentation_epoch)
                    self.runtime_epoch_active = True

        def on_epoch_end(self, initial=False):
            # Keras 3 invokes this before the first real epoch. Augmentation
            # advances only in on_epoch_begin, preventing an epoch offset.
            if self.runtime_epoch_active and recorder is not None:
                recorder.finalize_epoch()
                self.runtime_epoch_active = False
            if not initial:
                self.shuffle_epoch += 1
            if training:
                self.indices = np.random.default_rng(seed + self.shuffle_epoch).permutation(len(rows)).tolist()

    return AugmentedImageSequence()


def build_model(seed: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="input")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
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
        learning_rate=LEARNING_RATE, beta_1=.9, beta_2=.999, epsilon=1e-7, amsgrad=False,
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


def result_contains_seed(seed: int) -> bool:
    if not RESULTS.exists():
        return False
    with RESULTS.open(encoding="utf-8") as handle:
        return any(int(row["seed"]) == seed for row in csv.DictReader(handle))


def seed_is_complete(seed: int) -> bool:
    return all((
        result_contains_seed(seed),
        (RESULTS_DIR / f"keras_seed{seed}.keras").exists(),
        (HISTORY_DIR / f"keras_seed{seed}_history.csv").exists(),
        runtime_summary_path(RUNTIME_AUG_DIR, "keras", seed).exists(),
        runtime_values_path(RUNTIME_AUG_DIR, "keras", seed).exists(),
    ))


def seed_has_partial_artifacts(seed: int) -> bool:
    return any((
        result_contains_seed(seed),
        (RESULTS_DIR / f"keras_seed{seed}.keras").exists(),
        (HISTORY_DIR / f"keras_seed{seed}_history.csv").exists(),
        runtime_summary_path(RUNTIME_AUG_DIR, "keras", seed).exists(),
        runtime_values_path(RUNTIME_AUG_DIR, "keras", seed).exists(),
    ))


def print_training_header(seed: int) -> None:
    print("=" * 60)
    print("Experiment 03 - Augmentation Alignment")
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
    manifest = build_split_manifest()
    identities = [sample_identity(row["path"], find_dataset_root()) for row in manifest["train"]]
    recorder = AugmentationRuntimeRecorder("keras", seed, identities, RUNTIME_AUG_DIR)
    train = make_sequence(manifest["train"], seed, True, recorder)
    val = make_sequence(manifest["val"], seed, False)
    test = make_sequence(manifest["test"], seed, False)
    model = build_model(seed)
    optimizer, loss = make_optimizer_and_loss()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    history_rows: list[dict[str, object]] = []
    early, scheduler, history_callback = make_callbacks(seed, history_rows)
    started = time.perf_counter()
    history = model.fit(
        train, validation_data=val, epochs=EPOCHS,
        callbacks=[early, scheduler, history_callback], verbose=1, shuffle=False,
    )
    elapsed = time.perf_counter() - started
    history_path = HISTORY_DIR / f"keras_seed{seed}_history.csv"
    save_history(history_path, history_rows)
    print(f"Saved history immediately: {history_path}")
    if len(recorder.completed) != len(history_rows):
        raise RuntimeError(
            f"Runtime/history epoch mismatch: runtime={len(recorder.completed)} history={len(history_rows)}"
        )
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
            print(f"[SKIP] Seed {seed} already has complete result/model/history/runtime artifacts.")
        else:
            if seed_has_partial_artifacts(seed):
                print(f"[WARNING] Seed {seed} has partial artifacts; this run will replace/complete them.")
            train_one_seed(seed)


def run_sanity_check() -> None:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    set_seed(SEEDS[0])
    print_tensorflow_environment(EXPERIMENT, SEEDS[0])
    manifest = build_split_manifest()
    counts = split_counts(manifest)
    assert is_physically_split() and counts == {"train": 10251, "val": 2194, "test": 2203}
    assert all({str(row["class_name"]) for row in rows} == set(CLASSES) for rows in manifest.values())
    dataset_root = find_dataset_root()
    dry_rows = manifest["train"][:BATCH_SIZE]
    dry_identities = [sample_identity(row["path"], dataset_root) for row in dry_rows]
    dry_hashes = {}
    with tempfile.TemporaryDirectory(prefix="exp03-keras-sanity-") as temp_dir:
        for seed in SEEDS:
            recorder = AugmentationRuntimeRecorder("keras", seed, dry_identities, Path(temp_dir))
            sequence = make_sequence(dry_rows, seed, True, recorder)
            sequence.on_epoch_end()  # reproduce Keras 3 pre-fit reset
            sequence.on_epoch_begin()
            images, labels = sequence[0]
            sequence.on_epoch_end()
            observed = recorder.completed[0][1]
            expected = expected_records(dry_identities, seed, 0)
            assert observed == expected
            dry_hashes[str(seed)] = parameter_hash(observed)

    lifecycle = make_sequence(dry_rows, SEEDS[0], True)
    lifecycle.on_epoch_end()
    observed_epochs = []
    for _ in range(EPOCHS):
        lifecycle.on_epoch_begin()
        observed_epochs.append(lifecycle.augmentation_epoch)
        lifecycle.on_epoch_end()
    assert observed_epochs == list(range(EPOCHS))
    model = build_model(SEEDS[0])
    output = model(images, training=False).numpy()
    assert images.shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
    assert labels.shape == (BATCH_SIZE,)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES) and np.isfinite(output).all()
    diagnostic = numerical_operator_diagnostic(
        [str(row["path"]) for row in manifest["train"][:8]], SEEDS[0], 0, dataset_root,
    )
    save_json(SANITY_DIR / "augmentation_numerical_diagnostic.json", diagnostic)
    report = {
        "status": "PASS", "mode": "strict_sample_id", "dataset_counts": counts,
        "seed_dry_runtime_hashes": dry_hashes, "epoch_schedule_mapping": observed_epochs,
        "runtime_logging_path_tested": True,
        "configuration_fingerprint": configuration_fingerprint(),
        "model_output_shape": list(output.shape), "model_parameters": model.count_params(),
        "tensorflow_gpu_detected": bool(tf.config.list_physical_devices("GPU")),
        "run_training": RUN_TRAINING,
    }
    save_json(SANITY_DIR / "keras_sanity_report.json", report)
    print(f"Dataset: {dataset_root} | {counts}")
    print(f"Strict sample-ID dry runtime hashes: {dry_hashes}")
    print(f"Epoch mapping: {observed_epochs[0]}..{observed_epochs[-1]} | exact=True")
    print(
        f"Numerical diagnostic: MAE={diagnostic['mae_mean']:.6f} "
        f"MSE={diagnostic['mse_mean']:.6f} MaxAbs={diagnostic['max_absolute_difference']:.6f}"
    )
    print(f"Model Output: {output.shape} | parameters={model.count_params():,} | finite=True")


def print_experiment_config() -> None:
    print("Experiment: 03 Augmentation Alignment")
    print("Changed Variable: Augmentation only")
    print("Alignment Mode: Strict sample-ID")
    print(f"Seeds: {SEEDS}")
    print(f"RUN_TRAINING: {RUN_TRAINING}")


if __name__ == "__main__":
    print_experiment_config()
    run_sanity_check()
    if RUN_TRAINING:
        run_3seed_experiment()
    else:
        print("Training was not started.\nSet RUN_TRAINING = True to start training.")
