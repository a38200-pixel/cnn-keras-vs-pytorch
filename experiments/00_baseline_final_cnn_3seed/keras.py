"""Keras reproduction of the notebook's final RGB CNN. Training is opt-in."""
# ============================================================
# 1. Imports
# ============================================================
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
# This required filename is keras.py; remove its directory so it cannot shadow
# the installed keras package when TensorFlow initializes tf.keras.
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))
from common.alignment_utils import augmentation_decision, epoch_permutation
from common.environment_utils import print_tensorflow_environment
from common.dataset_utils import (
    CLASSES,
    build_split_manifest,
    find_dataset_root,
    is_physically_split,
    split_counts,
)
from common.result_utils import save_result
from common.seed_utils import seed_tensorflow

# ============================================================
# 2. Experiment Configuration
# ============================================================
EXPERIMENT = "00_baseline_final_cnn_3seed"
ALIGNMENT = "baseline"  # changed to exactly one factor in experiments 01-08
SEEDS = [42, 123, 2026]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS = 128, 32, 8, 30
LEARNING_RATE, EARLY_PATIENCE, LR_PATIENCE, MIN_DELTA = 0.001, 7, 4, 1e-4
RUN_TRAINING = False
RESULTS = Path(__file__).parent / "results" / "keras_results.csv"

# ============================================================
# 3. Random Seed
# ============================================================
def set_seed(seed: int) -> None:
    seed_tensorflow(seed)

# ============================================================
# 4. Dataset
# ============================================================
def _load_image(path: str, training: bool, seed: int, epoch: int, aligned_input: bool):
    """Baseline uses TensorFlow decoding; aligned branches use common PIL pixels."""
    from PIL import Image, ImageOps
    if aligned_input:
        image = Image.open(path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    else:
        import tensorflow as tf
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method="bilinear").numpy()
    if training and ALIGNMENT == "augmentation":
        flip, angle = augmentation_decision(path, seed, epoch)
        if flip:
            image = ImageOps.mirror(image)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    array = np.asarray(image, dtype=np.float32)
    return array / 255.0 if ALIGNMENT in {"input_tensor", "diagnostic"} else array


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
            ids = self.indices[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
            xs, ys = [], []
            for i in ids:
                row = rows[i]
                xs.append(_load_image(str(row["path"]), training, seed, self.epoch,
                                      aligned_input=ALIGNMENT in {"input_tensor", "augmentation", "diagnostic"}))
                ys.append(int(row["label"]))
            return np.stack(xs), np.asarray(ys, dtype=np.int32)
        def on_epoch_end(self, initial=False):
            if not initial:
                self.epoch += 1
            if training:
                if ALIGNMENT == "batch_order":
                    self.indices = epoch_permutation(len(rows), seed, self.epoch)
                else:
                    self.indices = np.random.default_rng(seed + self.epoch).permutation(len(rows)).tolist()
    return ImageSequence()

# ============================================================
# 5. CNN Model
# ============================================================
def build_model(seed: int):
    import tensorflow as tf
    from keras import layers
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="input")
    x = inputs
    if ALIGNMENT not in {"input_tensor", "diagnostic"}:
        x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)
    if ALIGNMENT not in {"augmentation", "diagnostic"}:
        x = layers.RandomFlip("horizontal", seed=seed, name="augmentation_flip")(x)
        x = layers.RandomRotation(0.014, seed=seed + 1, name="augmentation_rotation")(x)
    for i, filters in enumerate((32, 64, 128, 256), 1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        # Baseline Keras defaults are epsilon=.001 and momentum=.99.
        x = layers.BatchNormalization(epsilon=1e-3, momentum=.99, name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, name="fc128")(x)
    x = layers.ReLU(name="fc_relu")(x)
    logits = layers.Dense(NUM_CLASSES, name="logits")(x)
    outputs = logits if ALIGNMENT == "output_loss" else layers.Softmax(name="softmax")(logits)
    model = tf.keras.Model(inputs, outputs, name="final_rgb_cnn")
    if ALIGNMENT in {"initial_weight", "diagnostic"}:
        initialize_common_weights(model, seed)
    return model


def initialize_common_weights(model, seed: int) -> None:
    """NumPy Glorot values; BN gamma/beta/mean/variance retain shared constants."""
    rng = np.random.default_rng(seed)
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        updated = []
        for i, value in enumerate(weights):
            if value.ndim >= 2:
                fan_in, fan_out = np.prod(value.shape[:-1]), value.shape[-1]
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                updated.append(rng.uniform(-limit, limit, value.shape).astype("float32"))
            else:
                updated.append(value)
        layer.set_weights(updated)


def make_optimizer_and_loss():
    import tensorflow as tf
    epsilon = 1e-7  # Keras baseline default; explicit in Adam alignment.
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=.9, beta_2=.999,
                                         epsilon=epsilon, amsgrad=False)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=ALIGNMENT == "output_loss")
    return optimizer, loss

# ============================================================
# 6. Training
# ============================================================
def train_one_seed(seed: int) -> None:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score
    set_seed(seed)
    print_tensorflow_environment(EXPERIMENT, seed)
    manifest = build_split_manifest()
    train, val, test = (make_sequence(manifest[s], seed, s == "train") for s in ("train", "val", "test"))
    model = build_model(seed)
    optimizer, loss = make_optimizer_and_loss()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=EARLY_PATIENCE,
            min_delta=MIN_DELTA, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=.5,
            patience=LR_PATIENCE, min_delta=MIN_DELTA, min_lr=1e-6),
    ]
    start = time.perf_counter()
    history = model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=callbacks, verbose=1)
    elapsed = time.perf_counter() - start
    model.save(Path(__file__).parent / "results" / f"keras_seed{seed}.keras")
    test_loss, test_acc = model.evaluate(test, verbose=0)
    y_true = np.concatenate([test[i][1] for i in range(len(test))])
    predictions = np.argmax(model.predict(test, verbose=0), axis=1)
    best = int(getattr(callbacks[0], "best_epoch", np.argmin(history.history["val_loss"])))
    save_result(RESULTS, {"experiment": EXPERIMENT, "framework": "keras", "seed": seed,
        "test_accuracy": accuracy_score(y_true, predictions), "macro_f1": f1_score(y_true, predictions, average="macro"),
        "test_loss": test_loss, "best_epoch": best + 1, "epochs_trained": len(history.history["loss"]),
        "train_accuracy": history.history["accuracy"][best], "validation_accuracy": history.history["val_accuracy"][best],
        "train_loss": history.history["loss"][best], "validation_loss": history.history["val_loss"][best],
        "train_val_gap": history.history["accuracy"][best] - history.history["val_accuracy"][best],
        "training_time_seconds": elapsed})

# ============================================================
# 7. Evaluation / sanity check
# ============================================================
def run_sanity_check() -> None:
    import tensorflow as tf
    set_seed(SEEDS[0])
    print_tensorflow_environment(EXPERIMENT, SEEDS[0])
    manifest = build_split_manifest()
    counts = split_counts(manifest)
    assert is_physically_split()
    assert counts == {"train": 10251, "val": 2194, "test": 2203}
    for split, rows in manifest.items():
        assert {str(row["class_name"]) for row in rows} == set(CLASSES), split
    sequence = make_sequence(manifest["train"][:BATCH_SIZE], SEEDS[0], True)
    images, labels = sequence[0]
    model = build_model(SEEDS[0])
    output = model(images, training=False).numpy()
    assert images.shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
    assert labels.shape == (BATCH_SIZE,)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES) and np.isfinite(output).all()
    print(f"dataset={find_dataset_root()} splits={counts}")
    print(f"classes={CLASSES} class_to_index={dict((name, index) for index, name in enumerate(CLASSES))}")
    model.summary()
    print(f"batch={images.shape} labels={labels.shape} output={output.shape} "
          f"parameters={model.count_params():,} finite=True")

# ============================================================
# 8. Result Saving / entry point
# ============================================================
def print_experiment_config() -> None:
    print({"experiment": EXPERIMENT, "framework": "keras", "alignment": ALIGNMENT,
           "seeds": SEEDS, "run_training": RUN_TRAINING})

def run_3seed_experiment() -> None:
    for seed in SEEDS:
        train_one_seed(seed)

if __name__ == "__main__":
    print_experiment_config()
    run_sanity_check()
    if RUN_TRAINING:
        run_3seed_experiment()
    else:
        print("Training was not started.\nSet RUN_TRAINING = True to start training.")
