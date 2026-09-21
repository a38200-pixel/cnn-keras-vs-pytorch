"""Single source of truth for Experiment 04 controlled conditions."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "04_common_initialization_controlled_training"
PHASE = "strict_controlled_comparison"
RESULTS = ROOT / "experiments" / EXPERIMENT / "results"
SEEDS = (42, 123, 2026)
IMAGE_SIZE = 128
BATCH_SIZE = 32
MAX_EPOCHS = 30
NUM_CLASSES = 8
DTYPE = "float32"
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-7
WEIGHT_DECAY = 0.0
AMS_GRAD = False
BN_EPS = 1e-3
KERAS_BN_MOMENTUM = 0.99
TORCH_BN_MOMENTUM = 0.01
FLIP_PROB = 0.5
ROTATION_DEGREES = 5.0
CHECKPOINT_EPOCHS = (1, 5, 10, 20, 30)
TRAINABLE_PARAMETERS = 422_824
DIAGNOSTIC_STEPS = (0, 1, 2, 5, 10, 20, 50, 100)
SIGN_EPS = 1e-12
RELATIVE_L2_EPS = 1e-12
TOP_K_OUTLIERS = 10
SAVE_INITIAL_CHECKPOINT = True
SAVE_FIRST_STEP_CHECKPOINT = True
SAVE_CANONICAL_CHECKPOINT = True
SAVE_NATIVE_CHECKPOINT = False  # canonical model + optimizer state is the official analysis artifact
ALLOW_OVERWRITE = False
BATCH_SCHEDULE_VERSION = "numpy_seedsequence_order_v1"
AUGMENTATION_VERSION = "sha256_sample_epoch_pil_v1"


def fingerprint() -> dict[str, object]:
    return {
        "experiment": EXPERIMENT, "phase": PHASE, "seeds": SEEDS,
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "epochs": MAX_EPOCHS, "dtype": DTYPE, "classes": NUM_CLASSES,
        "loss": "integer-label mean cross-entropy from raw logits",
        "adam": {
            "lr": LEARNING_RATE, "beta1": BETA1, "beta2": BETA2,
            "epsilon": EPSILON, "weight_decay": WEIGHT_DECAY,
            "amsgrad": AMS_GRAD, "gradient_clipping": None,
        },
        "batchnorm": {
            "epsilon": BN_EPS, "keras_momentum": KERAS_BN_MOMENTUM,
            "pytorch_momentum": TORCH_BN_MOMENTUM,
        },
        "augmentation": "common PIL flip p=.5 then rotation Uniform[-5,5] bilinear fill=0",
        "shuffle": "persisted NumPy permutations; framework-native shuffle=False",
        "early_stopping": False, "lr_scheduler": False,
        "dropout": False, "mixed_precision": False,
        "batch_schedule_version": BATCH_SCHEDULE_VERSION,
        "augmentation_version": AUGMENTATION_VERSION,
    }


def config_hash() -> str:
    payload = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
