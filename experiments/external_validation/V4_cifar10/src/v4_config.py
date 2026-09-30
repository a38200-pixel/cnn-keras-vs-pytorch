"""Canonical V4 research configuration and safety constants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = ROOT / "experiments/external_validation/V4_cifar10"
RESULTS = EXPERIMENT_ROOT / "results"
PHASE2_RESULTS = ROOT / "experiments/06_batchnorm_state_divergence/results"
EXP08_RESULTS = ROOT / "experiments/08_layer_by_layer_trajectory/results"

SEEDS = (42, 123, 2026)
SPLIT_SEED = 42
IMAGE_SIZE = 32
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 30
TRAIN_SIZE = 45_000
VAL_SIZE = 5_000
TEST_SIZE = 10_000
STEPS_PER_EPOCH = 1_407
TOTAL_STEPS = 42_210
EPOCH_SNAPSHOT_EPOCHS = (0, 1, 5, 10, 20, 30)
STEP_SNAPSHOT_STEPS = (0, 321, 1_605, 3_210, 6_420, 9_630)
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPSILON = 1e-7
BN_EPSILON = 1e-3
BN_UPDATE_RATE = 0.01
DTYPE = "float32"
RUN_STAGE_A = False
RUN_FULL_TRAINING = False
REPEATABILITY_ATOL = 1e-12
REPEATABILITY_RTOL = 1e-7


def validate_static_config() -> None:
    if (TRAIN_SIZE + BATCH_SIZE - 1) // BATCH_SIZE != STEPS_PER_EPOCH:
        raise RuntimeError("V4 steps/epoch mismatch")
    if STEPS_PER_EPOCH * EPOCHS != TOTAL_STEPS:
        raise RuntimeError("V4 total-step mismatch")
    if TRAIN_SIZE % BATCH_SIZE != 8:
        raise RuntimeError("V4 final batch must contain 8 samples")
    if any(step <= 0 or step >= TOTAL_STEPS for step in STEP_SNAPSHOT_STEPS[1:]):
        raise RuntimeError("Invalid step snapshot schedule")


def fingerprint() -> dict[str, object]:
    validate_static_config()
    return {
        "validation": "V4_CIFAR10_independent_workload",
        "workload": {
            "dataset": "CIFAR-10", "input": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "classes": NUM_CLASSES, "split_seed": SPLIT_SEED,
            "train": TRAIN_SIZE, "validation": VAL_SIZE, "test": TEST_SIZE,
            "preprocessing": "uint8_to_float32_div255", "augmentation": None,
        },
        "training": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "steps_per_epoch": STEPS_PER_EPOCH, "total_steps": TOTAL_STEPS,
            "seeds": SEEDS, "shuffle": "persisted_numpy_permutation",
            "drop_last": False, "scheduler": None, "early_stopping": False,
        },
        "optimizer": {
            "implementation": "CommonAdam", "lr": LEARNING_RATE,
            "beta1": BETA1, "beta2": BETA2, "epsilon": ADAM_EPSILON,
            "weight_decay": 0.0, "amsgrad": False, "gradient_clipping": None,
        },
        "batchnorm": {
            "implementation": "CommonBN", "epsilon": BN_EPSILON,
            "variance": "population_ddof0", "running_update_rate": BN_UPDATE_RATE,
        },
        "dtype": DTYPE, "epoch_snapshots": EPOCH_SNAPSHOT_EPOCHS,
        "step_snapshots": STEP_SNAPSHOT_STEPS,
        "run_stage_a_default": RUN_STAGE_A,
        "run_full_training_default": RUN_FULL_TRAINING,
    }


def config_hash() -> str:
    payload = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
