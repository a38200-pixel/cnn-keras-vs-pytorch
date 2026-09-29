"""Experiment 07 identity and checkpoint-local probe constants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common.controlled_config import BATCH_SIZE, SEEDS

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "07_multistep_state_resynchronization"
TITLE = "Experiment 07 - Multi-Step Divergence & State Re-Synchronization"
RESULTS = ROOT / "experiments" / EXPERIMENT / "results"
PARENT_RESULTS = ROOT / "experiments/06_batchnorm_state_divergence/results"
PARENT_CONFIG_SHA256 = "30027b92347ba69e28d358345630ca2eb4f0b46abbc82729010fc9806044036c"

CHECKPOINTS = (
    "initial", "after_first_step", "epoch_001", "epoch_005",
    "epoch_010", "epoch_020", "epoch_030",
)
CHECKPOINT_STEPS = {
    "initial": 0,
    "after_first_step": 1,
    "epoch_001": 321,
    "epoch_005": 1605,
    "epoch_010": 3210,
    "epoch_020": 6420,
    "epoch_030": 9630,
}
PROBE_ID = "exp07_fixed_probe_v1"
PROBE_AUGMENTATION_SEED = 7007
PROBE_AUGMENTATION_EPOCH = 0
RUN_FULL_TRAINING = False


def anchors(checkpoint: str) -> tuple[str, ...]:
    return ("shared",) if checkpoint == "initial" else ("keras", "pytorch")


def fingerprint() -> dict[str, object]:
    return {
        "experiment": EXPERIMENT,
        "parent_experiment": "06_batchnorm_state_divergence",
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "seeds": list(SEEDS),
        "checkpoints": list(CHECKPOINTS),
        "checkpoint_steps": CHECKPOINT_STEPS,
        "probe_id": PROBE_ID,
        "probe_batch_size": BATCH_SIZE,
        "probe_augmentation_seed": PROBE_AUGMENTATION_SEED,
        "probe_augmentation_epoch": PROBE_AUGMENTATION_EPOCH,
        "probe_types": ["free_running", "full_state_resynchronized"],
        "anchors": ["keras", "pytorch"],
        "optimizer": "common_reference_adam_float32_v1",
        "batchnorm": "common_bn_population_variance_rate_0.01_v1",
        "full_training": False,
    }


def config_hash() -> str:
    payload = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
