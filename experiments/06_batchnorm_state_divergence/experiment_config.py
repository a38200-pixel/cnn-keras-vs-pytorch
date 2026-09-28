"""Experiment 06 identity; all non-BN controls inherit Experiment 05."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common.controlled_config import *  # noqa: F403 - intentional controlled constants
from common.controlled_config import fingerprint as baseline_fingerprint
from common.controlled_batchnorm import (
    COMMON_BN_EPSILON, COMMON_BN_SEMANTICS_ID, COMMON_BN_UPDATE_RATE,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "06_batchnorm_state_divergence"
TITLE = "Experiment 06 - BatchNorm State Divergence"
RESULTS = ROOT / "experiments" / EXPERIMENT / "results"
OPTIMIZER_IMPLEMENTATION = "common_reference_adam_float32_v1"


def fingerprint() -> dict[str, object]:
    value = baseline_fingerprint()
    value.update({
        "experiment": EXPERIMENT,
        "parent_experiment": "05_gradient_optimizer_divergence",
        "optimizer_implementation": OPTIMIZER_IMPLEMENTATION,
        "gradient_source": "framework-native backward exported to canonical float32 NumPy",
        "update_path": "canonical CommonAdam then load trainable values into framework model",
        "batchnorm": {
            "implementation": COMMON_BN_SEMANTICS_ID,
            "training_reduction": "channel statistics over N,H,W",
            "variance": "population_biased_ddof_0",
            "epsilon": COMMON_BN_EPSILON,
            "running_update_rate": COMMON_BN_UPDATE_RATE,
            "running_equation": "state=0.99*state+0.01*batch_stat",
            "evaluation": "running state, no state update",
            "keras_axes": [0, 1, 2],
            "pytorch_axes": [0, 2, 3],
        },
    })
    return value


def config_hash() -> str:
    payload = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
