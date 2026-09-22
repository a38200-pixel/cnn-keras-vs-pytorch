"""Experiment 05 identity; all other controls inherit Experiment 04."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common.controlled_config import *  # noqa: F403 - intentional single source of controlled constants
from common.controlled_config import fingerprint as baseline_fingerprint

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "05_gradient_optimizer_divergence"
TITLE = "Experiment 05 - Common Adam Optimizer Control"
RESULTS = ROOT / "experiments" / EXPERIMENT / "results"
OPTIMIZER_IMPLEMENTATION = "common_reference_adam_float32_v1"


def fingerprint() -> dict[str, object]:
    value = baseline_fingerprint()
    value.update({
        "experiment": EXPERIMENT,
        "parent_experiment": "04_common_initialization_controlled_training",
        "optimizer_implementation": OPTIMIZER_IMPLEMENTATION,
        "gradient_source": "framework-native backward exported to canonical float32 NumPy",
        "update_path": "canonical CommonAdam then load trainable values into framework model",
    })
    return value


def config_hash() -> str:
    payload = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
