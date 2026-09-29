"""V2 identity, immutable controls, and safety guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = ROOT / "experiments/external_validation/V2_resnet18_bn"
RESULTS = EXPERIMENT_ROOT / "results"
EXP07 = ROOT / "experiments/07_multistep_state_resynchronization"
EXP07_RESULTS = EXP07 / "results"
EXP08_RESULTS = ROOT / "experiments/08_layer_by_layer_trajectory/results"

SEEDS = (42, 123, 2026)
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 8
DTYPE = "float32"
RUN_FULL_TRAINING = False
PRETRAINED = False
DEVICE = {"keras": "TensorFlow Metal GPU", "pytorch": "MPS GPU"}
REPEATABILITY_ATOL = 1e-12
REPEATABILITY_RTOL = 1e-7
COMMON_BN_ID = "common_bn_population_variance_rate_0.01_v1"
COMMON_ADAM_ID = "common_reference_adam_float32_v1"


def cases() -> list[dict[str, object]]:
    return [{
        "case_id": f"seed{seed}_initial_shared",
        "seed": seed,
        "checkpoint": "initial",
        "anchor": "shared",
        "optimizer_step": 0,
    } for seed in SEEDS]


def fingerprint() -> dict[str, object]:
    return {
        "study": "external_validation",
        "validation": "V2_resnet18_batchnorm_architecture",
        "changed_factor": "architecture_only_custom_cnn_to_resnet18",
        "architecture": "direct_semantic_post_activation_resnet18_no_pretraining",
        "input": [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3],
        "classes": NUM_CLASSES,
        "seeds": SEEDS,
        "cases": cases(),
        "execution": DEVICE,
        "dtype": DTYPE,
        "optimizer": COMMON_ADAM_ID,
        "batchnorm": COMMON_BN_ID,
        "repeatability": {
            "runs_per_framework": 2,
            "valid": "all compared arrays exact",
            "qualified": {"atol": REPEATABILITY_ATOL, "rtol": REPEATABILITY_RTOL},
            "invalid": "non-finite or outside qualified tolerance",
        },
        "pretrained": PRETRAINED,
        "full_training": RUN_FULL_TRAINING,
    }


def config_hash() -> str:
    payload = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
