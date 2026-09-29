"""V3 identity, immutable controls, and safety guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = ROOT / "experiments/external_validation/V3_bn_free_cnn"
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
DEVICE = {"keras": "TensorFlow Metal GPU", "pytorch": "MPS GPU"}
REPEATABILITY_ATOL = 1e-12
REPEATABILITY_RTOL = 1e-7
COMMON_ADAM_ID = "common_reference_adam_float32_v1"
PHASE2_W0_SOURCE = "04_common_initialization_controlled_training/results/initialization"
FIXED_BATCH_SHA256 = "c3ef4b5c6cb842154ddb03fa6cdc175dd5ab4e69a1e02f2bbc5493c3e2a48e7c"


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
        "validation": "V3_batchnorm_free_custom_cnn",
        "changed_factor": "batchnorm_presence",
        "architecture": "phase2_custom_cnn_geometry_minus_all_batchnorm",
        "conv_bias": False,
        "input": [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3],
        "classes": NUM_CLASSES,
        "seeds": SEEDS,
        "cases": cases(),
        "execution": DEVICE,
        "dtype": DTYPE,
        "optimizer": COMMON_ADAM_ID,
        "initialization_source": PHASE2_W0_SOURCE,
        "fixed_batch_sha256": FIXED_BATCH_SHA256,
        "repeatability": {
            "runs_per_framework": 2,
            "valid": "all compared arrays exact",
            "qualified": {"atol": REPEATABILITY_ATOL, "rtol": REPEATABILITY_RTOL},
            "invalid": "non-finite or outside qualified tolerance",
        },
        "one_forward_backward_update_per_case": True,
        "full_training": RUN_FULL_TRAINING,
    }


def config_hash() -> str:
    payload = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
