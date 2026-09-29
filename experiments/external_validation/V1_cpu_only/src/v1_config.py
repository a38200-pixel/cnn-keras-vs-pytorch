"""Identity, inherited cases, and safety guards for V1 CPU-only validation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = ROOT / "experiments/external_validation/V1_cpu_only"
RESULTS = EXPERIMENT_ROOT / "results"
EXP06_RESULTS = ROOT / "experiments/06_batchnorm_state_divergence/results"
EXP07 = ROOT / "experiments/07_multistep_state_resynchronization"
EXP07_RESULTS = EXP07 / "results"
EXP08 = ROOT / "experiments/08_layer_by_layer_trajectory"
EXP08_RESULTS = EXP08 / "results"
RUN_FULL_TRAINING = False
TF_INTRA_OP_THREADS = 1
TF_INTER_OP_THREADS = 1
TORCH_NUM_THREADS = 1
TORCH_INTER_OP_THREADS = 1
DTYPE = "float32"

EXPECTED_CASES = (
    (42, "initial", "shared"),
    (123, "initial", "shared"),
    (2026, "initial", "shared"),
    (42, "epoch_001", "keras"),
    (42, "epoch_001", "pytorch"),
    (123, "epoch_030", "keras"),
    (123, "epoch_030", "pytorch"),
    (2026, "epoch_030", "keras"),
    (2026, "epoch_030", "pytorch"),
)


def selected_cases() -> list[dict[str, object]]:
    """Read, validate, and reuse Experiment 08's official case manifest."""
    path = EXP08_RESULTS / "manifests/trace_case_manifest.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual = tuple((int(row["seed"]), row["checkpoint"], row["anchor"]) for row in rows)
    if actual != EXPECTED_CASES:
        raise RuntimeError(f"Experiment 08 selected-case manifest changed: {actual}")
    return [{
        "case_id": row["case_id"],
        "seed": int(row["seed"]),
        "checkpoint": row["checkpoint"],
        "anchor": row["anchor"],
        "source_framework": "keras" if row["anchor"] == "shared" else row["anchor"],
        "optimizer_step": int(row["optimizer_step"]),
        "experiment08_source_state_hash": row["source_state_hash"],
        "probe_batch_hash": row["probe_batch_hash"],
    } for row in rows]


def fingerprint() -> dict[str, object]:
    return {
        "study": "external_validation",
        "validation": "V1_cpu_only_execution",
        "parent": "Experiment 08",
        "cases": selected_cases(),
        "execution": {"keras": "CPU", "pytorch": "CPU"},
        "threads": {
            "tensorflow_intra_op": TF_INTRA_OP_THREADS,
            "tensorflow_inter_op": TF_INTER_OP_THREADS,
            "pytorch_num_threads": TORCH_NUM_THREADS,
            "pytorch_inter_op_threads": TORCH_INTER_OP_THREADS,
        },
        "dtype": DTYPE,
        "optimizer": "common_reference_adam_float32_v1",
        "batchnorm": "common_bn_population_variance_rate_0.01_v1",
        "repeat_runs_per_framework": 2,
        "plain_vs_trace_equivalence": True,
        "full_training": RUN_FULL_TRAINING,
    }


def config_hash() -> str:
    payload = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
