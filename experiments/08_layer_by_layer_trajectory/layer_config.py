"""Experiment 08 identity and selected layer-trace cases."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "08_layer_by_layer_trajectory"
TITLE = "Experiment 08 - Layer-by-Layer Training Trajectory Analysis"
RESULTS = ROOT / "experiments" / EXPERIMENT / "results"
PARENT07 = ROOT / "experiments/07_multistep_state_resynchronization"
PARENT07_RESULTS = PARENT07 / "results"
PARENT06_RESULTS = ROOT / "experiments/06_batchnorm_state_divergence/results"
SEEDS = (42, 123, 2026)
UPDATE_RATIO_THRESHOLD = 1e-12
TRACE_EQUIVALENCE_ATOL = 1e-12
TRACE_EQUIVALENCE_RTOL = 1e-7
RUN_FULL_TRAINING = False


def select_epoch1_control() -> tuple[int, dict[int, float]]:
    """Choose the smallest mean K/P-anchor post-weight divergence at Epoch 1."""
    scores: dict[int, float] = {}
    for seed in SEEDS:
        path = PARENT07_RESULTS / "resynchronized" / f"seed{seed}_resync_probe.csv"
        with path.open(encoding="utf-8") as handle:
            rows = [
                row for row in csv.DictReader(handle)
                if row["checkpoint"] == "epoch_001" and row["anchor"] in {"keras", "pytorch"}
            ]
        if len(rows) != 2:
            raise RuntimeError(f"Expected two Experiment 07 Epoch 1 anchor rows: {path}")
        scores[seed] = sum(float(row["post_weight_relative_l2"]) for row in rows) / 2.0
    return min(scores, key=lambda seed: (scores[seed], seed)), scores


def selected_cases() -> list[dict[str, object]]:
    low_seed, _ = select_epoch1_control()
    cases = [
        {"seed": seed, "checkpoint": "initial", "anchor": "shared"}
        for seed in SEEDS
    ]
    cases.extend(
        {"seed": low_seed, "checkpoint": "epoch_001", "anchor": anchor}
        for anchor in ("keras", "pytorch")
    )
    cases.extend(
        {"seed": seed, "checkpoint": "epoch_030", "anchor": anchor}
        for seed in (123, 2026) for anchor in ("keras", "pytorch")
    )
    for case in cases:
        case["source_framework"] = "keras" if case["anchor"] == "shared" else case["anchor"]
        case["case_id"] = f"seed{case['seed']}_{case['checkpoint']}_{case['anchor']}"
    return cases


def fingerprint() -> dict[str, object]:
    low_seed, scores = select_epoch1_control()
    return {
        "experiment": EXPERIMENT,
        "parent_experiment": "07_multistep_state_resynchronization",
        "checkpoint_source": "06_batchnorm_state_divergence",
        "fixed_probe_id": "exp07_fixed_probe_v1",
        "cases": selected_cases(),
        "epoch1_control_selection": {
            "rule": "minimum mean K/P-anchor synchronized post_weight_relative_l2",
            "selected_seed": low_seed,
            "scores": scores,
        },
        "optimizer": "common_reference_adam_float32_v1",
        "batchnorm": "common_bn_population_variance_rate_0.01_v1",
        "update_ratio_threshold": UPDATE_RATIO_THRESHOLD,
        "trace_equivalence_atol": TRACE_EQUIVALENCE_ATOL,
        "trace_equivalence_rtol": TRACE_EQUIVALENCE_RTOL,
        "one_forward_backward_update_per_case": True,
        "full_training": False,
    }


def config_hash() -> str:
    payload = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
