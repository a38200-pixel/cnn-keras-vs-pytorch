"""Aggregate Experiment 07 free-running and full-state re-synchronized probes."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.trace_utils import write_csv, write_json
from experiment_config import CHECKPOINTS, RESULTS, SEEDS, anchors, config_hash
from probe_utils import require_preflight

CORE = (
    "gradient_relative_l2", "update_relative_l2", "post_weight_relative_l2",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def compare_results() -> dict[str, object]:
    require_preflight()
    free = []
    resync = []
    for seed in SEEDS:
        free.extend(_read(RESULTS / "free_running" / f"seed{seed}_free_running_probe.csv"))
        resync.extend(_read(RESULTS / "resynchronized" / f"seed{seed}_resync_probe.csv"))
    if len(free) != 21 or len(resync) != 39:
        raise RuntimeError(f"Unexpected probe row count: free={len(free)} resync={len(resync)}")
    if any(row["nan_or_inf"].lower() != "false" for row in [*free, *resync]):
        raise RuntimeError("NaN/Inf reported by a probe")

    summary_rows = []
    anchor_sensitivity = {}
    for checkpoint in CHECKPOINTS:
        free_rows = [row for row in free if row["checkpoint"] == checkpoint]
        if len(free_rows) != len(SEEDS):
            raise RuntimeError(f"Incomplete free-running checkpoint: {checkpoint}")
        for anchor in anchors(checkpoint):
            sync_rows = [
                row for row in resync
                if row["checkpoint"] == checkpoint and row["anchor"] == anchor
            ]
            if len(sync_rows) != len(SEEDS):
                raise RuntimeError(f"Incomplete re-synchronized checkpoint: {checkpoint}/{anchor}")
            summary_rows.append({
                "checkpoint": checkpoint,
                "optimizer_step": int(free_rows[0]["optimizer_step"]),
                "anchor": anchor,
                "free_before_weight_relative_l2_mean": _mean(free_rows, "before_weight_relative_l2"),
                "free_post_weight_relative_l2_mean": _mean(free_rows, "post_weight_relative_l2"),
                "free_weight_delta_mean": _mean(free_rows, "weight_divergence_delta"),
                "free_gradient_relative_l2_mean": _mean(free_rows, "gradient_relative_l2"),
                "free_update_relative_l2_mean": _mean(free_rows, "update_relative_l2"),
                "resync_gradient_relative_l2_mean": _mean(sync_rows, "gradient_relative_l2"),
                "resync_gradient_cosine_mean": _mean(sync_rows, "gradient_cosine"),
                "resync_update_relative_l2_mean": _mean(sync_rows, "update_relative_l2"),
                "resync_post_weight_relative_l2_mean": _mean(sync_rows, "post_weight_relative_l2"),
                "resync_bn_mean_relative_l2_mean": _mean(sync_rows, "bn_mean_relative_l2"),
                "resync_bn_variance_relative_l2_mean": _mean(sync_rows, "bn_variance_relative_l2"),
                "probe_config_sha256": config_hash(),
            })
        if checkpoint != "initial":
            per_metric = {}
            for metric in CORE:
                differences = []
                by_seed = {}
                for seed in SEEDS:
                    keras = next(
                        row for row in resync
                        if row["checkpoint"] == checkpoint and row["anchor"] == "keras"
                        and int(row["seed"]) == seed
                    )
                    pytorch = next(
                        row for row in resync
                        if row["checkpoint"] == checkpoint and row["anchor"] == "pytorch"
                        and int(row["seed"]) == seed
                    )
                    difference = abs(float(keras[metric]) - float(pytorch[metric]))
                    differences.append(difference)
                    by_seed[str(seed)] = {
                        "keras_anchor": float(keras[metric]),
                        "pytorch_anchor": float(pytorch[metric]),
                        "absolute_difference": difference,
                    }
                per_metric[metric] = {
                    "by_seed": by_seed,
                    "mean_absolute_anchor_difference": float(np.mean(differences)),
                    "max_absolute_anchor_difference": float(np.max(differences)),
                }
            anchor_sensitivity[checkpoint] = per_metric

    write_csv(RESULTS / "comparison/free_vs_resync_summary.csv", summary_rows)
    by_seed = {}
    for seed in SEEDS:
        by_seed[str(seed)] = {
            checkpoint: {
                "free_running": next(
                    {key: value for key, value in row.items() if key != "probe_config_sha256"}
                    for row in free if int(row["seed"]) == seed and row["checkpoint"] == checkpoint
                ),
                "resynchronized": {
                    anchor: next(
                        {key: value for key, value in row.items() if key != "probe_config_sha256"}
                        for row in resync
                        if int(row["seed"]) == seed and row["checkpoint"] == checkpoint
                        and row["anchor"] == anchor
                    ) for anchor in anchors(checkpoint)
                },
            } for checkpoint in CHECKPOINTS
        }
    output = {
        "experiment": "07_multistep_state_resynchronization",
        "status": "VALID",
        "resync_precheck": "VALID",
        "probe_ready": True,
        "config_sha256": config_hash(),
        "free_running_rows": len(free),
        "resynchronized_rows": len(resync),
        "nan_or_inf": False,
        "checkpoint_summary": summary_rows,
        "anchor_sensitivity": anchor_sensitivity,
        "by_seed": by_seed,
        "full_training_executed": False,
        "experiment06_checkpoints_modified": False,
    }
    write_json(RESULTS / "comparison/diagnostic_3seed_summary.json", output)
    print("Experiment 07 comparison completed: status=VALID; no full training was run.")
    return output


if __name__ == "__main__":
    compare_results()
