"""Validate Experiment 08 against Experiment 07 and create compact summaries."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.trace_utils import write_csv, write_json
from layer_config import (
    PARENT07_RESULTS, RESULTS, TRACE_EQUIVALENCE_ATOL, TRACE_EQUIVALENCE_RTOL,
    config_hash, select_epoch1_control, selected_cases,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _experiment07_index() -> dict[tuple[int, str, str], dict[str, str]]:
    indexed = {}
    for seed in (42, 123, 2026):
        for row in _read_csv(PARENT07_RESULTS / "resynchronized" / f"seed{seed}_resync_probe.csv"):
            indexed[(seed, row["checkpoint"], row["anchor"])] = row
    return indexed


def summarize() -> dict[str, object]:
    cases = selected_cases()
    summary_rows = _read_csv(RESULTS / "summaries/case_summary.csv")
    summary_index = {row["case_id"]: row for row in summary_rows}
    expected_ids = {str(case["case_id"]) for case in cases}
    selected_complete = set(summary_index) == expected_ids
    nan_or_inf = any(row["nan_or_inf"].lower() == "true" for row in summary_rows)

    parent = _experiment07_index()
    equivalence_rows = []
    mapping = {
        "loss_keras": "loss_keras",
        "loss_pytorch": "loss_pytorch",
        "global_gradient_rel_l2": "gradient_relative_l2",
        "global_update_rel_l2": "update_relative_l2",
        "global_post_weight_rel_l2": "post_weight_relative_l2",
    }
    for case in cases:
        actual = summary_index[str(case["case_id"])]
        expected = parent[(int(case["seed"]), str(case["checkpoint"]), str(case["anchor"]))]
        for actual_name, expected_name in mapping.items():
            a, e = float(actual[actual_name]), float(expected[expected_name])
            exact = a == e
            within = math.isclose(a, e, rel_tol=TRACE_EQUIVALENCE_RTOL, abs_tol=TRACE_EQUIVALENCE_ATOL)
            equivalence_rows.append({
                "case_id": case["case_id"], "metric": actual_name,
                "experiment08": a, "experiment07": e,
                "absolute_difference": abs(a - e), "exact_equal": exact,
                "within_tolerance": within,
                "atol": TRACE_EQUIVALENCE_ATOL, "rtol": TRACE_EQUIVALENCE_RTOL,
            })
    equivalence_valid = all(bool(row["within_tolerance"]) for row in equivalence_rows)
    write_csv(RESULTS / "preflight/trace_equivalence_rows.csv", equivalence_rows)
    equivalence_report = {
        "experiment": "08_layer_by_layer_trajectory",
        "trace_equivalence": "VALID" if equivalence_valid else "INVALID",
        "all_exact": all(bool(row["exact_equal"]) for row in equivalence_rows),
        "comparisons": len(equivalence_rows),
        "atol": TRACE_EQUIVALENCE_ATOL,
        "rtol": TRACE_EQUIVALENCE_RTOL,
        "instrumentation_behavior_unchanged_within_tolerance": equivalence_valid,
        "new_full_training_executed": False,
    }
    write_json(RESULTS / "preflight/trace_equivalence.json", equivalence_report)

    anchor_rows = []
    for seed, checkpoint in ((select_epoch1_control()[0], "epoch_001"), (123, "epoch_030"), (2026, "epoch_030")):
        k = summary_index[f"seed{seed}_{checkpoint}_keras"]
        p = summary_index[f"seed{seed}_{checkpoint}_pytorch"]
        anchor_rows.append({
            "seed": seed, "checkpoint": checkpoint,
            "keras_anchor_first_forward": k["first_nonzero_forward_stage"],
            "pytorch_anchor_first_forward": p["first_nonzero_forward_stage"],
            "keras_anchor_largest_forward": k["largest_forward_rel_l2_stage"],
            "pytorch_anchor_largest_forward": p["largest_forward_rel_l2_stage"],
            "keras_anchor_largest_activation_gradient": k["largest_activation_gradient_stage"],
            "pytorch_anchor_largest_activation_gradient": p["largest_activation_gradient_stage"],
            "keras_anchor_largest_parameter_gradient": k["largest_parameter_gradient_group"],
            "pytorch_anchor_largest_parameter_gradient": p["largest_parameter_gradient_group"],
            "keras_anchor_largest_update": k["largest_update_group"],
            "pytorch_anchor_largest_update": p["largest_update_group"],
            "keras_anchor_global_gradient_rel_l2": k["global_gradient_rel_l2"],
            "pytorch_anchor_global_gradient_rel_l2": p["global_gradient_rel_l2"],
            "keras_anchor_global_update_rel_l2": k["global_update_rel_l2"],
            "pytorch_anchor_global_update_rel_l2": p["global_update_rel_l2"],
            "keras_anchor_global_post_weight_rel_l2": k["global_post_weight_rel_l2"],
            "pytorch_anchor_global_post_weight_rel_l2": p["global_post_weight_rel_l2"],
        })
    write_csv(RESULTS / "summaries/anchor_comparison.csv", anchor_rows)

    preflight = json.loads((RESULTS / "preflight/layer_trace_precheck.json").read_text(encoding="utf-8"))
    overall_valid = (
        preflight.get("layer_trace_precheck") == "VALID"
        and equivalence_valid and selected_complete and not nan_or_inf
    )
    diagnostic = {
        "experiment": "08_layer_by_layer_trajectory",
        "status": "VALID" if overall_valid else "INVALID",
        "config_sha256": config_hash(),
        "layer_trace_precheck": preflight.get("layer_trace_precheck"),
        "trace_equivalence": equivalence_report["trace_equivalence"],
        "trace_equivalence_all_exact": equivalence_report["all_exact"],
        "selected_cases_complete": selected_complete,
        "selected_case_count": len(summary_rows),
        "nan_inf_found": nan_or_inf,
        "new_full_training_executed": False,
        "epoch1_control": preflight["epoch1_control"],
        "fixed_probe_batch_hash": preflight["fixed_probe_batch_hash"],
        "cases": summary_rows,
        "interpretation_guardrail": (
            "First non-zero, largest divergence, and local amplification are descriptive; "
            "none alone establishes a root cause."
        ),
        "phase2_termination_decided": False,
    }
    write_json(RESULTS / "summaries/diagnostic_summary.json", diagnostic)
    print(f"TRACE_EQUIVALENCE = {equivalence_report['trace_equivalence']}")
    print(f"SELECTED_CASES_COMPLETE = {str(selected_complete).upper()}")
    print(f"NAN_INF_FOUND = {str(nan_or_inf).upper()}")
    print(f"EXPERIMENT08_STATUS = {diagnostic['status']}")
    print("NEW_FULL_TRAINING_EXECUTED = FALSE")
    return diagnostic


if __name__ == "__main__":
    summarize()
