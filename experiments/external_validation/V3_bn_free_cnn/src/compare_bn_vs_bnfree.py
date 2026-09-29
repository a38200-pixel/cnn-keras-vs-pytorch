"""Join V3 BN-free initial traces to Experiment 08 BN initial traces."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from common.trace_utils import write_csv, write_json
from v3_config import EXP08_RESULTS, RESULTS, RUN_FULL_TRAINING, SEEDS, config_hash


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bn_forward(seed: int) -> list[dict[str, str]]:
    return _read_csv(EXP08_RESULTS / "forward" / f"seed{seed}_initial_shared_forward.csv")


def _exact_prefix(rows: list[dict[str, str]]) -> tuple[int, list[str]]:
    stages = []
    for row in rows:
        if row["stage"] == "loss" or float(row["max_abs_diff"]) > 0:
            break
        stages.append(row["stage"])
    return len(stages), stages


def run() -> dict[str, object]:
    if RUN_FULL_TRAINING:
        raise RuntimeError("V3 safety guard violated")
    v3_rows = {int(row["seed"]): row for row in _read_csv(RESULTS / "summaries/case_summary.csv")}
    bn_rows = {
        int(row["seed"]): row
        for row in _read_csv(EXP08_RESULTS / "summaries/case_summary.csv")
        if row["checkpoint"] == "initial" and row["anchor"] == "shared"
    }
    if set(v3_rows) != set(bn_rows) or set(v3_rows) != set(SEEDS):
        raise RuntimeError("V3/Experiment 08 initial Seed coverage mismatch")

    comparison = []
    for seed in SEEDS:
        bn, bnfree = bn_rows[seed], v3_rows[seed]
        bn_forward = _bn_forward(seed)
        bn_prefix_length, bn_prefix_stages = _exact_prefix(bn_forward)
        bn_logits = next(row for row in bn_forward if row["stage"] == "logits")
        comparison.append({
            "seed": seed,
            "bn_first_nonzero_stage": bn["first_nonzero_forward_stage"],
            "bnfree_first_nonzero_stage": bnfree["first_nonzero_forward_stage"],
            "bn_exact_prefix": bn_prefix_length,
            "bnfree_exact_prefix": bnfree["exact_forward_prefix_length"],
            "bn_exact_prefix_stages": json.dumps(bn_prefix_stages),
            "bnfree_exact_prefix_stages": bnfree["exact_forward_prefix_stages"],
            "bn_largest_forward_stage": bn["largest_forward_rel_l2_stage"],
            "bnfree_largest_forward_stage": bnfree["largest_forward_stage"],
            "bn_largest_forward_rel_l2": bn["largest_forward_rel_l2"],
            "bnfree_largest_forward_rel_l2": bnfree["largest_forward_rel_l2"],
            "bnfree_to_bn_forward_ratio": (
                float(bnfree["largest_forward_rel_l2"]) / float(bn["largest_forward_rel_l2"])
                if float(bn["largest_forward_rel_l2"]) else ""
            ),
            "bn_logits_rel_l2": bn_logits["relative_l2"],
            "bnfree_logits_rel_l2": bnfree["logits_rel_l2"],
            "bn_loss_abs_diff": bn["loss_abs_diff"],
            "bnfree_loss_abs_diff": bnfree["loss_abs_diff"],
            "bn_first_nonzero_backward_stage": bn["first_nonzero_backward_stage"],
            "bnfree_first_nonzero_backward_stage": bnfree["first_nonzero_backward_stage"],
            "bn_largest_backward_stage": bn["largest_activation_gradient_stage"],
            "bnfree_largest_backward_stage": bnfree["largest_backward_stage"],
            "bn_largest_parameter_gradient_group": bn["largest_parameter_gradient_group"],
            "bnfree_largest_parameter_gradient_group": bnfree["largest_parameter_gradient_group"],
            "bn_largest_update_group": bn["largest_update_group"],
            "bnfree_largest_update_group": bnfree["largest_update_group"],
            "bn_global_gradient_rel_l2": bn["global_gradient_rel_l2"],
            "bnfree_global_gradient_rel_l2": bnfree["global_gradient_rel_l2"],
            "bn_global_update_rel_l2": bn["global_update_rel_l2"],
            "bnfree_global_update_rel_l2": bnfree["global_update_rel_l2"],
            "bn_post_weight_rel_l2": bn["global_post_weight_rel_l2"],
            "bnfree_post_weight_rel_l2": bnfree["global_post_weight_rel_l2"],
        })
    write_csv(RESULTS / "summaries/bn_vs_bnfree_initial.csv", comparison)

    report_names = (
        "architecture_equivalence.json", "parameter_equivalence.json", "no_batchnorm_check.json",
        "gpu_device_check.json", "fixed_batch_check.json", "shared_w0_check.json",
        "resync_precheck.json", "trace_equivalence.json", "repeatability.json",
    )
    reports = {
        filename: json.loads((RESULTS / "preflight" / filename).read_text(encoding="utf-8"))
        for filename in report_names
    }
    required_valid = {
        "v3_architecture_equivalence": reports["architecture_equivalence.json"].get("v3_architecture_equivalence"),
        "v3_parameter_equivalence": reports["parameter_equivalence.json"].get("v3_parameter_equivalence"),
        "v3_no_batchnorm_check": reports["no_batchnorm_check.json"].get("v3_no_batchnorm_check"),
        "v3_gpu_device_check": reports["gpu_device_check.json"].get("v3_gpu_device_check"),
        "v3_fixed_batch_check": reports["fixed_batch_check.json"].get("v3_fixed_batch_check"),
        "v3_shared_w0_check": reports["shared_w0_check.json"].get("v3_shared_w0_check"),
        "v3_phase2_shared_parameter_w0": reports["shared_w0_check.json"].get("v3_phase2_shared_parameter_w0"),
        "v3_resync_precheck": reports["resync_precheck.json"].get("v3_resync_precheck"),
        "v3_trace_equivalence": reports["trace_equivalence.json"].get("v3_trace_equivalence"),
    }
    repeatability = reports["repeatability.json"].get("v3_gpu_repeatability", "INVALID")
    complete = len(v3_rows) == 3
    nan_inf = any(str(row["nan_inf"]).lower() == "true" for row in v3_rows.values())
    overall_valid = (
        all(value == "VALID" for value in required_valid.values())
        and repeatability in {"VALID", "QUALIFIED"}
        and complete and not nan_inf
    )
    diagnostic = {
        "study": "External Validation Study",
        "validation": "V3 - BatchNorm-Free Custom CNN Validation",
        "status": "VALID" if overall_valid else "INVALID",
        **required_valid,
        "v3_gpu_repeatability": repeatability,
        "v3_selected_cases_complete": complete, "selected_case_count": len(v3_rows),
        "nan_inf_found": nan_inf, "new_full_training_executed": False,
        "v4_executed": False, "config_sha256": config_hash(),
        "cases": list(v3_rows.values()),
    }
    write_json(RESULTS / "summaries/diagnostic_summary.json", diagnostic)
    print(f"V3_SELECTED_CASES_COMPLETE = {str(complete).upper()}")
    print(f"NAN_INF_FOUND = {str(nan_inf).upper()}")
    print(f"V3_STATUS = {'Completed / VALID' if overall_valid else 'INVALID'}")
    print("NEW_FULL_TRAINING_EXECUTED = FALSE")
    print("V4_EXECUTED = FALSE")
    return diagnostic


if __name__ == "__main__":
    run()
