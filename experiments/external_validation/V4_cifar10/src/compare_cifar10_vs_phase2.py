"""Post-process V4 Stage A and completed Stage B against read-only Phase 2 artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from common.trace_utils import write_csv, write_json
from v4_config import EXP08_RESULTS, RESULTS, SEEDS, config_hash


def _rows(path):
    with path.open() as handle: return list(csv.DictReader(handle))


def main() -> None:
    v4 = {int(row["seed"]): row for row in _rows(RESULTS / "summaries/stage_a_case_summary.csv")}
    phase2 = {int(row["seed"]): row for row in _rows(EXP08_RESULTS / "summaries/case_summary.csv") if row["checkpoint"] == "initial" and row["anchor"] == "shared"}
    comparison = []
    for seed in SEEDS:
        p, v = phase2[seed], v4[seed]
        comparison.append({
            "seed": seed, "phase2_first_nonzero_stage": p["first_nonzero_forward_stage"],
            "v4_first_nonzero_stage": v["first_nonzero_forward_stage"],
            "phase2_exact_prefix": 3, "v4_exact_prefix": v["exact_forward_prefix_length"],
            "phase2_largest_forward_stage": p["largest_forward_rel_l2_stage"],
            "v4_largest_forward_stage": v["largest_forward_stage"],
            "phase2_largest_forward_rel_l2": p["largest_forward_rel_l2"],
            "v4_largest_forward_rel_l2": v["largest_forward_rel_l2"],
            "phase2_first_backward_stage": p["first_nonzero_backward_stage"],
            "v4_first_backward_stage": v["first_nonzero_backward_stage"],
            "phase2_largest_parameter_gradient_group": p["largest_parameter_gradient_group"],
            "v4_largest_parameter_gradient_group": v["largest_parameter_gradient_group"],
            "phase2_largest_update_group": p["largest_update_group"],
            "v4_largest_update_group": v["largest_update_group"],
            "comparison_note": "independent workloads; not a pure dataset effect",
        })
    write_csv(RESULTS / "summaries/cifar10_vs_youngaffectnet_initial.csv", comparison)
    preflight_files = ("dataset_integrity.json", "split_check.json", "architecture_equivalence.json", "parameter_equivalence.json", "gpu_device_check.json", "fixed_batch_check.json", "shared_w0_check.json", "phase2_shared_body_w0_check.json", "resync_precheck.json", "repeatability.json", "trace_equivalence.json")
    reports = {name: json.loads((RESULTS / "preflight" / name).read_text()) for name in preflight_files}
    stage_a_valid = all(next(value for key, value in report.items() if key.startswith("v4_")) in ("VALID", True) for report in reports.values())
    training_path = RESULTS / "preflight/training_validity.json"
    training = json.loads(training_path.read_text()) if training_path.exists() else None
    final_valid = stage_a_valid and training is not None and training.get("v4_training_runs_complete") is True and training.get("v4_test_evaluation_complete") is True
    write_json(RESULTS / "summaries/diagnostic_summary.json", {
        "validation": "V4 - CIFAR-10 Independent Workload Validation",
        "status": "VALID" if final_valid else "STAGE_A_VALID_STAGE_B_PENDING" if stage_a_valid else "INVALID",
        "stage_a_valid": stage_a_valid, "stage_b_complete": training is not None,
        "v4_training_runs_complete": bool(training and training.get("v4_training_runs_complete")),
        "new_download_executed_by_analysis": False, "training_executed_by_analysis": False,
        "config_sha256": config_hash(), "stage_a_cases": list(v4.values()),
    })
    print(f"V4_STATUS = {'Completed / VALID' if final_valid else 'Stage A VALID / Stage B Pending' if stage_a_valid else 'INVALID'}")


if __name__ == "__main__": main()
