"""Join V2 ResNet18 initial traces to Experiment 08 custom-CNN initial traces."""

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
from v2_config import EXP08_RESULTS, RESULTS, RUN_FULL_TRAINING, SEEDS, config_hash


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run() -> dict[str, object]:
    if RUN_FULL_TRAINING:
        raise RuntimeError("V2 safety guard violated")
    v2_rows = {int(row["seed"]): row for row in _read_csv(RESULTS / "summaries/case_summary.csv")}
    custom_rows = {
        int(row["seed"]): row
        for row in _read_csv(EXP08_RESULTS / "summaries/case_summary.csv")
        if row["checkpoint"] == "initial" and row["anchor"] == "shared"
    }
    if set(v2_rows) != set(custom_rows) or set(v2_rows) != set(SEEDS):
        raise RuntimeError("V2/Experiment 08 initial Seed coverage mismatch")
    comparison = []
    for seed in SEEDS:
        custom, resnet = custom_rows[seed], v2_rows[seed]
        comparison.append({
            "seed": seed,
            "custom_first_nonzero_stage": custom["first_nonzero_forward_stage"],
            "resnet18_first_nonzero_stage": resnet["first_nonzero_forward_stage"],
            "custom_largest_forward_stage": custom["largest_forward_rel_l2_stage"],
            "resnet18_largest_forward_stage": resnet["largest_forward_stage"],
            "custom_largest_forward_rel_l2": custom["largest_forward_rel_l2"],
            "resnet18_largest_forward_rel_l2": resnet["largest_forward_rel_l2"],
            "resnet18_to_custom_forward_ratio": (
                float(resnet["largest_forward_rel_l2"]) / float(custom["largest_forward_rel_l2"])
                if float(custom["largest_forward_rel_l2"]) else ""
            ),
            "custom_largest_backward_stage": custom["largest_activation_gradient_stage"],
            "resnet18_largest_backward_stage": resnet["largest_backward_stage"],
            "custom_largest_parameter_gradient_group": custom["largest_parameter_gradient_group"],
            "resnet18_largest_parameter_gradient_group": resnet["largest_parameter_gradient_group"],
            "custom_largest_update_group": custom["largest_update_group"],
            "resnet18_largest_update_group": resnet["largest_update_group"],
            "custom_global_gradient_rel_l2": custom["global_gradient_rel_l2"],
            "resnet18_global_gradient_rel_l2": resnet["global_gradient_rel_l2"],
            "custom_global_update_rel_l2": custom["global_update_rel_l2"],
            "resnet18_global_update_rel_l2": resnet["global_update_rel_l2"],
            "custom_post_weight_rel_l2": custom["global_post_weight_rel_l2"],
            "resnet18_post_weight_rel_l2": resnet["global_post_weight_rel_l2"],
        })
    write_csv(RESULTS / "summaries/resnet18_vs_custom_cnn_initial.csv", comparison)

    reports = {
        filename: json.loads((RESULTS / "preflight" / filename).read_text(encoding="utf-8"))
        for filename in (
            "architecture_equivalence.json", "parameter_equivalence.json", "gpu_device_check.json",
            "resync_precheck.json", "trace_equivalence.json", "repeatability.json",
        )
    }
    required_valid = {
        "resnet18_architecture_equivalence": reports["architecture_equivalence.json"].get("resnet18_architecture_equivalence"),
        "resnet18_parameter_equivalence": reports["parameter_equivalence.json"].get("resnet18_parameter_equivalence"),
        "v2_gpu_device_check": reports["gpu_device_check.json"].get("v2_gpu_device_check"),
        "v2_fixed_batch_check": reports["resync_precheck.json"].get("v2_fixed_batch_check"),
        "v2_resync_precheck": reports["resync_precheck.json"].get("v2_resync_precheck"),
        "v2_trace_equivalence": reports["trace_equivalence.json"].get("v2_trace_equivalence"),
    }
    repeatability = reports["repeatability.json"].get("v2_gpu_repeatability", "INVALID")
    complete = len(v2_rows) == 3
    nan_inf = any(str(row["nan_inf"]).lower() == "true" for row in v2_rows.values())
    overall_valid = (
        all(value == "VALID" for value in required_valid.values())
        and repeatability in {"VALID", "QUALIFIED"}
        and complete and not nan_inf
    )
    diagnostic = {
        "study": "External Validation Study",
        "validation": "V2 - ResNet18 + BatchNorm Architecture Validation",
        "status": "VALID" if overall_valid else "INVALID",
        **required_valid,
        "v2_gpu_repeatability": repeatability,
        "v2_selected_cases_complete": complete, "selected_case_count": len(v2_rows),
        "nan_inf_found": nan_inf, "new_full_training_executed": False,
        "v3_v4_executed": False, "pretrained_model_used": False,
        "torchvision_model_used_for_comparison": False,
        "config_sha256": config_hash(), "cases": list(v2_rows.values()),
    }
    write_json(RESULTS / "summaries/diagnostic_summary.json", diagnostic)
    print(f"V2_SELECTED_CASES_COMPLETE = {str(complete).upper()}")
    print(f"NAN_INF_FOUND = {str(nan_inf).upper()}")
    print(f"V2_STATUS = {'Completed / VALID' if overall_valid else 'INVALID'}")
    print("NEW_FULL_TRAINING_EXECUTED = FALSE")
    print("V3_V4_EXECUTED = FALSE")
    return diagnostic


if __name__ == "__main__":
    run()
