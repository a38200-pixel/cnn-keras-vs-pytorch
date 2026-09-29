"""Run V1 plain/traced/repeated synchronized CPU one-step diagnostics."""

from __future__ import annotations

import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP07 = ROOT / "experiments/07_multistep_state_resynchronization"
EXP08 = ROOT / "experiments/08_layer_by_layer_trajectory"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP07))
sys.path.insert(0, str(EXP08))
sys.path.insert(0, str(HERE))

from common.common_adam_control import TRAINABLE_NAMES
from common.state_resynchronization import BN_MEAN_NAMES, BN_VARIANCE_NAMES, compare_group, states_exact
from common.trace_utils import assert_finite, compare, write_csv, write_json
from cpu_trace_utils import CpuLayerTraceRunner
from layer_trace import (
    _build_activation_gradient_rows, _build_bn_rows, _build_forward_rows,
    _build_parameter_rows, _build_update_rows, _first_nonzero,
)
from probe_utils import fixed_probe_batch, load_checkpoint
from v1_config import EXP08_RESULTS, RESULTS, config_hash, selected_cases


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _single_metric(left: np.ndarray, right: np.ndarray) -> tuple[bool, dict[str, object]]:
    return bool(np.array_equal(left, right)), compare(left, right)


def _state_metric(left, right, names):
    return states_exact(left, right, names), compare_group(left, right, names)


def _comparison_rows(case_id: str, comparison: str, framework: str, left, right):
    rows = []
    left_logits = left["logits"] if "logits" in left else left["forward"]["logits"]
    right_logits = right["logits"] if "logits" in right else right["forward"]["logits"]
    metrics = [
        ("logits",) + _single_metric(left_logits, right_logits),
        ("loss",) + _single_metric(
            np.asarray([left["loss"]], dtype=np.float32),
            np.asarray([right["loss"]], dtype=np.float32),
        ),
        ("global_gradient",) + _state_metric(
            left["parameter_gradients"], right["parameter_gradients"], TRAINABLE_NAMES,
        ),
        ("global_update",) + _state_metric(left["updates"], right["updates"], TRAINABLE_NAMES),
        ("post_weight",) + _state_metric(left["post"], right["post"], TRAINABLE_NAMES),
        ("post_bn_running_mean",) + _state_metric(left["post"], right["post"], BN_MEAN_NAMES),
        ("post_bn_running_variance",) + _state_metric(left["post"], right["post"], BN_VARIANCE_NAMES),
    ]
    for metric, exact, stats in metrics:
        rows.append({
            "case_id": case_id, "comparison": comparison, "framework": framework,
            "metric": metric, "exact_equal": exact,
            "max_abs_diff": stats["max_abs_diff"],
            "mean_abs_diff": stats["mean_abs_diff"],
            "relative_l2": stats["relative_l2_error"],
        })
    return rows


def _validate_finite(case_id, branch, values):
    for name in ("parameter_gradients", "updates", "adam_m", "adam_v", "post"):
        assert_finite(values[name], f"{case_id} {branch} {name}")
    if "forward" in values:
        assert_finite(values["forward"], f"{case_id} {branch} forward")
        assert_finite(values["activation_gradients"], f"{case_id} {branch} activation gradients")
    else:
        assert_finite({"logits": values["logits"]}, f"{case_id} {branch} logits")
    if not np.isfinite(values["loss"]):
        raise RuntimeError(f"NaN/Inf in {case_id} {branch} loss")


def run() -> dict[str, object]:
    preflight_path = RESULTS / "preflight/resync_precheck.json"
    device_path = RESULTS / "preflight/cpu_device_check.json"
    if not preflight_path.exists() or not device_path.exists():
        raise RuntimeError("Run v1_preflight.py before V1 trace")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    device = json.loads(device_path.read_text(encoding="utf-8"))
    if (
        preflight.get("v1_resync_precheck") != "VALID"
        or device.get("cpu_only_device_check") != "VALID"
        or preflight.get("config_sha256") != config_hash()
    ):
        raise RuntimeError("V1 CPU/resync preflight is not current and VALID")

    images, labels, _ = fixed_probe_batch(write_manifest=False)
    runner = CpuLayerTraceRunner(images, labels)
    case_summaries = []
    repeatability_rows = []
    equivalence_rows = []

    for case in selected_cases():
        case_id = str(case["case_id"])
        state, optimizer, step, _ = load_checkpoint(
            int(case["seed"]), str(case["source_framework"]), str(case["checkpoint"]),
        )
        exact = runner.validate_resync(state, optimizer, step)
        plain_k, plain_p = runner.run_plain(state, optimizer, step)
        _, trace1_k, trace1_p = runner.run_trace(state, optimizer, step)
        _, trace2_k, trace2_p = runner.run_trace(state, optimizer, step)
        for branch, values in (
            ("plain_keras", plain_k), ("plain_pytorch", plain_p),
            ("trace1_keras", trace1_k), ("trace1_pytorch", trace1_p),
            ("trace2_keras", trace2_k), ("trace2_pytorch", trace2_p),
        ):
            _validate_finite(case_id, branch, values)

        equivalence_rows.extend(_comparison_rows(case_id, "plain_vs_trace", "keras", plain_k, trace1_k))
        equivalence_rows.extend(_comparison_rows(case_id, "plain_vs_trace", "pytorch", plain_p, trace1_p))
        repeatability_rows.extend(_comparison_rows(case_id, "trace_run1_vs_run2", "keras", trace1_k, trace2_k))
        repeatability_rows.extend(_comparison_rows(case_id, "trace_run1_vs_run2", "pytorch", trace1_p, trace2_p))

        forward_rows = _build_forward_rows(case, trace1_k, trace1_p)
        activation_rows = _build_activation_gradient_rows(case, trace1_k, trace1_p)
        parameter_rows = _build_parameter_rows(case, trace1_k, trace1_p)
        update_rows = _build_update_rows(case, trace1_k, trace1_p)
        bn_rows = _build_bn_rows(case, forward_rows, parameter_rows, update_rows, trace1_k, trace1_p)
        write_csv(RESULTS / "forward" / f"{case_id}_forward.csv", forward_rows)
        write_csv(RESULTS / "backward" / f"{case_id}_activation_gradients.csv", activation_rows)
        write_csv(RESULTS / "backward" / f"{case_id}_parameter_gradients.csv", parameter_rows)
        write_csv(RESULTS / "optimizer" / f"{case_id}_parameter_updates.csv", update_rows)
        write_csv(RESULTS / "bn" / f"{case_id}_bn_trace.csv", bn_rows)

        largest_forward = max(forward_rows, key=lambda row: float(row["relative_l2"]))
        largest_backward = max(activation_rows, key=lambda row: float(row["relative_l2"]))
        largest_parameter = max(parameter_rows, key=lambda row: float(row["relative_l2"]))
        largest_update = max(update_rows, key=lambda row: float(row["update_relative_l2"]))
        global_gradient = compare_group(trace1_k["parameter_gradients"], trace1_p["parameter_gradients"], TRAINABLE_NAMES)
        global_update = compare_group(trace1_k["updates"], trace1_p["updates"], TRAINABLE_NAMES)
        global_post = compare_group(trace1_k["post"], trace1_p["post"], TRAINABLE_NAMES)
        case_repeatability = all(
            bool(row["exact_equal"]) for row in repeatability_rows if row["case_id"] == case_id
        )
        case_equivalence = all(
            bool(row["exact_equal"]) for row in equivalence_rows if row["case_id"] == case_id
        )
        case_summaries.append({
            "case_id": case_id, "seed": case["seed"], "checkpoint": case["checkpoint"],
            "optimizer_step": step, "anchor": case["anchor"],
            "full_state_sync_exact": exact["all_exact"],
            "first_nonzero_forward_stage": _first_nonzero(forward_rows),
            "largest_forward_stage": largest_forward["stage"],
            "largest_forward_rel_l2": largest_forward["relative_l2"],
            "loss_abs_diff": abs(trace1_k["loss"] - trace1_p["loss"]),
            "largest_backward_stage": largest_backward["stage"],
            "largest_backward_rel_l2": largest_backward["relative_l2"],
            "largest_parameter_gradient_group": largest_parameter["parameter"],
            "largest_parameter_gradient_rel_l2": largest_parameter["relative_l2"],
            "largest_update_group": largest_update["parameter"],
            "largest_update_rel_l2": largest_update["update_relative_l2"],
            "global_gradient_rel_l2": global_gradient["relative_l2_error"],
            "global_update_rel_l2": global_update["relative_l2_error"],
            "global_post_weight_rel_l2": global_post["relative_l2_error"],
            "keras_repeatable": all(
                bool(row["exact_equal"]) for row in repeatability_rows
                if row["case_id"] == case_id and row["framework"] == "keras"
            ),
            "pytorch_repeatable": all(
                bool(row["exact_equal"]) for row in repeatability_rows
                if row["case_id"] == case_id and row["framework"] == "pytorch"
            ),
            "trace_equivalent": case_equivalence,
            "nan_inf": False,
        })
        print(
            f"V1 case={case_id} first={case_summaries[-1]['first_nonzero_forward_stage']} "
            f"forward_max={case_summaries[-1]['largest_forward_rel_l2']:.6g} "
            f"repeatable={case_repeatability} equivalent={case_equivalence}", flush=True,
        )
        del plain_k, plain_p, trace1_k, trace1_p, trace2_k, trace2_p
        gc.collect()

    write_csv(RESULTS / "summaries/case_summary.csv", case_summaries)
    write_csv(RESULTS / "preflight/cpu_repeatability_rows.csv", repeatability_rows)
    write_csv(RESULTS / "preflight/cpu_trace_equivalence_rows.csv", equivalence_rows)

    repeatability_valid = all(bool(row["exact_equal"]) for row in repeatability_rows)
    equivalence_valid = all(bool(row["exact_equal"]) for row in equivalence_rows)
    repeatability_report = {
        "validation": "V1_cpu_only", "cpu_repeatability": "VALID" if repeatability_valid else "INVALID",
        "all_exact": repeatability_valid, "comparisons": len(repeatability_rows),
        "cases": len(case_summaries), "config_sha256": config_hash(),
        "new_full_training_executed": False,
    }
    equivalence_report = {
        "validation": "V1_cpu_only", "cpu_trace_equivalence": "VALID" if equivalence_valid else "INVALID",
        "all_exact": equivalence_valid, "comparisons": len(equivalence_rows),
        "metrics": ["logits", "loss", "global_gradient", "global_update", "post_weight", "post_bn_running_mean", "post_bn_running_variance"],
        "cases": len(case_summaries), "config_sha256": config_hash(),
        "new_full_training_executed": False,
    }
    write_json(RESULTS / "preflight/cpu_repeatability.json", repeatability_report)
    write_json(RESULTS / "preflight/cpu_trace_equivalence.json", equivalence_report)

    gpu_rows = {row["case_id"]: row for row in _read_csv(EXP08_RESULTS / "summaries/case_summary.csv")}
    cpu_vs_gpu = []
    for cpu in case_summaries:
        gpu = gpu_rows[cpu["case_id"]]
        cpu_vs_gpu.append({
            "case_id": cpu["case_id"], "seed": cpu["seed"], "checkpoint": cpu["checkpoint"], "anchor": cpu["anchor"],
            "gpu_first_nonzero_forward_stage": gpu["first_nonzero_forward_stage"],
            "cpu_first_nonzero_forward_stage": cpu["first_nonzero_forward_stage"],
            "gpu_largest_forward_stage": gpu["largest_forward_rel_l2_stage"],
            "cpu_largest_forward_stage": cpu["largest_forward_stage"],
            "gpu_largest_forward_rel_l2": gpu["largest_forward_rel_l2"],
            "cpu_largest_forward_rel_l2": cpu["largest_forward_rel_l2"],
            "cpu_to_gpu_forward_rel_l2_ratio": (
                float(cpu["largest_forward_rel_l2"]) / float(gpu["largest_forward_rel_l2"])
                if float(gpu["largest_forward_rel_l2"]) else ""
            ),
            "gpu_largest_backward_stage": gpu["largest_activation_gradient_stage"],
            "cpu_largest_backward_stage": cpu["largest_backward_stage"],
            "gpu_largest_parameter_gradient_group": gpu["largest_parameter_gradient_group"],
            "cpu_largest_parameter_gradient_group": cpu["largest_parameter_gradient_group"],
            "gpu_largest_update_group": gpu["largest_update_group"],
            "cpu_largest_update_group": cpu["largest_update_group"],
            "gpu_global_gradient_rel_l2": gpu["global_gradient_rel_l2"],
            "cpu_global_gradient_rel_l2": cpu["global_gradient_rel_l2"],
            "gpu_global_update_rel_l2": gpu["global_update_rel_l2"],
            "cpu_global_update_rel_l2": cpu["global_update_rel_l2"],
            "gpu_post_weight_rel_l2": gpu["global_post_weight_rel_l2"],
            "cpu_post_weight_rel_l2": cpu["global_post_weight_rel_l2"],
        })
    write_csv(RESULTS / "summaries/cpu_vs_gpu_summary.csv", cpu_vs_gpu)

    selected_complete = len(case_summaries) == 9
    nan_inf = any(bool(row["nan_inf"]) for row in case_summaries)
    overall_valid = repeatability_valid and equivalence_valid and selected_complete and not nan_inf
    diagnostic = {
        "study": "External Validation Study", "validation": "V1 - CPU-only Execution Validation",
        "status": "VALID" if overall_valid else "INVALID",
        "cpu_only_device_check": device["cpu_only_device_check"],
        "v1_resync_precheck": preflight["v1_resync_precheck"],
        "cpu_repeatability": repeatability_report["cpu_repeatability"],
        "cpu_trace_equivalence": equivalence_report["cpu_trace_equivalence"],
        "v1_selected_cases_complete": selected_complete,
        "selected_case_count": len(case_summaries), "nan_inf_found": nan_inf,
        "new_full_training_executed": False, "v2_v3_v4_executed": False,
        "config_sha256": config_hash(), "cases": case_summaries,
    }
    write_json(RESULTS / "summaries/diagnostic_summary.json", diagnostic)
    print(f"CPU_REPEATABILITY = {repeatability_report['cpu_repeatability']}")
    print(f"CPU_TRACE_EQUIVALENCE = {equivalence_report['cpu_trace_equivalence']}")
    print(f"V1_SELECTED_CASES_COMPLETE = {str(selected_complete).upper()}")
    print(f"NAN_INF_FOUND = {str(nan_inf).upper()}")
    print(f"V1_STATUS = {'Completed / VALID' if overall_valid else 'INVALID'}")
    print("NEW_FULL_TRAINING_EXECUTED = FALSE")
    return diagnostic


if __name__ == "__main__":
    run()
