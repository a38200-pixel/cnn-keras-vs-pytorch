"""Phase-2 artifact schemas and common evaluation arithmetic."""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

from common.controlled_config import ALLOW_OVERWRITE, EXPERIMENT, PHASE, RESULTS, config_hash
from common.controlled_checkpoint import verify_checkpoint

HISTORY_FIELDS = [
    "framework", "experiment", "phase", "seed", "epoch", "train_loss",
    "train_accuracy", "val_loss", "val_accuracy", "learning_rate",
    "num_optimizer_steps", "elapsed_epoch_sec",
    "config_sha256",
]

RESULT_FIELDS = [
    "experiment", "phase", "framework", "seed", "epochs_trained",
    "optimizer_steps", "final_test_accuracy", "final_macro_f1", "final_test_loss",
    "best_val_epoch", "best_val_loss", "best_test_accuracy", "best_macro_f1",
    "best_test_loss", "training_time_seconds", "python_version",
    "framework_version", "device", "architecture",
    "config_sha256",
]


def ensure_seed_available(framework: str, seed: int) -> None:
    """Never silently resume or overwrite partial/full runs."""
    if ALLOW_OVERWRITE:
        return
    targets = [
        RESULTS / "history" / f"{framework}_seed{seed}_history.csv",
        RESULTS / "checkpoints" / f"seed{seed}" / framework,
    ]
    result_path = RESULTS / f"{framework}_controlled_results.csv"
    if result_path.exists():
        with result_path.open(encoding="utf-8") as handle:
            if any(int(row["seed"]) == seed for row in csv.DictReader(handle)):
                targets.append(result_path)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Seed artifacts exist; automatic resume/overwrite disabled: {existing}")


def write_history(framework: str, seed: int, rows: list[dict[str, object]]) -> Path:
    path = RESULTS / "history" / f"{framework}_seed{seed}_history.csv"
    if path.exists() and not ALLOW_OVERWRITE:
        raise FileExistsError(f"History exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_result(framework: str, seed: int, row: dict[str, object]) -> Path:
    path = RESULTS / f"{framework}_controlled_results.csv"
    row = dict(row)
    row.update({
        "experiment": EXPERIMENT, "phase": PHASE, "framework": framework,
        "seed": seed, "python_version": sys.version.split()[0],
        "architecture": platform.machine(),
        "config_sha256": config_hash(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if any(int(item["seed"]) == seed for item in existing) and not ALLOW_OVERWRITE:
            raise FileExistsError(f"Controlled result exists: {path} seed={seed}")
        existing = [item for item in existing if int(item["seed"]) != seed]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow({field: row[field] for field in RESULT_FIELDS})
    return path


def require_valid_preflight() -> dict[str, object]:
    path = RESULTS / "preflight" / "preflight_extended_summary.json"
    if not path.exists():
        raise RuntimeError("Run preflight_validate.py before full training.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("controlled_precheck") != "VALID" or report.get("extended_diagnostic_status") != "VALID" or report.get("checkpoint_roundtrip") != "VALID":
        raise RuntimeError(f"Controlled preflight is not VALID: {report.get('errors')}")
    if report.get("full_training_ready") is not True:
        raise RuntimeError("Extended diagnostic/checkpoint gates are not VALID.")
    if report.get("config_sha256") != config_hash():
        raise RuntimeError("Controlled config hash changed since preflight.")
    legacy = json.loads((RESULTS / "preflight" / "preflight_summary.json").read_text(encoding="utf-8"))
    if legacy.get("controlled_precheck") != "VALID" or not all(legacy.get("checks", {}).values()):
        raise RuntimeError("Original controlled preflight is no longer VALID.")
    roundtrip = json.loads((RESULTS / "preflight" / "checkpoint_roundtrip.json").read_text(encoding="utf-8"))
    if roundtrip.get("checkpoint_roundtrip") != "VALID" or roundtrip.get("config_sha256") != config_hash():
        raise RuntimeError("Checkpoint round-trip status/config mismatch.")
    for seed in (42, 123, 2026):
        for kind in ("gradient_distribution", "update_distribution", "adam_state_trace", "reference_adam_trace"):
            if not (RESULTS / "first_step" / f"seed{seed}_{kind}.csv").exists():
                raise RuntimeError(f"Missing extended first-step trace: seed={seed} {kind}")
        if not (RESULTS / "early_steps" / f"seed{seed}_step_summary.csv").exists():
            raise RuntimeError(f"Missing early-step trace: seed={seed}")
        for step in (0, 1):
            folder = RESULTS / "early_steps" / "checkpoints" / f"seed{seed}" / f"step{step:03d}"
            for framework in ("keras", "pytorch"):
                verify_checkpoint(folder / f"{framework}_state.npz",
                                  folder / f"{framework}_optimizer.npz",
                                  folder / f"{framework}_metadata.json")
    return report
