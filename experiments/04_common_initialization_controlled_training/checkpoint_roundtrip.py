"""Read-only canonical checkpoint load/integrity and fresh-model round-trip audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_checkpoint import verify_checkpoint
from common.controlled_config import RESULTS, SEEDS, config_hash
from common.controlled_initialization import (
    build_keras_model, build_torch_model, export_keras_weights, export_torch_weights,
    load_keras_weights, load_torch_weights,
)
from common.trace_utils import state_hash, write_json


def validate() -> dict[str, object]:
    import tensorflow as tf

    details = []
    for seed in SEEDS:
        for step in (0, 1):
            folder = RESULTS / "early_steps" / "checkpoints" / f"seed{seed}" / f"step{step:03d}"
            for framework in ("keras", "pytorch"):
                model_path = folder / f"{framework}_state.npz"
                optimizer_path = folder / f"{framework}_optimizer.npz"
                metadata_path = folder / f"{framework}_metadata.json"
                model_state, optimizer_state = verify_checkpoint(model_path, optimizer_path, metadata_path)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata["optimizer_step"] != step or metadata["config_sha256"] != config_hash():
                    raise RuntimeError(f"Checkpoint metadata mismatch: {metadata_path}")
                if framework == "keras":
                    tf.keras.backend.clear_session()
                    model = build_keras_model()
                    load_keras_weights(model, model_state)
                    restored = export_keras_weights(model)
                else:
                    model = build_torch_model()
                    load_torch_weights(model, model_state)
                    restored = export_torch_weights(model)
                exact = set(restored) == set(model_state) and all(
                    np.array_equal(restored[name], model_state[name]) for name in model_state
                )
                if not exact:
                    raise RuntimeError(f"Model checkpoint round-trip failed: {model_path}")
                # Adam m/v are canonical NPZ analysis state, not a native resume file.
                optimizer_exact = state_hash(optimizer_state) == metadata["optimizer_state_sha256"]
                if not optimizer_exact:
                    raise RuntimeError(f"Optimizer state round-trip failed: {optimizer_path}")
                details.append({
                    "seed": seed, "step": step, "framework": framework,
                    "model_exact": exact, "optimizer_state_exact": optimizer_exact,
                    "model_max_abs_diff": 0.0,
                    "model_keys": len(model_state), "optimizer_keys": len(optimizer_state),
                })
    result = {
        "status": "VALID", "checkpoint_roundtrip": "VALID",
        "config_sha256": config_hash(), "details": details,
        "native_resume_tested": False,
        "note": "Canonical model state reloaded into a fresh model; canonical Adam m/v NPZ integrity round-tripped. Native optimizer resume is not implemented.",
    }
    write_json(RESULTS / "preflight" / "checkpoint_roundtrip.json", result)
    print("CHECKPOINT_ROUNDTRIP = VALID")
    return result


if __name__ == "__main__":
    validate()
