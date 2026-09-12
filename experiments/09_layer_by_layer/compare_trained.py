"""Compare trained aligned-weight models only when both checkpoints exist."""
from __future__ import annotations
import json
from compare_initial import RESULTS, SEEDS, build_pair, compare_models, shared_batch


def main() -> None:
    import tensorflow as tf
    import torch
    batch, _ = shared_batch(); source = RESULTS.parents[1] / "04_initial_weight_alignment" / "results"
    found = False
    for seed in SEEDS:
        keras_path, torch_path = source / f"keras_seed{seed}.keras", source / f"pytorch_seed{seed}.pt"
        if not (keras_path.exists() and torch_path.exists()):
            print(f"Seed {seed}: trained checkpoints not found; skipped."); continue
        keras_model, torch_model = build_pair(seed)
        keras_model.set_weights(tf.keras.models.load_model(keras_path).get_weights())
        # Load portably on CPU; load_state_dict copies into the model's current
        # CUDA/MPS/CPU device selected by build_pair().
        torch_model.load_state_dict(torch.load(torch_path, map_location="cpu", weights_only=True))
        rows = compare_models(keras_model, torch_model, batch, seed, "trained")
        (RESULTS / f"trained_seed{seed}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        found = True
    if not found: print("Experiment results were not found.\nRun the training scripts first.")


if __name__ == "__main__": main()
