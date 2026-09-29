"""One-step probes from the distinct Experiment 06 framework states."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.trace_utils import write_csv
from experiment_config import CHECKPOINTS, RESULTS, SEEDS
from probe_utils import ProbeRunner, fixed_probe_batch, load_checkpoint, require_preflight


def run() -> None:
    require_preflight()
    images, labels, _ = fixed_probe_batch()
    runner = ProbeRunner(images, labels)
    for seed in SEEDS:
        rows = []
        for checkpoint in CHECKPOINTS:
            keras_state, keras_optimizer, keras_step, _ = load_checkpoint(seed, "keras", checkpoint)
            torch_state, torch_optimizer, torch_step, _ = load_checkpoint(seed, "pytorch", checkpoint)
            metrics = runner.run_pair(
                keras_state, keras_optimizer, keras_step,
                torch_state, torch_optimizer, torch_step,
            )
            row = {"seed": seed, "checkpoint": checkpoint, "probe": "free_running", **metrics}
            rows.append(row)
            print(
                f"free seed={seed} checkpoint={checkpoint} "
                f"weight={metrics['before_weight_relative_l2']:.6g}"
                f"->{metrics['post_weight_relative_l2']:.6g} "
                f"grad={metrics['gradient_relative_l2']:.6g}",
                flush=True,
            )
        write_csv(RESULTS / "free_running" / f"seed{seed}_free_running_probe.csv", rows)
    print("Experiment 07 free-running probes completed; no full training was run.")


if __name__ == "__main__":
    run()
