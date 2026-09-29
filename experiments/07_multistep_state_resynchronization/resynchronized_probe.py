"""One-step probes after exact full-state synchronization from both anchors."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.trace_utils import write_csv
from experiment_config import CHECKPOINTS, RESULTS, SEEDS, anchors
from probe_utils import ProbeRunner, fixed_probe_batch, load_checkpoint, require_preflight


def run() -> None:
    require_preflight()
    images, labels, _ = fixed_probe_batch()
    runner = ProbeRunner(images, labels)
    for seed in SEEDS:
        rows = []
        for checkpoint in CHECKPOINTS:
            for anchor in anchors(checkpoint):
                source_framework = "keras" if anchor == "shared" else anchor
                state, optimizer, step, _ = load_checkpoint(seed, source_framework, checkpoint)
                exact = runner.validate_resync(state, optimizer, step)
                if not exact["all_exact"]:
                    raise RuntimeError(f"RESYNC_PRECHECK invalid at {seed}/{checkpoint}/{anchor}")
                metrics = runner.run_pair(state, optimizer, step, state, optimizer, step)
                if any(float(metrics[name]) != 0.0 for name in (
                    "before_weight_relative_l2", "before_adam_m_relative_l2",
                    "before_adam_v_relative_l2", "before_bn_mean_relative_l2",
                    "before_bn_variance_relative_l2",
                )):
                    raise RuntimeError(f"Non-zero synchronized pre-state: {seed}/{checkpoint}/{anchor}")
                row = {
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "anchor": anchor,
                    "source_framework": source_framework,
                    "probe": "full_state_resynchronized",
                    **metrics,
                }
                rows.append(row)
                print(
                    f"resync seed={seed} checkpoint={checkpoint} anchor={anchor} "
                    f"grad={metrics['gradient_relative_l2']:.6g} "
                    f"update={metrics['update_relative_l2']:.6g} "
                    f"post_weight={metrics['post_weight_relative_l2']:.6g}",
                    flush=True,
                )
        write_csv(RESULTS / "resynchronized" / f"seed{seed}_resync_probe.csv", rows)
    print("Experiment 07 re-synchronized probes completed; no full training was run.")


if __name__ == "__main__":
    run()
