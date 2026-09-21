"""Classify exact/nonzero differences without imposing a significance threshold."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.controlled_config import RESULTS, SEEDS, config_hash
from common.trace_utils import write_json


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    base = RESULTS / "first_step"
    for seed in SEEDS:
        status = {"seed": seed, "config_sha256": config_hash(), "significance_claim": False}
        for name, key in (("forward", "layer"), ("gradient", "parameter"), ("update", "parameter")):
            source = base / f"seed{seed}_{name}_distribution.csv"
            status[name] = {
                row[key]: {"status": "EXACT" if float(row["max_abs_diff"]) == 0 else "NONZERO",
                           "max_abs_diff": float(row["max_abs_diff"]),
                           "relative_l2_error": float(row["relative_l2_error"])}
                for row in read(source)
            }
        write_json(base / f"seed{seed}_exactness_status.json", status)
    print("EXACT/NONZERO status summaries created; no significance threshold applied.")


if __name__ == "__main__":
    main()
