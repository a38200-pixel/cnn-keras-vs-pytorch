"""Canonical numerical comparisons and protected diagnostic artifact writes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from common.controlled_config import RELATIVE_L2_EPS, SIGN_EPS, TOP_K_OUTLIERS


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_hash(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def assert_finite(values: dict[str, np.ndarray], context: str) -> None:
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"NaN/Inf in {context}: {name}")


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    if a.shape != b.shape:
        raise RuntimeError(f"Shape mismatch: {a.shape} != {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("NaN/Inf in comparison")
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    delta = x - y
    absolute = np.abs(delta)
    xn, yn, dn = np.linalg.norm(x), np.linalg.norm(y), np.linalg.norm(delta)
    sign_x = np.where(np.abs(x) <= SIGN_EPS, 0, np.sign(x))
    sign_y = np.where(np.abs(y) <= SIGN_EPS, 0, np.sign(y))
    result: dict[str, object] = {
        "shape": json.dumps(a.shape), "num_elements": x.size,
        "keras_mean": float(x.mean()), "pytorch_mean": float(y.mean()),
        "keras_std": float(x.std()), "pytorch_std": float(y.std()),
        "keras_norm": float(xn), "pytorch_norm": float(yn),
        "norm_ratio": float(yn / max(xn, RELATIVE_L2_EPS)),
        "mean_abs_diff": float(absolute.mean()),
        "median_abs_diff": float(np.median(absolute)),
        "max_abs_diff": float(absolute.max()),
        "mse": float(np.mean(delta * delta)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "l2_norm_difference": float(dn),
        "relative_l2_error": float(dn / max(xn, yn, RELATIVE_L2_EPS)),
        "cosine_similarity": float(np.dot(x, y) / (xn * yn)) if xn and yn else (1.0 if np.array_equal(x, y) else 0.0),
        "sign_mismatch_ratio": float(np.mean(sign_x != sign_y)),
        "nonzero_difference_ratio": float(np.mean(absolute > 0)),
    }
    for percentile in (50, 90, 95, 99, 99.9):
        label = "99_9" if percentile == 99.9 else str(int(percentile))
        result[f"p{label}_abs_diff"] = float(np.percentile(absolute, percentile))
    return result


def outliers(a: np.ndarray, b: np.ndarray, *, seed: int, parameter: str, top_k: int = TOP_K_OUTLIERS) -> list[dict[str, object]]:
    absolute = np.abs(a.astype(np.float64) - b.astype(np.float64)).ravel()
    ranked = np.argsort(-absolute, kind="stable")[: min(top_k, absolute.size)]
    return [{
        "seed": seed, "parameter": parameter, "rank": rank + 1,
        "flat_index": int(index),
        "canonical_index": json.dumps(tuple(int(i) for i in np.unravel_index(index, a.shape))),
        "keras_value": float(a.ravel()[index]), "pytorch_value": float(b.ravel()[index]),
        "absolute_difference": float(absolute[index]),
    } for rank, index in enumerate(ranked)]


def _protect(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return
        raise FileExistsError(f"Existing artifact differs; refusing overwrite: {path}")
    path.write_bytes(data)


def write_json(path: Path, value: object) -> None:
    _protect(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _protect(path, buffer.getvalue().encode("utf-8"))
