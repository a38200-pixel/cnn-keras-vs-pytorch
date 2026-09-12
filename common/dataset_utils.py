"""Discover and validate the physical class-stratified 70/15/15 dataset.

Flat class folders remain supported so the idempotent migration utility can
recover safely if a move is interrupted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_dataset_root() -> Path:
    for name in ("dataset", "emotion_dataset"):
        root = PROJECT_ROOT / name
        flat = root.is_dir() and all((root / c).is_dir() for c in CLASSES)
        split = root.is_dir() and all(
            (root / part / class_name).is_dir()
            for part in ("train", "val", "test")
            for class_name in CLASSES
        )
        if flat or split:
            return root
    raise FileNotFoundError("Expected dataset/ or emotion_dataset/ with eight class folders")


def _files(root: Path, class_name: str) -> list[Path]:
    return sorted(p for p in (root / class_name).iterdir() if p.suffix.lower() in EXTENSIONS)


def dataset_inventory(root: Path | None = None) -> dict[str, int]:
    root = root or find_dataset_root()
    if is_physically_split(root):
        return {
            name: sum(len(_files(root / split, name)) for split in ("train", "val", "test"))
            for name in CLASSES
        }
    return {name: len(_files(root, name)) for name in CLASSES}


def is_physically_split(root: Path | None = None) -> bool:
    root = root or find_dataset_root()
    return all(
        (root / split / class_name).is_dir()
        for split in ("train", "val", "test")
        for class_name in CLASSES
    )


def build_split_manifest(split_seed: int = 42) -> dict[str, list[dict[str, object]]]:
    """Read a physical split, or deterministically derive one from flat folders."""
    root = find_dataset_root()
    result: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    if is_physically_split(root):
        for split in ("train", "val", "test"):
            for label, class_name in enumerate(CLASSES):
                result[split].extend(
                    {"path": str(path), "label": label, "class_name": class_name}
                    for path in _files(root / split, class_name)
                )
        return result
    for label, class_name in enumerate(CLASSES):
        paths = _files(root, class_name)
        paths.sort(key=lambda p: hashlib.sha256(f"{split_seed}:{p.name}".encode()).hexdigest())
        n_train = int(len(paths) * 0.70)
        n_val = int(len(paths) * 0.15)
        groups = {"train": paths[:n_train], "val": paths[n_train:n_train+n_val], "test": paths[n_train+n_val:]}
        for split, members in groups.items():
            result[split].extend({"path": str(p), "label": label, "class_name": class_name} for p in members)
    return result


def save_manifest(path: Path, split_seed: int = 42) -> dict[str, list[dict[str, object]]]:
    manifest = build_split_manifest(split_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def split_counts(manifest: dict[str, list[dict[str, object]]]) -> dict[str, int]:
    return {split: len(rows) for split, rows in manifest.items()}
