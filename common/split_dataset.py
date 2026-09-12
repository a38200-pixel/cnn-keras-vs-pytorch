"""Physically arrange emotion_dataset into train/val/test/class directories.

The operation is resumable: it inventories both flat and already moved files,
recomputes the same per-class hash assignment, and moves only misplaced files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from dataset_utils import CLASSES, EXTENSIONS, PROJECT_ROOT

SPLITS = ("train", "val", "test")
SPLIT_SEED = 42


def class_files(root: Path, class_name: str) -> list[Path]:
    candidates = []
    flat = root / class_name
    if flat.is_dir():
        candidates.extend(p for p in flat.iterdir() if p.suffix.lower() in EXTENSIONS)
    for split in SPLITS:
        folder = root / split / class_name
        if folder.is_dir():
            candidates.extend(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    by_name: dict[str, Path] = {}
    for path in candidates:
        if path.name in by_name and by_name[path.name] != path:
            raise RuntimeError(f"Duplicate filename for {class_name}: {path.name}")
        by_name[path.name] = path
    return list(by_name.values())


def target_map(root: Path) -> dict[Path, Path]:
    moves: dict[Path, Path] = {}
    for class_name in CLASSES:
        paths = class_files(root, class_name)
        paths.sort(key=lambda p: hashlib.sha256(f"{SPLIT_SEED}:{p.name}".encode()).hexdigest())
        n_train, n_val = int(len(paths) * .70), int(len(paths) * .15)
        groups = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:],
        }
        for split, members in groups.items():
            for source in members:
                moves[source] = root / split / class_name / source.name
    return moves


def main() -> None:
    root = PROJECT_ROOT / "emotion_dataset"
    if not root.is_dir():
        raise FileNotFoundError(root)
    moves = target_map(root)
    if len(moves) != 14648:
        raise RuntimeError(f"Expected 14,648 images before migration, found {len(moves):,}")
    for split in SPLITS:
        for class_name in CLASSES:
            (root / split / class_name).mkdir(parents=True, exist_ok=True)
    moved = 0
    for source, target in moves.items():
        if source == target:
            continue
        if target.exists():
            raise FileExistsError(target)
        source.replace(target)
        moved += 1
    for class_name in CLASSES:
        old = root / class_name
        if old.is_dir():
            old.rmdir()
    print(f"Physical split complete: {moved:,} files moved")


if __name__ == "__main__":
    main()
