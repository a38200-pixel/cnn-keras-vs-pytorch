"""Experiment 01: PyTorch baseline with deterministic input preprocessing aligned."""

# ============================================================
# 1. Imports
# ============================================================
from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.dataset_utils import (
    CLASSES,
    build_split_manifest,
    find_dataset_root,
    is_physically_split,
    split_counts,
)
from common.environment_utils import print_torch_environment
from common.history_utils import save_history
from common.input_alignment import compare_framework_inputs, load_aligned_rgb
from common.result_utils import save_result
from common.seed_utils import seed_pytorch


# ============================================================
# 2. Experiment Configuration
# ============================================================
EXPERIMENT = "01_input_tensor_alignment"
ALIGNMENT = "input_tensor"
SEEDS = [42, 123, 2026]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS = 128, 32, 8, 30
LEARNING_RATE, EARLY_PATIENCE, LR_PATIENCE, MIN_DELTA = 0.001, 7, 4, 1e-4
RUN_TRAINING = False
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS = RESULTS_DIR / "pytorch_results.csv"
HISTORY_DIR = RESULTS_DIR / "history"


# ============================================================
# 3. Random Seed
# ============================================================
def set_seed(seed: int) -> None:
    seed_pytorch(seed)


# ============================================================
# 4. Dataset / Input Preprocessing / Baseline Augmentation
# ============================================================
def load_deterministic_input(path: str):
    """Shared HWC float32 [0,1] input, changed to CHW only by transpose."""
    import torch

    array = load_aligned_rgb(path, IMG_SIZE)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def make_dataset(rows, seed: int, training: bool):
    import torch
    from torchvision import transforms

    # Keep the PyTorch Baseline random operators and their framework-specific
    # RNG/interpolation/fill behavior. Experiment 03 aligns augmentation.
    baseline_augmentation = transforms.Compose([
        # The aligned values are exact uint8/255 values. This round-trip restores
        # the Baseline PIL augmentation path without applying a second scaling.
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    class EmotionDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            tensor = load_deterministic_input(str(row["path"]))
            if training:
                tensor = baseline_augmentation(tensor)
            return tensor, int(row["label"])

    return EmotionDataset()


def make_loader(dataset, seed: int, training: bool, epoch: int = 0):
    import torch

    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=training,
        generator=generator if training else None,
        num_workers=0,
    )


# ============================================================
# 5. CNN Model
# ============================================================
def build_model(seed: int):
    from torch import nn

    class FinalRGBModel(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_channels = 3
            for channels in (32, 64, 128, 256):
                blocks.extend([
                    nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels, eps=1e-3, momentum=.01),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ])
                in_channels = channels
            self.features = nn.Sequential(*blocks)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc128 = nn.Linear(256, 128)
            self.relu = nn.ReLU()
            self.logits = nn.Linear(128, NUM_CLASSES)

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x).flatten(1)
            return self.logits(self.relu(self.fc128(x)))

    return FinalRGBModel()


# ============================================================
# 6. Optimizer / Loss
# ============================================================
def make_optimizer_and_loss(model):
    import torch

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(.9, .999),
        eps=1e-8, weight_decay=0, amsgrad=False,
    )
    return optimizer, torch.nn.CrossEntropyLoss()


# ============================================================
# 7. Training / History Saving / Evaluation
# ============================================================
def print_training_header(seed: int, device) -> None:
    print("=" * 60)
    print("Experiment 01 - Input Tensor Alignment")
    print("Framework : PyTorch")
    print(f"Seed      : {seed}")
    print(f"Device    : {device}")
    print(f"Epochs    : max {EPOCHS}")
    print(f"Batch     : {BATCH_SIZE}")
    print("=" * 60)


def result_contains_seed(seed: int) -> bool:
    if not RESULTS.exists():
        return False
    with RESULTS.open(encoding="utf-8") as handle:
        return any(int(row["seed"]) == seed for row in csv.DictReader(handle))


def seed_is_complete(seed: int) -> bool:
    return (
        result_contains_seed(seed)
        and (RESULTS_DIR / f"pytorch_seed{seed}.pt").exists()
        and (HISTORY_DIR / f"pytorch_seed{seed}_history.csv").exists()
    )


def train_one_seed(seed: int) -> None:
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm.auto import tqdm

    set_seed(seed)
    device = print_torch_environment(EXPERIMENT, seed)
    print_training_header(seed, device)
    manifest = build_split_manifest()
    train_data, val_data, test_data = (
        make_dataset(manifest[split], seed, split == "train")
        for split in ("train", "val", "test")
    )
    train_loader = make_loader(train_data, seed, True)
    val_loader = make_loader(val_data, seed, False)
    test_loader = make_loader(test_data, seed, False)
    model = build_model(seed).to(device)
    optimizer, criterion = make_optimizer_and_loss(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=LR_PATIENCE,
        threshold=MIN_DELTA, threshold_mode="abs", min_lr=1e-6,
    )

    history_rows: list[dict[str, object]] = []
    best_loss, best_epoch, bad_epochs, best_state = float("inf"), 0, 0, None
    training_start = time.perf_counter()
    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = correct = total = 0
        progress = tqdm(
            train_loader,
            desc=f"Seed {seed} | Epoch {epoch + 1:02d}/{EPOCHS} | Training",
            unit="batch",
            dynamic_ncols=True,
        )
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
            progress.set_postfix(
                loss=f"{loss_sum / total:.4f}",
                acc=f"{correct / total:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6g}",
            )
        train_loss, train_acc = loss_sum / total, correct / total

        model.eval()
        loss_sum = correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_sum += loss.item() * len(labels)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)
        val_loss, val_acc = loss_sum / total, correct / total

        improved = val_loss < best_loss - MIN_DELTA
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        history_rows.append({
            "framework": "pytorch",
            "experiment": EXPERIMENT,
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": learning_rate,
            "elapsed_epoch_sec": time.perf_counter() - epoch_start,
        })
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | "
            f"LR: {learning_rate:.8f} | bad_epochs: {bad_epochs}",
            flush=True,
        )
        if bad_epochs >= EARLY_PATIENCE:
            print(f"EarlyStopping at epoch {epoch + 1}; restoring epoch {best_epoch + 1}.")
            break

    elapsed = time.perf_counter() - training_start
    history_path = HISTORY_DIR / f"pytorch_seed{seed}_history.csv"
    save_history(history_path, history_rows)
    print(f"Saved history immediately: {history_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = RESULTS_DIR / f"pytorch_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)
    model.eval()
    y_true, y_pred, loss_sum, total = [], [], 0.0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            loss_sum += criterion(outputs, labels.to(device)).item() * len(labels)
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            total += len(labels)
    save_result(RESULTS, {
        "experiment": EXPERIMENT, "framework": "pytorch", "seed": seed,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "test_loss": loss_sum / total, "best_epoch": best_epoch + 1,
        "epochs_trained": len(history_rows),
        "train_accuracy": history_rows[best_epoch]["train_accuracy"],
        "validation_accuracy": history_rows[best_epoch]["val_accuracy"],
        "train_loss": history_rows[best_epoch]["train_loss"],
        "validation_loss": history_rows[best_epoch]["val_loss"],
        "train_val_gap": history_rows[best_epoch]["train_accuracy"] - history_rows[best_epoch]["val_accuracy"],
        "training_time_seconds": elapsed,
    })
    print(f"Completed and saved Seed {seed}: {model_path}")


def run_3seed_experiment() -> None:
    for seed in SEEDS:
        if seed_is_complete(seed):
            print(f"[SKIP] Seed {seed} already has result, model, and history files.")
            continue
        if any((
            (RESULTS_DIR / f"pytorch_seed{seed}.pt").exists(),
            (HISTORY_DIR / f"pytorch_seed{seed}_history.csv").exists(),
            result_contains_seed(seed),
        )):
            print(f"[WARNING] Seed {seed} has partial artifacts and will be rerun.")
        train_one_seed(seed)


# ============================================================
# 8. Numerical Equality / Model Sanity Check
# ============================================================
def run_input_alignment_check(paths: list[str]) -> dict[str, object]:
    metrics = compare_framework_inputs(paths, IMG_SIZE)
    print("Input Alignment Check")
    print(f"Samples : {metrics['samples']}")
    print(f"Keras Shape  : {metrics['keras_shape']}")
    print(f"PyTorch Shape: {metrics['pytorch_shape']}")
    print(f"Dtype   : {metrics['dtype']}")
    print(f"Range   : [{metrics['minimum']:.8f}, {metrics['maximum']:.8f}]")
    print(f"MAE     : {metrics['mae']:.12g}")
    print(f"Max Abs : {metrics['max_abs']:.12g}")
    print(f"MSE     : {metrics['mse']:.12g}")
    print(f"All Close: {metrics['allclose']}")
    assert metrics["allclose"], metrics
    print("Input Alignment Check: PASS")
    return metrics


def run_sanity_check() -> None:
    import torch

    set_seed(SEEDS[0])
    device = print_torch_environment(EXPERIMENT, SEEDS[0])
    manifest = build_split_manifest()
    counts = split_counts(manifest)
    assert is_physically_split()
    assert counts == {"train": 10251, "val": 2194, "test": 2203}
    assert all({str(row["class_name"]) for row in rows} == set(CLASSES) for rows in manifest.values())

    fixed_paths = [str(row["path"]) for row in manifest["train"][:8]]
    run_input_alignment_check(fixed_paths)
    data = make_dataset(manifest["train"][:BATCH_SIZE], SEEDS[0], True)
    images, labels = next(iter(make_loader(data, SEEDS[0], False)))
    images, labels = images.to(device), labels.to(device)
    model = build_model(SEEDS[0]).to(device).eval()
    with torch.no_grad():
        output = model(images)
    assert tuple(images.shape) == (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    assert tuple(labels.shape) == (BATCH_SIZE,)
    assert tuple(output.shape) == (BATCH_SIZE, NUM_CLASSES) and torch.isfinite(output).all()
    assert next(model.parameters()).device == images.device == labels.device == output.device
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Dataset: {find_dataset_root()} | {counts}")
    print(f"Classes: {dict((name, index) for index, name in enumerate(CLASSES))}")
    print(
        f"Model Output: {tuple(output.shape)} | parameters={parameters:,} | finite=True | "
        f"model/input/output device={output.device}"
    )


# ============================================================
# 9. Entry Point
# ============================================================
def print_experiment_config() -> None:
    print("Experiment: 01 Input Tensor Alignment")
    print("Changed Variable: Input Tensor only")
    print(f"Seeds: {SEEDS}")
    print(f"RUN_TRAINING: {RUN_TRAINING}")


if __name__ == "__main__":
    print_experiment_config()
    run_sanity_check()
    if RUN_TRAINING:
        run_3seed_experiment()
    else:
        print("Training was not started.\nSet RUN_TRAINING = True to start training.")
