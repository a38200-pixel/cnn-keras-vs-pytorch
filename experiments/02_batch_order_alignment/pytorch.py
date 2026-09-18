"""Experiment 02: PyTorch Baseline with only training batch order aligned."""

from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.batch_order import (
    batch_groups,
    initialize_runtime_order_log,
    order_hash,
    prepare_batch_order_artifacts,
    record_runtime_order,
    runtime_order_path,
)
from common.dataset_utils import CLASSES, build_split_manifest, find_dataset_root, is_physically_split, split_counts
from common.environment_utils import print_torch_environment
from common.history_utils import save_history
from common.result_utils import save_result
from common.seed_utils import seed_pytorch


# ============================================================
# 1. Configuration
# ============================================================
EXPERIMENT = "02_batch_order_alignment"
ALIGNMENT = "batch_order"
SEEDS = [42, 123, 2026]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS = 128, 32, 8, 30
LEARNING_RATE, EARLY_PATIENCE, LR_PATIENCE, MIN_DELTA = 0.001, 7, 4, 1e-4
RUN_TRAINING = False
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS = RESULTS_DIR / "pytorch_results.csv"
HISTORY_DIR = RESULTS_DIR / "history"
BATCH_ORDER_DIR = RESULTS_DIR / "batch_order"
RUNTIME_ORDER_DIR = RESULTS_DIR / "runtime_order"


# ============================================================
# 2. Framework Seed (separate from persisted batch-order RNG)
# ============================================================
def set_seed(seed: int) -> None:
    seed_pytorch(seed)


# ============================================================
# 3. Baseline PyTorch Input / Augmentation
# ============================================================
def make_dataset(rows, training: bool):
    import torch
    from PIL import Image
    from torchvision import transforms

    baseline_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    baseline_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    class EmotionDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            with Image.open(str(row["path"])) as source:
                image = source.convert("RGB")
                tensor = (baseline_train if training else baseline_eval)(image)
            return tensor, int(row["label"])

    return EmotionDataset()


def make_ordered_loader(dataset, order: np.ndarray | list[int] | None = None):
    import torch

    class FixedOrderSampler(torch.utils.data.Sampler):
        def __init__(self, indices):
            self.indices = [int(index) for index in indices]

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    if order is None:
        return torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False,
        )
    sampler = FixedOrderSampler(order)
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler,
        num_workers=0, drop_last=False,
    )


# ============================================================
# 4. Baseline PyTorch Model / Optimizer / Loss
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


def make_optimizer_and_loss(model):
    import torch

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(.9, .999),
        eps=1e-8, weight_decay=0, amsgrad=False,
    )
    return optimizer, torch.nn.CrossEntropyLoss()


# ============================================================
# 5. Batch-Order Artifacts / Exact-Match Sanity
# ============================================================
def prepare_orders():
    manifest = build_split_manifest()
    canonical, schedules = prepare_batch_order_artifacts(
        manifest["train"], find_dataset_root(), BATCH_ORDER_DIR, SEEDS, EPOCHS,
    )
    return manifest, canonical, schedules


def run_batch_order_check(canonical, schedules) -> dict[int, str]:
    hashes: dict[int, str] = {}
    first_epoch_orders = []
    dataset = make_dataset(canonical, training=False)
    for seed in SEEDS:
        selected_hashes = {}
        observed_schedule_indices = []
        for expected_index in range(EPOCHS):
            epoch_loader = make_ordered_loader(dataset, schedules[seed][expected_index])
            actual_sampler_order = list(iter(epoch_loader.sampler))
            expected_order = [int(index) for index in schedules[seed][expected_index]]
            assert actual_sampler_order == expected_order
            observed_schedule_indices.append(expected_index)
            if expected_index in {0, 1, EPOCHS - 1}:
                selected_hashes[expected_index + 1] = order_hash(actual_sampler_order)
        assert observed_schedule_indices == list(range(EPOCHS))

        keras_sequence_order = [int(index) for index in schedules[seed][0]]
        pytorch_loader = make_ordered_loader(dataset, schedules[seed][0])
        pytorch_sampler_order = list(iter(pytorch_loader.sampler))
        keras_batches = batch_groups(keras_sequence_order, BATCH_SIZE)
        pytorch_batches = batch_groups(pytorch_sampler_order, BATCH_SIZE)
        exact = keras_batches == pytorch_batches
        keras_hash = order_hash(keras_sequence_order)
        pytorch_hash = order_hash(pytorch_sampler_order)
        assert exact and keras_hash == pytorch_hash
        assert len(pytorch_loader) == 321 and len(pytorch_batches[-1]) == 11
        assert not np.array_equal(schedules[seed][0], schedules[seed][1])
        first_epoch_orders.append(schedules[seed][0])
        hashes[seed] = pytorch_hash
        print("Batch Order Alignment Check")
        print(f"Seed: {seed} | Epoch: 1 | Samples: {len(pytorch_sampler_order)}")
        print(f"First 3 Keras batches : {keras_batches[:3]}")
        print(f"First 3 PyTorch batches: {pytorch_batches[:3]}")
        print(f"Keras order hash : {keras_hash}")
        print(f"PyTorch order hash: {pytorch_hash}")
        print(f"Exact match: {exact}")
        print(f"Expected lifecycle: {list(range(EPOCHS))}")
        print(f"Observed lifecycle: {observed_schedule_indices}")
        print(f"Lifecycle exact match: {observed_schedule_indices == list(range(EPOCHS))}")
        print(f"Epoch 1/2/30 shared hashes: {selected_hashes}")
    assert all(
        not np.array_equal(first_epoch_orders[left], first_epoch_orders[right])
        for left in range(len(SEEDS)) for right in range(left + 1, len(SEEDS))
    )
    print("Batch Order exact-match sanity check: PASS")
    return hashes


# ============================================================
# 6. Training / History / Evaluation
# ============================================================
def result_contains_seed(seed: int) -> bool:
    if not RESULTS.exists():
        return False
    with RESULTS.open(encoding="utf-8") as handle:
        return any(int(row["seed"]) == seed for row in csv.DictReader(handle))


def seed_is_complete(seed: int) -> bool:
    return result_contains_seed(seed) and (RESULTS_DIR / f"pytorch_seed{seed}.pt").exists() and (
        HISTORY_DIR / f"pytorch_seed{seed}_history.csv"
    ).exists() and runtime_order_path(RUNTIME_ORDER_DIR, "pytorch", seed).exists()


def seed_has_partial_artifacts(seed: int) -> bool:
    return result_contains_seed(seed) or (RESULTS_DIR / f"pytorch_seed{seed}.pt").exists() or (
        HISTORY_DIR / f"pytorch_seed{seed}_history.csv"
    ).exists() or runtime_order_path(RUNTIME_ORDER_DIR, "pytorch", seed).exists()


def print_training_header(seed: int, device) -> None:
    print("=" * 60)
    print("Experiment 02 - Batch Order Alignment")
    print("Framework : PyTorch")
    print(f"Seed      : {seed}")
    print(f"Device    : {device}")
    print(f"Batch     : {BATCH_SIZE}")
    print(f"Max Epoch : {EPOCHS}")
    print("=" * 60)


def train_one_seed(seed: int) -> None:
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm.auto import tqdm

    set_seed(seed)
    device = print_torch_environment(EXPERIMENT, seed)
    print_training_header(seed, device)
    manifest, canonical, schedules = prepare_orders()
    initialize_runtime_order_log(RUNTIME_ORDER_DIR, "pytorch", seed)
    train_data = make_dataset(canonical, training=True)
    val_loader = make_ordered_loader(make_dataset(manifest["val"], training=False))
    test_loader = make_ordered_loader(make_dataset(manifest["test"], training=False))
    model = build_model(seed).to(device)
    optimizer, criterion = make_optimizer_and_loss(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=LR_PATIENCE,
        threshold=MIN_DELTA, threshold_mode="abs", min_lr=1e-6,
    )
    history_rows: list[dict[str, object]] = []
    best_loss, best_epoch, bad_epochs, best_state = float("inf"), 0, 0, None
    training_started = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_started = time.perf_counter()
        train_loader = make_ordered_loader(train_data, schedules[seed][epoch])
        actual_sampler_order = list(iter(train_loader.sampler))
        record_runtime_order(
            RUNTIME_ORDER_DIR, "pytorch", seed,
            epoch=epoch + 1, schedule_index=epoch,
            order=actual_sampler_order, canonical=canonical,
            dataset_root=find_dataset_root(), batch_size=BATCH_SIZE,
        )
        model.train()
        loss_sum = correct = total = 0
        progress = tqdm(
            train_loader, desc=f"Seed {seed} | Epoch {epoch + 1:02d}/{EPOCHS} | Training",
            unit="batch", dynamic_ncols=True,
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
                loss=f"{loss_sum / total:.4f}", acc=f"{correct / total:.4f}",
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
            "framework": "pytorch", "experiment": EXPERIMENT, "seed": seed, "epoch": epoch + 1,
            "train_loss": train_loss, "train_accuracy": train_acc,
            "val_loss": val_loss, "val_accuracy": val_acc,
            "learning_rate": learning_rate,
            "elapsed_epoch_sec": time.perf_counter() - epoch_started,
        })
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss:.6f} | "
            f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.6f} | "
            f"Val Acc: {val_acc:.4f} | LR: {learning_rate:.8f} | bad_epochs: {bad_epochs}",
            flush=True,
        )
        if bad_epochs >= EARLY_PATIENCE:
            print(f"EarlyStopping at epoch {epoch + 1}; restoring epoch {best_epoch + 1}.")
            break

    elapsed = time.perf_counter() - training_started
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
        "test_loss": loss_sum / total, "best_epoch": best_epoch + 1, "epochs_trained": len(history_rows),
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
        else:
            if seed_has_partial_artifacts(seed):
                print(f"[WARNING] Seed {seed} has partial artifacts; this run will complete/replace them.")
            train_one_seed(seed)


# ============================================================
# 7. Dataset / Model Sanity and Entry Point
# ============================================================
def run_sanity_check() -> None:
    import torch

    set_seed(SEEDS[0])
    device = print_torch_environment(EXPERIMENT, SEEDS[0])
    manifest, canonical, schedules = prepare_orders()
    counts = split_counts(manifest)
    assert is_physically_split() and counts == {"train": 10251, "val": 2194, "test": 2203}
    assert all({str(row["class_name"]) for row in rows} == set(CLASSES) for rows in manifest.values())
    run_batch_order_check(canonical, schedules)
    loader = make_ordered_loader(make_dataset(canonical[:BATCH_SIZE], training=True))
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    model = build_model(SEEDS[0]).to(device).eval()
    with torch.no_grad():
        output = model(images)
    assert tuple(images.shape) == (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE) and tuple(labels.shape) == (BATCH_SIZE,)
    assert tuple(output.shape) == (BATCH_SIZE, NUM_CLASSES) and torch.isfinite(output).all()
    assert next(model.parameters()).device == images.device == labels.device == output.device
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Dataset: {find_dataset_root()} | {counts}")
    print(f"Classes: {dict((name, index) for index, name in enumerate(CLASSES))}")
    print(f"Model Output: {tuple(output.shape)} | parameters={parameters:,} | finite=True | device={output.device}")


def print_experiment_config() -> None:
    print("Experiment: 02 Batch Order Alignment")
    print("Changed Variable: Training Batch Order only")
    print(f"Seeds: {SEEDS}")
    print(f"RUN_TRAINING: {RUN_TRAINING}")


if __name__ == "__main__":
    print_experiment_config()
    run_sanity_check()
    if RUN_TRAINING:
        run_3seed_experiment()
    else:
        print("Training was not started.\nSet RUN_TRAINING = True to start training.")
