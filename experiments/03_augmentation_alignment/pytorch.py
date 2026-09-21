"""Experiment 03: PyTorch Baseline with only augmentation parameters aligned."""

from __future__ import annotations

import copy
import csv
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.augmentation_runtime import (
    AugmentationRuntimeRecorder,
    apply_torchvision_augmentation,
    augmentation_parameters,
    expected_records,
    parameter_hash,
    runtime_summary_path,
    runtime_values_path,
    sample_identity,
    save_json,
)
from common.dataset_utils import (
    CLASSES, build_split_manifest, find_dataset_root, is_physically_split, split_counts,
)
from common.environment_utils import print_torch_environment, select_torch_device
from common.history_utils import save_history
from common.result_utils import save_result
from common.seed_utils import seed_pytorch


EXPERIMENT = "03_augmentation_alignment"
ALIGNMENT = "augmentation"
SEEDS = [42, 123, 2026]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS = 128, 32, 8, 30
LEARNING_RATE, EARLY_PATIENCE, LR_PATIENCE, MIN_DELTA = 0.001, 7, 4, 1e-4
RUN_TRAINING = False
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS = RESULTS_DIR / "pytorch_results.csv"
HISTORY_DIR = RESULTS_DIR / "history"
RUNTIME_AUG_DIR = RESULTS_DIR / "runtime_augmentation"
SANITY_DIR = RESULTS_DIR / "sanity"


def set_seed(seed: int) -> None:
    seed_pytorch(seed)


def configuration_fingerprint() -> dict[str, object]:
    return {
        "branch_source": "00_baseline_final_cnn_3seed",
        "changed_variable": "augmentation_only",
        "model": "Conv(32,64,128,256)-BN-ReLU-MaxPool-GAP-Linear128-Linear8",
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "optimizer": "Adam(beta=(0.9,0.999),eps=1e-8,weight_decay=0,amsgrad=False)",
        "learning_rate": LEARNING_RATE,
        "loss": "CrossEntropyLoss(logits)",
        "batchnorm": "eps=1e-3,momentum=0.01",
        "early_stopping": f"val_loss,patience={EARLY_PATIENCE},min_delta={MIN_DELTA},restore_best",
        "lr_scheduler": f"val_loss,factor=0.5,patience={LR_PATIENCE},min_lr=1e-6",
        "input": "PIL RGB -> torchvision Resize bilinear -> ToTensor(/255)",
        "shuffle": "Baseline PyTorch DataLoader shuffle=True with seeded Generator",
        "augmentation": "strict sample-ID; flip p=0.5 then Uniform[-5,+5] degree rotation; bilinear; constant 0",
        "validation_test_augmentation": False,
    }


def make_dataset(
    rows,
    seed: int,
    training: bool,
    recorder: AugmentationRuntimeRecorder | None = None,
):
    import torch
    from PIL import Image
    from torchvision import transforms

    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    to_tensor = transforms.ToTensor()
    dataset_root = find_dataset_root()

    class EmotionDataset(torch.utils.data.Dataset):
        epoch = 0

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            path = str(row["path"])
            image = Image.open(path).convert("RGB")
            image = resize(image)
            if training:
                identity = sample_identity(path, dataset_root)
                flip, angle = augmentation_parameters(identity, seed, self.epoch)
                image = apply_torchvision_augmentation(image, flip, angle)
                if recorder is not None:
                    recorder.record(identity, flip, angle)
            return to_tensor(image), int(row["label"])

    return EmotionDataset()


def make_loader(dataset, seed: int, training: bool):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=training,
        generator=generator if training else None, num_workers=0,
    )


def build_model(seed: int):
    import torch
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
                    nn.ReLU(), nn.MaxPool2d(2),
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


def result_contains_seed(seed: int) -> bool:
    if not RESULTS.exists():
        return False
    with RESULTS.open(encoding="utf-8") as handle:
        return any(int(row["seed"]) == seed for row in csv.DictReader(handle))


def seed_is_complete(seed: int) -> bool:
    return all((
        result_contains_seed(seed),
        (RESULTS_DIR / f"pytorch_seed{seed}.pt").exists(),
        (HISTORY_DIR / f"pytorch_seed{seed}_history.csv").exists(),
        runtime_summary_path(RUNTIME_AUG_DIR, "pytorch", seed).exists(),
        runtime_values_path(RUNTIME_AUG_DIR, "pytorch", seed).exists(),
    ))


def seed_has_partial_artifacts(seed: int) -> bool:
    return any((
        result_contains_seed(seed),
        (RESULTS_DIR / f"pytorch_seed{seed}.pt").exists(),
        (HISTORY_DIR / f"pytorch_seed{seed}_history.csv").exists(),
        runtime_summary_path(RUNTIME_AUG_DIR, "pytorch", seed).exists(),
        runtime_values_path(RUNTIME_AUG_DIR, "pytorch", seed).exists(),
    ))


def print_training_header(seed: int, device) -> None:
    print("=" * 60)
    print("Experiment 03 - Augmentation Alignment")
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
    manifest = build_split_manifest()
    identities = [sample_identity(row["path"], find_dataset_root()) for row in manifest["train"]]
    recorder = AugmentationRuntimeRecorder("pytorch", seed, identities, RUNTIME_AUG_DIR)
    train_data = make_dataset(manifest["train"], seed, True, recorder)
    val_loader = make_loader(make_dataset(manifest["val"], seed, False), seed, False)
    test_loader = make_loader(make_dataset(manifest["test"], seed, False), seed, False)
    # Create once, as in Baseline: native RandomSampler generator state advances each epoch.
    train_loader = make_loader(train_data, seed, True)
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
        train_data.epoch = epoch
        recorder.begin_epoch(epoch)
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
        recorder.finalize_epoch()
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
    if len(recorder.completed) != len(history_rows):
        raise RuntimeError(
            f"Runtime/history epoch mismatch: runtime={len(recorder.completed)} history={len(history_rows)}"
        )
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
            print(f"[SKIP] Seed {seed} already has complete result/model/history/runtime artifacts.")
        else:
            if seed_has_partial_artifacts(seed):
                print(f"[WARNING] Seed {seed} has partial artifacts; this run will replace/complete them.")
            train_one_seed(seed)


def run_sanity_check() -> None:
    import torch

    set_seed(SEEDS[0])
    device = print_torch_environment(EXPERIMENT, SEEDS[0])
    manifest = build_split_manifest()
    counts = split_counts(manifest)
    assert is_physically_split() and counts == {"train": 10251, "val": 2194, "test": 2203}
    assert all({str(row["class_name"]) for row in rows} == set(CLASSES) for rows in manifest.values())
    dataset_root = find_dataset_root()
    dry_rows = manifest["train"][:BATCH_SIZE]
    dry_identities = [sample_identity(row["path"], dataset_root) for row in dry_rows]
    dry_hashes = {}
    with tempfile.TemporaryDirectory(prefix="exp03-pytorch-sanity-") as temp_dir:
        for seed in SEEDS:
            recorder = AugmentationRuntimeRecorder("pytorch", seed, dry_identities, Path(temp_dir))
            data = make_dataset(dry_rows, seed, True, recorder)
            data.epoch = 0
            recorder.begin_epoch(0)
            images, labels = next(iter(make_loader(data, seed, False)))
            recorder.finalize_epoch()
            observed = recorder.completed[0][1]
            expected = expected_records(dry_identities, seed, 0)
            assert observed == expected
            dry_hashes[str(seed)] = parameter_hash(observed)

    images, labels = images.to(device), labels.to(device)
    model = build_model(SEEDS[0]).to(device).eval()
    with torch.no_grad():
        output = model(images)
    assert tuple(images.shape) == (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    assert tuple(labels.shape) == (BATCH_SIZE,)
    assert tuple(output.shape) == (BATCH_SIZE, NUM_CLASSES) and torch.isfinite(output).all()
    assert next(model.parameters()).device == images.device == labels.device == output.device
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report = {
        "status": "PASS", "mode": "strict_sample_id", "dataset_counts": counts,
        "seed_dry_runtime_hashes": dry_hashes,
        "epoch_schedule_mapping": list(range(EPOCHS)),
        "runtime_logging_path_tested": True,
        "configuration_fingerprint": configuration_fingerprint(),
        "model_output_shape": list(output.shape), "model_parameters": parameters,
        "device": str(device), "mps_available": bool(torch.backends.mps.is_available()),
        "run_training": RUN_TRAINING,
    }
    save_json(SANITY_DIR / "pytorch_sanity_report.json", report)
    print(f"Dataset: {dataset_root} | {counts}")
    print(f"Strict sample-ID dry runtime hashes: {dry_hashes}")
    print("Epoch mapping: 0..29 | exact=True")
    print(
        f"Model Output: {tuple(output.shape)} | parameters={parameters:,} "
        f"| finite=True | device={output.device}"
    )


def print_experiment_config() -> None:
    print("Experiment: 03 Augmentation Alignment")
    print("Changed Variable: Augmentation only")
    print("Alignment Mode: Strict sample-ID")
    print(f"Seeds: {SEEDS}")
    print(f"RUN_TRAINING: {RUN_TRAINING}")


if __name__ == "__main__":
    print_experiment_config()
    run_sanity_check()
    if RUN_TRAINING:
        run_3seed_experiment()
    else:
        print("Training was not started.\nSet RUN_TRAINING = True to start training.")
