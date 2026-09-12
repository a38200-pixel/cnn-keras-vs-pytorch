"""PyTorch reproduction of the notebook's final RGB CNN. Training is opt-in."""
# ============================================================
# 1. Imports
# ============================================================
from __future__ import annotations
import copy, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.alignment_utils import augmentation_decision, epoch_permutation
from common.environment_utils import print_torch_environment, select_torch_device
from common.dataset_utils import CLASSES, build_split_manifest, find_dataset_root
from common.result_utils import save_result
from common.seed_utils import seed_pytorch

# ============================================================
# 2. Experiment Configuration
# ============================================================
EXPERIMENT = "03_augmentation_alignment"
ALIGNMENT = "augmentation"
SEEDS = [42, 123, 2026]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS = 128, 32, 8, 30
LEARNING_RATE, EARLY_PATIENCE, LR_PATIENCE, MIN_DELTA = 0.001, 7, 4, 1e-4
RUN_TRAINING = False
RESULTS = Path(__file__).parent / "results" / "pytorch_results.csv"

# ============================================================
# 3. Random Seed
# ============================================================
def set_seed(seed: int) -> None:
    seed_pytorch(seed)

# ============================================================
# 4. Dataset
# ============================================================
def make_dataset(rows, seed: int, training: bool):
    import torch
    from PIL import Image, ImageOps
    from torchvision import transforms
    baseline_train = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(.5), transforms.RandomRotation(5,
        interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    baseline_eval = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    native_aug = transforms.Compose([transforms.RandomHorizontalFlip(.5), transforms.RandomRotation(5,
        interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    class EmotionDataset(torch.utils.data.Dataset):
        epoch = 0
        def __len__(self): return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            image = Image.open(str(row["path"])).convert("RGB")
            if ALIGNMENT == "input_tensor":
                image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
                tensor = (native_aug if training else transforms.ToTensor())(image)
            elif ALIGNMENT in {"augmentation", "diagnostic"}:
                image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
                if training and ALIGNMENT == "augmentation":
                    flip, angle = augmentation_decision(str(row["path"]), seed, self.epoch)
                    if flip: image = ImageOps.mirror(image)
                    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                array = np.asarray(image, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            else:
                tensor = (baseline_train if training else baseline_eval)(image)
            return tensor, int(row["label"])
    return EmotionDataset()


def make_loader(dataset, seed: int, training: bool, epoch: int = 0):
    import torch
    if training and ALIGNMENT == "batch_order":
        sampler = epoch_permutation(len(dataset), seed, epoch)
        return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=training,
        generator=generator if training else None, num_workers=0)

# ============================================================
# 5. CNN Model
# ============================================================
def build_model(seed: int):
    import torch
    from torch import nn
    class FinalRGBModel(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_channels = 3
            for channels in (32, 64, 128, 256):
                blocks.extend([nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels, eps=1e-3, momentum=.01), nn.ReLU(), nn.MaxPool2d(2)])
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
    model = FinalRGBModel()
    if ALIGNMENT in {"initial_weight", "diagnostic"}: initialize_common_weights(model, seed)
    return model


def initialize_common_weights(model, seed: int) -> None:
    import torch
    rng = np.random.default_rng(seed)
    modules = [m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        for module in modules:
            if isinstance(module, torch.nn.Conv2d):
                keras_shape = (module.kernel_size[0], module.kernel_size[1], module.in_channels, module.out_channels)
                fan_in, fan_out = np.prod(keras_shape[:-1]), keras_shape[-1]
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                values = rng.uniform(-limit, limit, keras_shape).astype("float32").transpose(3, 2, 0, 1)
            else:
                keras_shape = (module.in_features, module.out_features)
                limit = np.sqrt(6.0 / (module.in_features + module.out_features))
                values = rng.uniform(-limit, limit, keras_shape).astype("float32").T
            module.weight.copy_(torch.from_numpy(values))
            if module.bias is not None: module.bias.zero_()


def make_optimizer_and_loss(model):
    import torch
    epsilon = 1e-7 if ALIGNMENT == "adam" else 1e-8
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(.9, .999),
                                 eps=epsilon, weight_decay=0, amsgrad=False)
    return optimizer, torch.nn.CrossEntropyLoss()

# ============================================================
# 6. Training
# ============================================================
def train_one_seed(seed: int) -> None:
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    set_seed(seed)
    device = print_torch_environment(EXPERIMENT, seed)
    manifest = build_split_manifest()
    train_data, val_data, test_data = (make_dataset(manifest[s], seed, s == "train") for s in ("train", "val", "test"))
    val_loader, test_loader = make_loader(val_data, seed, False), make_loader(test_data, seed, False)
    train_loader = make_loader(train_data, seed, True)
    model = build_model(seed).to(device)
    optimizer, criterion = make_optimizer_and_loss(model)
    # PyTorch reduces after `patience + 1` bad epochs whereas Keras reduces when
    # wait reaches patience. Experiment 08 subtracts one to align epoch timing.
    scheduler_patience = LR_PATIENCE - 1 if ALIGNMENT == "callbacks" else LR_PATIENCE
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
        patience=scheduler_patience, threshold=MIN_DELTA, threshold_mode="abs", min_lr=1e-6)
    history = {k: [] for k in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")}
    best_loss, best_epoch, bad_epochs, best_state = float("inf"), 0, 0, None
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        train_data.epoch = epoch
        if ALIGNMENT == "batch_order":
            train_loader = make_loader(train_data, seed, True, epoch)
        model.train(); loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True); outputs = model(images); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(labels); correct += (outputs.argmax(1) == labels).sum().item(); total += len(labels)
        train_loss, train_acc = loss_sum / total, correct / total
        model.eval(); loss_sum = correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device); outputs = model(images); loss = criterion(outputs, labels)
                loss_sum += loss.item() * len(labels); correct += (outputs.argmax(1) == labels).sum().item(); total += len(labels)
        val_loss, val_acc = loss_sum / total, correct / total
        for key, value in (("train_loss", train_loss), ("train_accuracy", train_acc),
                           ("val_loss", val_loss), ("val_accuracy", val_acc)): history[key].append(value)
        improved = val_loss < best_loss - MIN_DELTA
        if improved:
            best_loss, best_epoch, bad_epochs, best_state = val_loss, epoch, 0, copy.deepcopy(model.state_dict())
        else: bad_epochs += 1
        scheduler.step(val_loss)
        if bad_epochs >= EARLY_PATIENCE: break
    elapsed = time.perf_counter() - start
    if best_state is not None: model.load_state_dict(best_state)
    torch.save(model.state_dict(), Path(__file__).parent / "results" / f"pytorch_seed{seed}.pt")
    model.eval(); y_true, y_pred, loss_sum, total = [], [], 0., 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device)); loss_sum += criterion(outputs, labels.to(device)).item() * len(labels)
            y_true.extend(labels.tolist()); y_pred.extend(outputs.argmax(1).cpu().tolist()); total += len(labels)
    save_result(RESULTS, {"experiment": EXPERIMENT, "framework": "pytorch", "seed": seed,
        "test_accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "test_loss": loss_sum / total, "best_epoch": best_epoch + 1, "epochs_trained": len(history["train_loss"]),
        "train_accuracy": history["train_accuracy"][best_epoch], "validation_accuracy": history["val_accuracy"][best_epoch],
        "train_loss": history["train_loss"][best_epoch], "validation_loss": history["val_loss"][best_epoch],
        "train_val_gap": history["train_accuracy"][best_epoch] - history["val_accuracy"][best_epoch],
        "training_time_seconds": elapsed})

# ============================================================
# 7. Evaluation / sanity check
# ============================================================
def run_sanity_check() -> None:
    import torch
    set_seed(SEEDS[0]); device = print_torch_environment(EXPERIMENT, SEEDS[0]); manifest = build_split_manifest()
    data = make_dataset(manifest["train"][:BATCH_SIZE], SEEDS[0], True)
    images, labels = next(iter(make_loader(data, SEEDS[0], False)))
    images, labels = images.to(device), labels.to(device)
    model = build_model(SEEDS[0]).to(device).eval()
    with torch.no_grad(): output = model(images)
    assert tuple(images.shape) == (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    assert tuple(output.shape) == (BATCH_SIZE, NUM_CLASSES) and torch.isfinite(output).all()
    assert next(model.parameters()).device == images.device == labels.device == output.device
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"dataset={find_dataset_root()} classes={CLASSES}")
    print(f"batch={tuple(images.shape)} output={tuple(output.shape)} parameters={parameters:,} finite=True device={images.device}")

# ============================================================
# 8. Result Saving / entry point
# ============================================================
def print_experiment_config() -> None:
    print({"experiment": EXPERIMENT, "framework": "pytorch", "alignment": ALIGNMENT,
           "seeds": SEEDS, "run_training": RUN_TRAINING})
def run_3seed_experiment() -> None:
    for seed in SEEDS: train_one_seed(seed)
if __name__ == "__main__":
    print_experiment_config(); run_sanity_check()
    if RUN_TRAINING: run_3seed_experiment()
    else: print("Training was not started.\nSet RUN_TRAINING = True to start training.")
