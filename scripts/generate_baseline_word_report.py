"""Generate a Word report from the saved baseline result CSV files."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cnn-report-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "00_baseline_final_cnn_3seed"
RESULTS = EXP_DIR / "results"
OUTPUT = EXP_DIR / "baseline_3seed_comparison_report.docx"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def mean_std(series: pd.Series, as_percent: bool = False) -> str:
    mean, std = float(series.mean()), float(series.std(ddof=1))
    scale = 100 if as_percent else 1
    suffix = "%" if as_percent else ""
    return f"{mean * scale:.2f} ± {std * scale:.2f}{suffix}"


def performance_plot(keras: pd.DataFrame, torch: pd.DataFrame, path: Path) -> None:
    x = np.arange(3)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("test_accuracy", "macro_f1"),
        ("Test Accuracy by Seed", "Macro F1 by Seed"),
    ):
        ax.bar(x - width / 2, keras[metric] * 100, width, label="Keras", color="#4C78A8")
        ax.bar(x + width / 2, torch[metric] * 100, width, label="PyTorch", color="#F58518")
        ax.set(title=title, xlabel="Seed", ylabel="Percent (%)")
        ax.set_xticks(x, keras["seed"].astype(str))
        ax.set_ylim(0, 60)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def snapshot_plot(keras: pd.DataFrame, torch: pd.DataFrame, path: Path) -> None:
    x = np.arange(3)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for row, (name, data) in enumerate((("Keras", keras), ("PyTorch", torch))):
        acc_ax, loss_ax = axes[row]
        acc_ax.plot(x, data["train_accuracy"] * 100, "o-", label="Train")
        acc_ax.plot(x, data["validation_accuracy"] * 100, "o-", label="Validation")
        acc_ax.set(title=f"{name}: Accuracy at Best Epoch", ylabel="Accuracy (%)")
        loss_ax.plot(x, data["train_loss"], "o-", label="Train")
        loss_ax.plot(x, data["validation_loss"], "o-", label="Validation")
        loss_ax.set(title=f"{name}: Loss at Best Epoch", ylabel="Loss")
        for ax in (acc_ax, loss_ax):
            ax.set_xticks(x, data["seed"].astype(str))
            ax.grid(alpha=0.25)
            ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel("Seed")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def time_plot(keras: pd.DataFrame, torch: pd.DataFrame, path: Path) -> None:
    x = np.arange(3)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, keras["training_time_seconds"] / 60, width, label="Keras")
    ax.bar(x + width / 2, torch["training_time_seconds"] / 60, width, label="PyTorch")
    ax.set(title="Training Time by Seed", xlabel="Seed", ylabel="Minutes")
    ax.set_xticks(x, keras["seed"].astype(str))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    keras = pd.read_csv(RESULTS / "keras_results.csv").sort_values("seed").reset_index(drop=True)
    torch = pd.read_csv(RESULTS / "pytorch_results.csv").sort_values("seed").reset_index(drop=True)
    expected = [42, 123, 2026]
    if keras["seed"].tolist() != expected or torch["seed"].tolist() != expected:
        raise RuntimeError("Both CSV files must contain seeds 42, 123, and 2026.")
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("pandoc was not found.")

    merged = keras.merge(torch, on="seed", suffixes=("_keras", "_torch"))
    rows = []
    for row in merged.itertuples(index=False):
        rows.append(
            f"| {row.seed} | {pct(row.test_accuracy_keras)} | {pct(row.test_accuracy_torch)} | "
            f"{(row.test_accuracy_torch - row.test_accuracy_keras) * 100:+.2f}%p | "
            f"{pct(row.macro_f1_keras)} | {pct(row.macro_f1_torch)} |"
        )

    with tempfile.TemporaryDirectory(prefix="baseline-report-") as temp_dir:
        temp = Path(temp_dir)
        performance = temp / "performance.png"
        snapshots = temp / "snapshots.png"
        times = temp / "times.png"
        performance_plot(keras, torch, performance)
        snapshot_plot(keras, torch, snapshots)
        time_plot(keras, torch, times)

        acc_delta = float((torch["test_accuracy"] - keras["test_accuracy"]).mean() * 100)
        f1_delta = float((torch["macro_f1"] - keras["macro_f1"]).mean() * 100)
        keras_minutes = float(keras["training_time_seconds"].mean() / 60)
        torch_minutes = float(torch["training_time_seconds"].mean() / 60)
        markdown = f"""# CNN Baseline 3-Seed 성능 비교 보고서

작성일: 2026-09-14  
실험: `00_baseline_final_cnn_3seed`  
Seed: 42, 123, 2026

## 1. 실험 개요

동일한 최종 CNN 구조를 Keras와 PyTorch로 구현해 동일한 물리적 train/val/test 데이터셋에서 비교했다. Keras는 TensorFlow Metal GPU, PyTorch는 MPS를 사용했다.

## 2. Seed별 Test 성능

| Seed | Keras Accuracy | PyTorch Accuracy | PyTorch - Keras | Keras Macro F1 | PyTorch Macro F1 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![Seed별 Test Accuracy와 Macro F1]({performance.as_posix()})

## 3. 3-Seed 요약

| 지표 | Keras 평균 ± 표준편차 | PyTorch 평균 ± 표준편차 | 평균 차이 |
|---|---:|---:|---:|
| Test Accuracy | {mean_std(keras['test_accuracy'], True)} | {mean_std(torch['test_accuracy'], True)} | {acc_delta:+.2f}%p |
| Macro F1 | {mean_std(keras['macro_f1'], True)} | {mean_std(torch['macro_f1'], True)} | {f1_delta:+.2f}%p |
| Test Loss | {mean_std(keras['test_loss'])} | {mean_std(torch['test_loss'])} | {float((torch['test_loss'] - keras['test_loss']).mean()):+.4f} |
| Train Accuracy | {mean_std(keras['train_accuracy'], True)} | {mean_std(torch['train_accuracy'], True)} | {float((torch['train_accuracy'] - keras['train_accuracy']).mean() * 100):+.2f}%p |
| Validation Accuracy | {mean_std(keras['validation_accuracy'], True)} | {mean_std(torch['validation_accuracy'], True)} | {float((torch['validation_accuracy'] - keras['validation_accuracy']).mean() * 100):+.2f}%p |
| Train-Val Gap | {mean_std(keras['train_val_gap'], True)} | {mean_std(torch['train_val_gap'], True)} | {float((torch['train_val_gap'] - keras['train_val_gap']).mean() * 100):+.2f}%p |

PyTorch는 평균 Test Accuracy에서 Keras보다 **{acc_delta:.2f}%p**, Macro F1에서 **{f1_delta:.2f}%p** 높았다. PyTorch의 Seed 간 Test Accuracy 표준편차는 {torch['test_accuracy'].std(ddof=1) * 100:.2f}%p로 Keras의 {keras['test_accuracy'].std(ddof=1) * 100:.2f}%p보다 작았다. 반면 PyTorch의 Train-Val gap은 더 커서 과적합 경향도 함께 관찰된다.

## 4. 저장된 Best-Epoch Loss/Accuracy

아래 그래프는 각 Seed의 **기록된 best epoch 한 시점**에서 Train/Validation loss와 accuracy를 비교한다. Seed 사이의 선은 시각적 비교용이며 연속 epoch 학습곡선이 아니다.

![Best epoch의 Train/Validation Loss와 Accuracy]({snapshots.as_posix()})

| Framework | Seed 42 | Seed 123 | Seed 2026 | 학습 Epoch |
|---|---:|---:|---:|---:|
| Keras best epoch | {int(keras.loc[0, 'best_epoch'])} | {int(keras.loc[1, 'best_epoch'])} | {int(keras.loc[2, 'best_epoch'])} | 30 / 30 / 30 |
| PyTorch best epoch | {int(torch.loc[0, 'best_epoch'])} | {int(torch.loc[1, 'best_epoch'])} | {int(torch.loc[2, 'best_epoch'])} | 30 / 30 / 30 |

### Epoch별 전체 곡선의 제한사항

이번 학습은 epoch별 history를 별도 CSV/JSON/TensorBoard 파일로 저장하지 않았다. Keras `.keras` 파일에는 모델 설정과 가중치만 있고 PyTorch `.pt` 파일에는 `state_dict`만 있다. 따라서 완료된 학습의 epoch별 loss/accuracy 전체 곡선을 정확히 복원할 수 없다. 데이터를 임의 생성하지 않고, 본 문서에는 CSV에 실제 저장된 best-epoch 값만 시각화했다. 전체 epoch 곡선은 향후 실행에서 history 저장 기능을 활성화해야 생성할 수 있다.

## 5. 학습 시간

![Seed별 학습 시간]({times.as_posix()})

| Framework | Seed당 평균 학습 시간 | 비교 |
|---|---:|---:|
| Keras | {keras_minutes:.2f}분 | 기준 |
| PyTorch | {torch_minutes:.2f}분 | {keras_minutes - torch_minutes:.2f}분 단축 |

두 프레임워크의 GPU backend가 TensorFlow Metal과 PyTorch MPS로 다르므로 학습 시간은 참고 지표로 해석해야 한다.

## 6. 결론

Baseline에서는 PyTorch가 세 Seed 모두에서 더 높은 Test Accuracy와 Macro F1을 보였고 Seed 간 변동도 작았다. 그러나 초기화, augmentation, batch 순서, optimizer 세부값, loss/output 처리 및 callback 차이가 남아 있으므로 프레임워크 자체가 원인이라고 단정할 수 없다. 실험 01~08에서 요인을 순차 정렬해 성능 격차 변화를 검증해야 한다.

## 7. 원본 결과

- `results/keras_results.csv`
- `results/pytorch_results.csv`
- `results/keras_seed{{seed}}.keras`
- `results/pytorch_seed{{seed}}.pt`
"""
        source = temp / "report.md"
        source.write_text(markdown, encoding="utf-8")
        subprocess.run(
            [pandoc, str(source), "--resource-path", str(temp), "-o", str(OUTPUT)],
            check=True,
        )

    print(OUTPUT)


if __name__ == "__main__":
    main()
