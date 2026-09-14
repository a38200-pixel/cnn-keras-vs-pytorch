# Experiment 00 - Baseline Final CNN 3-Seed

## 목적

기존 단일 Seed에서 관찰된 framework gap이 Seed 42, 123, 2026에서도 같은 방향으로 반복되는지 확인한다.

## Baseline

새 공통 모델이 아니라 두 기존 노트북의 최종 RGB CNN 구현을 각 framework 방식 그대로 재현한다.

## Hypothesis

같은 방향의 gap이 세 Seed에서 반복되면 단일 초기화의 우연만으로 설명하기 어려운 재현 가능한 차이일 수 있다.

## Changed Variable

없음.

## Fixed / Unchanged Conditions

입력 128×128 RGB, batch 32, 최대 30 epochs, Adam 0.001, CNN 구조와 callback 설정을 유지한다.

## Seeds

42, 123, 2026

## Implementation

Keras와 PyTorch의 입력 처리, 증강, 초기화, output/loss, Adam 기본 epsilon 및 학습 loop 차이를 보존한다. 현재 데이터는 클래스별 고정 70/15/15 물리 폴더로 나뉘며 모든 실험에서 같은 표본을 사용한다.

## Results

3개 Seed 모두 최대 30 epoch 학습과 고정 Test split 평가를 완료했다.

### Seed별 Test 성능

| Seed | Keras Accuracy | PyTorch Accuracy | PyTorch - Keras | Keras Macro F1 | PyTorch Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 42 | 49.34% | 51.16% | +1.82%p | 48.37% | 50.23% |
| 123 | 47.25% | 51.07% | +3.81%p | 45.69% | 50.18% |
| 2026 | 44.98% | 51.29% | +6.31%p | 43.48% | 49.82% |

### 3-Seed 요약

표준편차는 3개 Seed의 표본 표준편차(`ddof=1`)다.

| 지표 | Keras 평균 ± 표준편차 | PyTorch 평균 ± 표준편차 | PyTorch - Keras 평균 차이 |
|---|---:|---:|---:|
| Test Accuracy | 47.19 ± 2.18% | **51.17 ± 0.11%** | **+3.98%p** |
| Macro F1 | 45.84 ± 2.45% | **50.08 ± 0.23%** | **+4.23%p** |
| Test Loss | 1.3669 ± 0.0404 | **1.2733 ± 0.0181** | **-0.0936** |
| Train Accuracy at Best Epoch | 51.42 ± 2.48% | 60.55 ± 1.41% | +9.13%p |
| Validation Accuracy at Best Epoch | 46.75 ± 2.64% | **51.08 ± 0.71%** | +4.33%p |
| Train-Val Accuracy Gap | **4.67 ± 0.69%p** | 9.47 ± 2.11%p | +4.80%p |

### Epoch와 학습 시간

| Framework | Seed별 Best Epoch | 실제 학습 Epoch | Seed당 평균 학습 시간 |
|---|---|---|---:|
| Keras | 42: 30, 123: 28, 2026: 27 | 30 / 30 / 30 | 81.56분 |
| PyTorch | 42: 30, 123: 24, 2026: 23 | 30 / 30 / 30 | 75.58분 |

PyTorch가 Seed당 평균 약 5.98분(7.3%) 빨랐다. 다만 Keras는 TensorFlow Metal, PyTorch는 MPS를 사용하므로 이 시간 차이는 순수 framework 속도 차이로 해석하지 않는다.

### 결과 파일

- [Keras 결과 CSV](results/keras_results.csv)
- [PyTorch 결과 CSV](results/pytorch_results.csv)
- [3-Seed 비교 Word 보고서](baseline_3seed_comparison_report.docx)
- 학습 모델: `results/keras_seed{seed}.keras`, `results/pytorch_seed{seed}.pt`

현재 상태: **Baseline 3-Seed GPU 학습 및 Test 평가 완료**.

## Comparison with Baseline

이 실험 자체가 비교 기준이다.

## Interpretation

PyTorch는 세 Seed 모두에서 Keras보다 높은 Test Accuracy와 Macro F1을 기록했다. 평균 차이는 Accuracy +3.98%p, Macro F1 +4.23%p이며, PyTorch의 Accuracy 표준편차도 0.11%p로 Keras의 2.18%p보다 작아 이번 3개 Seed에서는 더 안정적이었다.

PyTorch는 Train Accuracy가 더 높고 Train-Val gap도 평균 9.47%p로 Keras의 4.67%p보다 컸다. 즉 더 강하게 fitting하면서 Test 성능도 높아졌지만 과적합 경향 역시 더 크게 나타났다.

이 결과로 framework 자체가 성능 차이의 원인이라고 단정할 수는 없다. Baseline은 각 framework의 기존 구현 방식을 보존하므로 초기화, 입력 tensor 처리, augmentation, batch 순서, output/loss, Adam epsilon과 callback 동작 차이가 함께 남아 있다. 실험 01~08에서 각 요인을 하나씩 정렬한 뒤 gap의 변화량으로 원인을 검증한다.

## Limitations

- 고정 split 하나와 3개 Seed만으로 전체 변동 분포나 통계적 유의성을 충분히 추정할 수 없다.
- TensorFlow Metal과 PyTorch MPS는 서로 다른 GPU backend라 학습시간의 직접 비교에는 제약이 있다.
- 이번 실행은 epoch별 history를 별도 파일로 저장하지 않았다. 결과 CSV에는 best epoch 한 시점의 train/validation loss와 accuracy만 있으므로 전체 epoch 학습곡선은 사후 복원할 수 없다.
