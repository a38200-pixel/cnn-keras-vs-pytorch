# Keras vs PyTorch CNN Framework Difference Analysis

## 동일 출발점에서 CNN 학습 과정의 Framework별 수치 분기 추적

### 1. 프로젝트 개요

기존 어린이 얼굴 감정 분류 프로젝트에서 동일한 최종 CNN의 Keras/PyTorch 성능 차이를 관찰했다. 00–03에서는 Framework별 native 조건을 탐색했고, 04부터는 동일한 W0·입력·배치·증강 결과 등 외부 조건을 통제해 **Forward → Loss → Gradient → Optimizer Update → BN State → 학습 trajectory 중 어디서 처음 수치적으로 달라지는지** 추적한다. 목적은 어느 Framework가 본질적으로 우수한지 순위를 매기는 것이 아니다. 기존 두 notebook과 00–03 결과는 보존한다.

### Experimental Environment

00–09의 모든 실험은 아래 한 환경으로 고정한다. Framework 차이를 연구하는 동안 hardware, Python 또는 framework version이 바뀌면 새로운 혼란 변수가 생기기 때문이다.

| Category | Fixed environment |
|---|---|
| Hardware | MacBook Pro, Apple M1 Pro, 14-core GPU, Metal 4 |
| Architecture | arm64 |
| Project environment | `.venv-metal` |
| Python | 3.10.3 |
| TensorFlow / Keras | TensorFlow 2.18.1, Keras 3.12.4 |
| TensorFlow backend | tensorflow-metal 1.2.0, Apple GPU `/GPU:0` |
| PyTorch | PyTorch 2.14.0, torchvision 0.29.0 |
| PyTorch backend | MPS, Apple M1 Pro GPU |

TensorFlow 2.21.0과 tensorflow-metal 조합에서는 `libmetal_plugin.dylib` / `_pywrap_tensorflow_internal.so` loading 호환성 문제가 있었다. 별도 `.venv-metal`에서 TensorFlow 2.18.1 + tensorflow-metal 1.2.0의 실제 GPU 연산을 검증했으며, 이후 이 환경을 최종 실행환경으로 고정했다. MPS의 미지원 연산을 조용히 CPU에서 실행시키는 `PYTORCH_ENABLE_MPS_FALLBACK=1`은 사용하지 않는다.

### 2. 연구 배경

기존 단일 Seed 실험에서는 Keras와 PyTorch 사이에 성능 차이가 관찰됐다. 그러나 random initialization, data order, augmentation, optimizer와 framework 내부 구현의 영향을 단일 실행으로 분리할 수 없다. 00–03은 native 차이를 하나씩 살핀 preliminary OFAT이고, 04부터는 공통 초기 상태·입력·update 수에서 실제 학습 연산의 수치 분기를 추적한다.

```text
기존 CNN 구현 → Phase 1: 00–03 native/OFAT preliminary comparison
→ Phase 2: 04 common W0·input·batch·augmentation controlled baseline
→ 05–09 forward/loss/gradient/update/BN/multi-step/layer trajectory 분석
→ fixed Epoch 30 성능을 마지막 결과로 해석
```

### 3. Baseline 정의

Baseline은 새로 만든 공통 CNN이 아니다. [Keras notebook](Keras_emotion_classification.ipynb)과 [PyTorch notebook](Pytorch_emotion_classification.ipynb)를 기반하여 **Final RGB CNN**을 framework별 구현 차이를 보존해 재구현한 것이다. 

#### Baseline 구현 확인

| 항목 | Keras 최종 notebook | PyTorch 최종 notebook | 확인 결과 |
|---|---|---|---|
| 입력 | RGB NHWC, directory loader의 0–255 tensor 후 model 내부 `Rescaling(1/255)` | RGB NCHW, PIL `Resize` 후 `ToTensor` | 수치 변환 위치/layout이 다름 |
| 증강 | `RandomFlip`, `RandomRotation(0.014)`; rotation 기본 reflect fill | `RandomHorizontalFlip(.5)`, `RandomRotation(5°, bilinear)`; 기본 zero fill | 각도는 약 ±5°지만 operator/fill/RNG가 다름 |
| Conv | same padding, bias 없음, 32→64→128→256 | padding=1, bias 없음, 32→64→128→256 | 구조 일치 |
| BN | Keras 기본 epsilon=.001, momentum=.99 | eps=.001, momentum=.01 | update 정의가 반대라 실질적으로 이미 대응됨 |
| Head | GAP→Dense128+ReLU→Dense8+Softmax | GAP→Linear128→ReLU→Linear8 logits | output/loss 경로가 다름 |
| Loss | sparse categorical cross-entropy (probability 입력) | cross-entropy (logits 입력) | 다름 |
| Adam | lr=.001, 그 밖은 framework 기본값 | lr=.001, 그 밖은 framework 기본값 | 대표적으로 epsilon 기본값이 다름 |
| Callback | ES 7, LR 4, min_delta=1e-4, best restore | 수동 ES 7, scheduler 4, threshold=1e-4, state restore | 의도는 같지만 scheduler patience epoch 정의가 다름 |
| Dropout | 없음 | 없음 | 일치 |

Notebook의 최종 코드와 사용자 제공 설정은 구조, RGB 입력, batch 32, 최대 30 epochs, Adam lr=.001, Seed 42, augmentation, dropout 없음 및 patience에서 일치한다. Keras 총 parameter는 BN moving statistics를 포함해 423,784개이며 PyTorch trainable parameter는 422,824개다. 이 960개 차이는 Keras `count_params()`가 4개 BN의 non-trainable moving mean/variance도 세기 때문이다.

### 4. 연구 목적

1. 기존 성능 차이의 3-Seed 재현성과 native 구현 조건을 탐색한다(Phase 1).
2. 동일한 초기 가중치·입력 tensor·batch/augmentation schedule·update 수를 확립한다(Phase 2).
3. Forward, Loss, Gradient, Adam update, BN state 중 첫 수치 차이와 그 크기를 추적한다.
4. 그 차이가 고정 Epoch 30의 Accuracy/Macro F1과 어떻게 연결되는지 검토한다.

### 5. Research Questions

- **RQ1:** 기존 성능 차이는 Seed 42, 123, 2026에서 일관되게 반복되는가?
- **RQ2:** Phase 1에서 Input Tensor·Batch Order·Augmentation 조건을 개별 정렬하면 native 성능 Gap은 어떻게 바뀌는가?
- **RQ3:** Phase 2의 동일 W0·입력·batch에서 어느 연산부터 numerical difference가 관찰되는가?
- **RQ4:** 첫 차이가 gradient/update/BN state와 30-epoch trajectory·최종 성능에서 어떻게 나타나는가?

### 6. Hypotheses

- **H0 — Null:** 관찰된 차이는 stochastic variation 범위이며 3-Seed에서 일관된 framework gap이 나타나지 않을 것이다.
- **H1 — Reproducibility:** 주요 구조와 hyperparameter가 같아도 framework-specific 구현 차이로 여러 Seed에서 같은 방향의 gap이 반복될 수 있다.
- **H2 — Phase 1 Single-Factor Alignment:** 01–03의 개별 조건 정렬은 Baseline 대비 성능 Gap에 영향을 줄 수 있다.
- **H3 — Phase 2 Numerical Divergence:** 동일 W0·input·label에서도 native 연산 이후 작은 numerical difference가 처음 관찰될 수 있으며, 이후 update·trajectory에서 변화할 수 있다. 첫 비영 차이를 곧바로 실용적으로 유의한 분기로 해석하지 않는다.

### 7. Dataset

현재 실제 경로는 명세의 `dataset/`이 아니라 `emotion_dataset/`이다. 이미지는 아래처럼 물리적으로 분할되어 있으며 코드는 `dataset/`과 `emotion_dataset/` 두 이름을 모두 탐색한다.

```text
emotion_dataset/
├── train/{anger, contempt, disgust, fear, happy, neutral, sad, surprise}/
├── val/{anger, contempt, disgust, fear, happy, neutral, sad, surprise}/
└── test/{anger, contempt, disgust, fear, happy, neutral, sad, surprise}/
```

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| anger | 1,275 | 273 | 274 | 1,822 |
| contempt | 1,283 | 274 | 276 | 1,833 |
| disgust | 1,218 | 261 | 261 | 1,740 |
| fear | 1,287 | 275 | 277 | 1,839 |
| happy | 1,303 | 279 | 280 | 1,862 |
| neutral | 1,316 | 282 | 282 | 1,880 |
| sad | 1,274 | 273 | 274 | 1,821 |
| surprise | 1,295 | 277 | 279 | 1,851 |
| **Total** | **10,251** | **2,194** | **2,203** | **14,648** |

과거 split은 이번 검증 조건으로 사용하지 않는다. 현재 데이터 전체를 파일명 hash와 split seed 42로 **각 클래스별 고정 70/15/15**로 물리 분할했다: train 10,251, validation 2,194, test 2,203. 파일은 복제하지 않고 해당 폴더로 이동했으며 모든 framework와 실험 branch가 이 동일한 split을 직접 읽는다. `emotion_dataset/` 전체는 `.gitignore`로 Git 추적에서 제외한다.

### 8. Baseline CNN Architecture

| Stage | Operation | Output |
|---|---|---|
| Input | RGB | 128×128×3 |
| Block 1 | Conv 3→32, 3×3, bias=False → BN → ReLU → MaxPool | 64×64×32 |
| Block 2 | Conv 32→64 → BN → ReLU → MaxPool | 32×32×64 |
| Block 3 | Conv 64→128 → BN → ReLU → MaxPool | 16×16×128 |
| Block 4 | Conv 128→256 → BN → ReLU → MaxPool | 8×8×256 |
| Head | GAP → Dense/Linear 256→128 → ReLU → 128→8 | 8 classes |

Keras는 `padding="same"`, PyTorch는 `padding=1`을 사용한다.

### 9. Experimental Methodology

**Phase 1 (00–03)**은 Framework별 native 조건을 보존하거나 한 요인씩 정렬하는 preliminary OFAT다. **Phase 2 (04–09)**는 여러 외부 조건을 동시에 고정하는 strict controlled comparison이다. 04는 Phase 1 OFAT 표의 다음 행이 아니라 새로운 controlled baseline이다. 04부터 Accuracy/F1보다 W0→입력→activation→loss→gradient→update→BN state 순서를 먼저 분석한다.

Seed별 Gap은 `PyTorch metric_seed − Keras metric_seed`로 정의한다. `Signed Mean Gap = mean(Gap_seed)`은 평균 우위 방향을, `Mean Absolute Paired Gap = mean(abs(Gap_seed))`은 Seed별 차이의 평균 크기를 나타낸다. 각 지표의 Gap Reduction은 `baseline gap − current gap`으로 계산한다. 양수는 Gap 감소, 0 근처는 영향이 작음, 음수는 Gap 증가를 뜻하며 Baseline gap이 거의 0이면 reduction rate는 N/A다.

### 10. 왜 누적 통제를 사용하지 않았는가?

이 원칙은 **Phase 1의 01–03에만** 적용된다. 각 branch가 00 Baseline으로 돌아가 한 요소만 정렬하므로 그 결과는 누적 실험이 아니다. Phase 2는 다른 질문—동일 출발점에서 첫 수치 분기 위치—에 답하기 위해 여러 조건을 동시에 고정한다. 03에서 augmentation stochastic parameter를 맞춘 뒤에도 Seed별 결과와 학습 dynamics가 달랐으므로, 04에서는 W0를 포함한 공통 초기 상태를 먼저 확립한다.

### 11. Experiment Plan

#### Phase 1 - Native / Preliminary Framework Comparison

| ID | Experiment | Baseline 대비 변경 변수 | Seeds | 목적 |
|---|---|---|---|---|
| 00 | Baseline 3-Seed | 없음 | 42, 123, 2026 | 기존 성능 차이 재현성 확인 |
| 01 | Input Tensor Alignment | Input Tensor | 42, 123, 2026 | 입력 처리 영향 확인 |
| 02 | Batch Order Alignment | Batch Order | 42, 123, 2026 | Mini-batch 순서 영향 확인 |
| 03 | Augmentation Alignment | Augmentation | 42, 123, 2026 | 증강 구현 영향 확인 |

#### Phase 2 - Strict Controlled Framework Comparison

| ID | Experiment | Status | Primary focus |
|---|---|---|---|
| 04 | [Common Initialization & Controlled Training](experiments/04_common_initialization_controlled_training/README.md) | Extended diagnostic VALID / Full Training Pending | W0·입력 exact, first-step Adam/reference 진단, 독립 0–100-step trajectory, fixed Epoch 30 baseline |
| 05 | Forward & Loss Divergence Analysis | Placeholder | Activation·logits·native/reference CE |
| 06 | Gradient & Optimizer Update Divergence | Placeholder | Gradient와 native Adam update |
| 07 | BatchNorm State Divergence | Placeholder | BN output/running state |
| 08 | Multi-Step / Epoch-Level Divergence | Placeholder | update·epoch trajectory |
| 09 | Layer-by-Layer Training Trajectory | Placeholder | 저장된 checkpoint별 layer 비교 |

### 12. Experiment Progress

| ID | Experiment | 구현 | 3-Seed 학습 | 분석 |
|---|---|---|---|---|
| ENV | `.venv-metal` 환경 | 완료 | - | GPU 연산 검증 완료 |
| DATA | Dataset physical split | 완료 | - | 70/15/15 검증 완료 |
| 00 | Baseline | 완료 | 완료 | 3-Seed 결과 분석 완료 |
| 01 | Input Tensor | 완료, Input equality/GPU sanity 통과 | 완료 | Baseline 비교 및 history 분석 완료 |
| 02 | Batch Order | Attempt 1 Invalid / Bug Fix 완료 | Attempt 2 VALID / 3-Seed 완료 | Runtime 검증 및 Baseline 분석 완료 |
| 03 | Augmentation | 구현 완료, strict sample-ID runtime 검증 | 3-Seed 완료 | VALID / Baseline 비교 및 history 분석 완료 |
| 04 | Common Initialization & Controlled Training | 구현 완료 / 3 gate VALID / first-step·100-step 진단 완료 | 대기 | W0·input exact, 첫 비영 BN1 차이 관찰; checkpoint round-trip PASS |
| 05 | Forward & Loss Divergence | Placeholder | 대기 | 대기 |
| 06 | Gradient & Optimizer Update Divergence | Placeholder | 대기 | 대기 |
| 07 | BatchNorm State Divergence | Placeholder | 대기 | 대기 |
| 08 | Multi-Step / Epoch-Level Divergence | Placeholder | 대기 | 대기 |
| 09 | Layer-by-Layer Training Trajectory | Placeholder | 대기 | 대기 |

### 13. Baseline 3-Seed Results

<!-- BASELINE_RESULTS_START -->
| Metric | Keras (mean ± sample std) | PyTorch (mean ± sample std) | Signed Gap (PyTorch - Keras) |
|---|---:|---:|---:|
| Test Accuracy | 47.19 ± 2.18% | **51.17 ± 0.11%** | **+3.98%p** |
| Macro F1 | 45.84 ± 2.45% | **50.08 ± 0.23%** | **+4.23%p** |
| Test Loss | 1.3669 ± 0.0404 | **1.2733 ± 0.0181** | -0.0936 |

세 Seed 모두 PyTorch가 Keras보다 높은 Accuracy와 Macro F1을 기록했다. 다만 이 결과를 framework 자체의 우월성으로 해석하지 않는다. 01–03은 native 조건의 개별 정렬을 탐색하며, 04부터는 동일 출발점의 수치 분기를 추적한다. 상세 결과는 [Experiment 00 README](experiments/00_baseline_final_cnn_3seed/README.md)에 기록했다.
<!-- BASELINE_RESULTS_END -->

### 14. Phase 1 Preliminary / Single-Factor Results

<!-- ABLATION_RESULTS_START -->
| Experiment | Aligned Variable | Keras F1 | PyTorch F1 | Signed Mean Gap | Mean Absolute Paired Gap | Status |
|---|---|---:|---:|---:|---:|---|
| 00 Baseline | None | 45.84% | 50.08% | 4.23%p | 4.23%p | Completed |
| 01 Input | Input Tensor | 46.46% | 48.66% | 2.20%p | 3.91%p | Completed |
| 02 Batch | Batch Order | 46.73% | 48.05% | 1.32%p | 3.17%p | Attempt 2 VALID |
| 03 Augmentation | Augmentation | 43.15% | 45.46% | 2.31%p | 3.65%p | VALID |
<!-- ABLATION_RESULTS_END -->

Signed Mean Gap은 `mean(PyTorch - Keras)`, Mean Absolute Paired Gap은 `mean(abs(PyTorch - Keras))`다. Seed별 우위 방향이 바뀌면 signed 값이 상쇄될 수 있으므로 두 값을 함께 본다. Experiment 01–03은 서로 독립적인 Baseline branch다.

Experiment 02 Attempt 1은 실제 order mismatch로 archive에 보존하고 공식 표에서 제외했다. Bug Fix 후 Attempt 2는 runtime에서 공통 수행한 모든 epoch의 전체 order hash가 일치해 `VALID` 판정을 받았으며, 위 Batch 행은 Attempt 2만 사용한다.

Experiment 03은 strict sample-ID runtime augmentation validation이 `VALID`다. 공통 수행 epoch의 sample별 flip·rotation parameter hash가 모두 일치했지만, Framework별 회전 연산 결과가 pixel-exact라는 뜻은 아니다. 위 03 행은 실제 유효한 3-Seed 결과를 사용한다.

### 15. Phase 2 Controlled Numerical Diagnostics

<!-- LAYER_RESULTS_START -->
Experiment 04의 [사전 검증과 첫 step 진단](experiments/04_common_initialization_controlled_training/README.md)은 완료됐다. 세 Seed 모두 canonical W0·입력·label과 Conv1 출력이 exact match이고, 첫 비영 수치 차이는 BN1 출력에서 관찰됐다. 이는 첫 *비영* 위치이지 실용적으로 유의한 divergence의 증거는 아니다. 30-epoch 학습과 최종 Test 결과는 **Training Pending**이다.

확장 진단은 gradient/update 분포, native Adam m/v와 NumPy reference Adam, 독립된 0–100-step weight trajectory를 기록했다. Canonical checkpoint의 model·optimizer state hash 및 fresh-model load round-trip도 통과했다. 이 진단은 Full Training 결과가 아니다.
<!-- LAYER_RESULTS_END -->

### 16. 주요 발견

Baseline 3-Seed에서 같은 방향의 Framework Gap이 반복됐다. Accuracy 평균 Gap은 PyTorch 기준 +3.98%p, Macro F1 평균 Gap은 +4.23%p였으며, Keras의 Seed 변동성이 더 컸다. 이는 추가 원인 분석을 수행할 근거지만 framework 자체의 인과 효과를 확정하지 않는다.

Experiment 01에서 augmentation 이전 deterministic input preprocessing을 동일화한 결과, Macro F1 Gap은 4.23%p에서 2.20%p로 48.02%, Accuracy Gap은 3.98%p에서 2.18%p로 45.25% 감소했다. 그러나 Keras Macro F1이 +0.61%p 상승한 것과 동시에 PyTorch Macro F1이 -1.42%p 하락했고, PyTorch의 Macro F1 표준편차가 0.23%p에서 3.72%p로 증가했다. Seed 42에서는 Keras가 PyTorch를 역전했다. 따라서 Input Tensor 처리는 Gap에 영향을 주는 요인이지만 단독 원인으로 보기는 어렵다. 상세 결과는 [Experiment 01 README](experiments/01_input_tensor_alignment/README.md)에 정리했다.

Experiment 02 Attempt 1은 Keras lifecycle bug로 무효 처리하고 archive에만 보존했다. 수정 후 Attempt 2는 runtime order validation을 통과했다. 공식 결과에서 Macro F1 Signed Mean Gap은 4.23%p에서 1.32%p로 68.87% 감소했고, Mean Absolute Paired Gap은 4.23%p에서 3.17%p로 25.18% 감소했다. Baseline의 세 Seed 모두 PyTorch 우위였던 방향도 Keras/PyTorch/Keras로 바뀌었다. 다만 Seed 123의 Macro F1 Gap은 +6.73%p였고 양쪽 Seed 변동성이 증가했으므로 Batch Order를 단독 원인으로 보지 않는다. 상세 결과는 [Experiment 02 README](experiments/02_batch_order_alignment/README.md)에 정리했다.

Experiment 03의 유효한 Augmentation Alignment에서는 Macro F1 Signed Mean Gap이 4.23→2.31%p로 45.52% 감소했으나 Mean Absolute Paired Gap은 4.23→3.65%p로 13.71% 감소했다. Seed 2026에서는 Keras가 역전했고 Seed 123에는 +6.40%p Gap이 남았다. 양쪽 Framework의 평균 Accuracy와 F1도 Baseline보다 낮아져 Gap 감소를 성능 향상으로 해석할 수 없다. Phase 1의 독립 branch 중 02가 가장 큰 Seed-level absolute F1 Gap 감소를 보였지만 이를 주요 원인으로 확정하지 않는다. [Experiment 03 README](experiments/03_augmentation_alignment/README.md)에 runtime·history·한계를 기록했다.

- Seed마다 우위가 바뀌면 framework 효과보다 stochastic variation이 큰 것으로 보고 H0를 기각하지 않는다.
- 세 Seed에서 같은 방향의 gap이 반복되면 H1을 검토할 재현성 근거로 사용한다.
- 특정 single-factor branch에서만 gap이 크게 줄면 해당 요소를 주요 원인 후보로 본다.
- 어느 branch에서도 줄지 않으면 변수 간 interaction 또는 framework 내부 연산 차이를 검토한다.
- 이 규칙은 자동으로 인과성을 확정하지 않으며, Seed별 분산과 layer 진단을 함께 해석한다.

### 17. Hypothesis Evaluation

<!-- HYPOTHESIS_RESULTS_START -->
| Hypothesis | Result | Evidence |
|---|---|---|
| H0 | 근거 약화 | 3 Seed 모두 같은 방향의 Accuracy/Macro F1 Gap이 관찰됨. 3 Seed만으로 통계적 기각을 주장하지 않음 |
| H1 | 추가 분석 근거 확보 | 반복 가능한 Framework Gap이 관찰되어 Phase 1 탐색과 Phase 2 통제 실험을 진행함 |
| H2 | 부분 지지 | Input, Batch Order, Augmentation 독립 branch 모두 Baseline 대비 F1 Gap 감소 방향. 03은 signed 45.52%, mean absolute paired 13.71% 감소했으나 양쪽 절대 성능 하락·Seed 분산 증가가 동반됨. 단독 원인으로 확정하지 않음 |
| H3 | 첫 step 진단 완료 / 장기 결과 Pending | W0·입력·Conv1 exact, BN1 출력부터 비영 차이 관찰. 30-epoch trajectory와 성능은 미실행 |
<!-- HYPOTHESIS_RESULTS_END -->

### 18. Conclusion

> 모든 실험 완료 후 작성 예정.

### 19. Limitations

- Seed가 3개뿐이라 stochastic distribution 추정력이 제한적이다.
- 하나의 dataset과 하나의 CNN architecture만 연구한다.
- 고정 70/15/15 split 하나만 사용하므로 다른 split에서의 변동은 측정하지 않는다.
- GPU reduction과 일부 kernel은 완전한 bitwise reproducibility를 보장하지 않는다.
- 설정을 맞춰도 framework 내부 연산과 Adam 구현이 bitwise identical하지 않을 수 있다.
- 증강 operator의 interpolation/fill/rounding을 완전히 등가화하기 어렵다.
- OFAT은 개별 효과를 확인하지만 둘 이상의 변수가 함께 작용하는 interaction effect를 직접 측정하지 못한다.
- Keras는 TensorFlow Metal, PyTorch는 MPS backend를 사용한다. 동일 Apple GPU에서도 backend 구현이 다르므로 `training_time_seconds`는 기록하되 성능 차이 원인의 주 판단 지표로 사용하지 않는다. 주요 지표는 Macro F1, Accuracy, Test Loss, Framework Gap과 Baseline 대비 Gap 변화다.

### 20. Future Work

Phase 2의 30-epoch 학습과 05–09 수치 분기 분석 후, 더 많은 Seed·architecture·dataset 및 변수 조합의 factorial 실험으로 확장할 수 있다.

### 21. Project Structure

```text
common/                 Phase 1 유틸리티 + Phase 2 canonical config/data/W0/training 유틸리티
experiments/00...03/    Phase 1 native/preliminary 실험 및 보존된 결과
experiments/04_common_initialization_controlled_training/  Phase 2 controlled baseline·preflight·first-step
experiments/05...09/    Phase 2 divergence roadmap placeholder
scripts/                 학습 없는 실행환경/GPU 검증
summary/                 전체 집계 및 README marker 갱신
emotion_dataset/         물리적 train/val/test × 8 classes (Git 제외)
*.ipynb                  수정하지 않은 reference notebooks
```

### 22. How to Run

macOS Apple Silicon에서 `.venv-metal`만 최종 실행환경으로 사용한다. Conda base가 활성화돼 있다면 먼저 비활성화한 뒤 환경을 활성화한다.

```bash
conda deactivate  # Conda를 사용 중인 경우만
source .venv-metal/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt`는 직접 사용하는 package만 읽기 쉽게 고정하고, `requirements-lock.txt`는 검증된 `.venv-metal`의 전체 transitive dependency를 보존한다. 환경을 완전히 동일하게 복원하려면 `python -m pip install -r requirements-lock.txt`를 사용한다.

환경 및 실제 GPU tensor 연산 검증:

```bash
.venv-metal/bin/python scripts/check_environment.py
.venv-metal/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
.venv-metal/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

정상 기대값은 TensorFlow `GPU:0` 감지와 PyTorch MPS `True`다.

```bash
.venv-metal/bin/python experiments/00_baseline_final_cnn_3seed/keras.py
.venv-metal/bin/python experiments/00_baseline_final_cnn_3seed/pytorch.py
```

00–03은 완료된 Phase 1 결과이며 재실행할 필요가 없다. 각 Phase 1 학습 파일은 `RUN_TRAINING`을 켤 때만 세 Seed를 순차 학습한다. 04는 별도 Phase 2 실험으로, 아래 사전 검증과 첫-step 진단은 전체 학습을 시작하지 않는다.

```bash
.venv-metal/bin/python experiments/00_baseline_final_cnn_3seed/compare.py
.venv-metal/bin/python experiments/01_input_tensor_alignment/keras.py
.venv-metal/bin/python experiments/01_input_tensor_alignment/pytorch.py
.venv-metal/bin/python experiments/01_input_tensor_alignment/compare.py
.venv-metal/bin/python summary/compare_all_experiments.py
.venv-metal/bin/python summary/update_readme.py
```

Phase 1 집계 도구는 00–03 결과만 다룬다. 04는 다음 경로를 사용한다. 현재 두 training script의 `RUN_TRAINING = False`는 그대로 유지하며, 실제 30-epoch 학습은 사용자가 각 파일에서 이를 `True`로 바꾼 후에만 시작된다.

```bash
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/first_step_trace.py
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/early_step_trace.py
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/checkpoint_roundtrip.py
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/preflight_validate.py
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/keras_controlled_train.py
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/pytorch_controlled_train.py
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/compare_controlled.py
```

사전 검증은 `CONTROLLED_PRECHECK=VALID`를 확인한다. 첫-step 진단은 full training result가 아니다. 학습 전 `compare_controlled.py`는 최종 결과를 생성하지 않고 Pending 상태를 알린다.
