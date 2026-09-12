# Keras vs PyTorch CNN Framework Difference Analysis

## 동일 CNN 아키텍처의 성능 차이 재현성 및 원인 분석

### 1. 프로젝트 개요

기존 어린이 얼굴 감정 분류 프로젝트에서 동일한 최종 CNN을 Keras와 PyTorch로 구현했을 때 관찰된 성능 차이가 재현되는지 확인하고, 그 원인을 통제 실험으로 추적하는 연구형 프로젝트다. 핵심은 “어느 framework가 항상 우수한가”가 아니라 기존 결과에 의문을 제기하고 재현성, 구현 차이, 수치 차이를 분리해 검증하는 데 있다. 기존 두 notebook은 수정하지 않은 연구 기록이며 신규 실험은 모두 `.py`로 작성했다.

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

기존 단일 Seed 실험에서는 Keras와 PyTorch 사이에 성능 차이가 관찰됐다. 그러나 deep learning 결과는 random initialization, data order, augmentation, optimizer와 framework 내부 구현의 영향을 함께 받으므로 단일 실행만으로 framework 효과라고 결론 내릴 수 없다. 이에 3-Seed 재현성 확인 → 원인 가설 설정 → 변수별 독립 통제 → layer-level numerical diagnosis 순서로 후속 연구를 설계했다.

```text
기존 CNN 구현 → 성능 차이 발견 → 단일 Seed 한계 인식
→ 3-Seed 재현성 실험 → 원인 가설 수립 → 변수별 독립 실험
→ Baseline 대비 gap 분석 → Layer-by-Layer 진단
→ 가설 평가 → 한계 및 후속 연구
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

1. 기존 gap이 Seed 변경 후에도 반복되는지 확인한다.
2. 반복된다면 framework별 구현 차이를 하나씩 독립 통제해 각 요소의 영향을 분석한다.
3. framework 자체 차이와 framework를 사용하는 과정에서 생긴 구현 차이를 구분한다.
4. 동일 입력/weight 조건에서 forward 또는 학습 차이가 최초로 커지는 layer를 추적한다.

### 5. Research Questions

- **RQ1:** 기존 성능 차이는 Seed 42, 123, 2026에서 일관되게 반복되는가?
- **RQ2:** Input Tensor, Batch Order, Augmentation, Initial Weight, Output/Loss, Adam, BatchNorm, EarlyStopping/LR Scheduler 중 무엇이 gap에 영향을 주는가?
- **RQ3:** 각 요소를 개별 정렬했을 때 baseline 대비 gap은 얼마나 증가하거나 감소하는가?
- **RQ4:** 동일 input/weight의 layer 비교에서 numerical output은 어디부터 의미 있게 달라지는가?

### 6. Hypotheses

- **H0 — Null:** 관찰된 차이는 stochastic variation 범위이며 3-Seed에서 일관된 framework gap이 나타나지 않을 것이다.
- **H1 — Reproducibility:** 주요 구조와 hyperparameter가 같아도 framework-specific 구현 차이로 여러 Seed에서 같은 방향의 gap이 반복될 수 있다.
- **H2 — Single-Factor Alignment:** 주요 원인인 구현 요소 하나를 정렬하면 baseline 대비 gap이 유의미하게 감소할 것이다.
- **H3 — Layer-Level Difference:** 동일 input과 initial weight에서도 내부 연산 차이가 있다면 특정 연산 이후 numerical difference가 점진적으로 증가할 것이다.

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

본 연구에서는 Keras와 PyTorch 사이의 구현 차이를 하나씩 독립적으로 동일화하는 **OFAT(One-Factor-at-a-Time) 기반 ablation-style controlled experiment**를 수행한다. 이는 model layer 제거 실험이 아니라 implementation factor alignment 연구다. 모든 성능 실험은 Keras 3회 + PyTorch 3회이며 주 지표는 Macro F1이다.

`Gap_seed = Keras Macro F1_seed − PyTorch Macro F1_seed`이며 Mean Gap과 Mean Absolute Gap을 함께 본다. 각 branch는 `current absolute gap − baseline absolute gap`을 Gap Change, 그 반대를 Gap Reduction으로 계산한다. Baseline gap이 거의 0이면 reduction rate는 N/A다.

### 10. 왜 누적 통제를 사용하지 않았는가?

누적 통제에서는 후반 변화가 방금 추가한 변수 때문인지 앞서 통제한 변수와의 interaction 때문인지 구분하기 어렵다. 따라서 각 실험은 언제나 Final CNN Baseline으로 돌아가 한 요소만 바꾸며 다른 branch의 정렬을 이어받지 않는다. OFAT은 개별 후보를 선별하는 대신 변수 간 interaction을 직접 측정하지 못한다.

### 11. Experiment Plan

| ID | Experiment | Baseline 대비 변경 변수 | Seeds | 목적 |
|---|---|---|---|---|
| 00 | Baseline 3-Seed | 없음 | 42, 123, 2026 | 기존 성능 차이 재현성 확인 |
| 01 | Input Tensor Alignment | Input Tensor | 42, 123, 2026 | 입력 처리 영향 확인 |
| 02 | Batch Order Alignment | Batch Order | 42, 123, 2026 | Mini-batch 순서 영향 확인 |
| 03 | Augmentation Alignment | Augmentation | 42, 123, 2026 | 증강 구현 영향 확인 |
| 04 | Initial Weight Alignment | Initial Weight | 42, 123, 2026 | 초기화 영향 확인 |
| 05 | Output / Loss Alignment | Output / Loss | 42, 123, 2026 | Loss 계산 방식 영향 확인 |
| 06 | Adam Alignment | Adam | 42, 123, 2026 | Optimizer 설정 영향 확인 |
| 07 | BatchNorm Alignment | BatchNorm | 42, 123, 2026 | BN 동작 차이 영향 확인 |
| 08 | EarlyStopping / LR Scheduler Alignment | Callback / Scheduler | 42, 123, 2026 | 종료/LR 정책 영향 확인 |
| 09 | Layer-by-Layer Analysis | Numerical Diagnostic | 42, 123, 2026 | Layer별 numerical difference 추적 |

### 12. Experiment Progress

| ID | Experiment | 구현 | 3-Seed 학습 | 분석 |
|---|---|---|---|---|
| ENV | `.venv-metal` 환경 | 완료 | - | GPU 연산 검증 완료 |
| DATA | Dataset physical split | 완료 | - | 70/15/15 검증 완료 |
| 00 | Baseline | 완료 | 대기 | GPU sanity 재검증 완료 |
| 01 | Input Tensor | 완료 (GPU sanity 통과) | 대기 | 대기 |
| 02 | Batch Order | 완료 (GPU sanity 통과) | 대기 | 대기 |
| 03 | Augmentation | 완료 (GPU sanity 통과) | 대기 | 대기 |
| 04 | Initial Weight | 완료 (GPU sanity 통과) | 대기 | 대기 |
| 05 | Output / Loss | 완료 (GPU sanity 통과) | 대기 | 대기 |
| 06 | Adam | 완료 (GPU sanity 통과) | 대기 | 대기 |
| 07 | BatchNorm | 완료 (GPU sanity 통과) | 대기 | 대기 |
| 08 | Callback / Scheduler | 완료 (GPU sanity 통과) | 대기 | 대기 |
| 09 | Layer-by-Layer | GPU 변환 점검 완료 | 대기 | 대기 |

### 13. Baseline 3-Seed Results

<!-- BASELINE_RESULTS_START -->
> 아직 학습하지 않음.
<!-- BASELINE_RESULTS_END -->

### 14. Single-Factor Ablation Results

<!-- ABLATION_RESULTS_START -->
| Experiment | Aligned Variable | Keras F1 | PyTorch F1 | Abs. Gap | Gap Reduction vs Baseline |
|---|---|---:|---:|---:|---:|
| Baseline | None | - | - | - | 0 |
| Input | Input Tensor | - | - | - | - |
| Batch | Batch Order | - | - | - | - |
| Augmentation | Augmentation | - | - | - | - |
| Weight | Initial Weight | - | - | - | - |
| Loss | Output / Loss | - | - | - | - |
| Adam | Adam | - | - | - | - |
| BatchNorm | BatchNorm | - | - | - | - |
| Callback | ES / LR Scheduler | - | - | - | - |
<!-- ABLATION_RESULTS_END -->

### 15. Layer-by-Layer Results

<!-- LAYER_RESULTS_START -->
> 아직 학습하지 않음. Initial/one-step 진단도 사용자가 명시적으로 실행한 뒤 갱신한다.
<!-- LAYER_RESULTS_END -->

### 16. 주요 발견

아직 결과가 없으므로 결론을 미리 정하지 않는다. 결과 생성 후 Seed variation, gap 방향, baseline 대비 reduction을 함께 근거로 작성한다.

- Seed마다 우위가 바뀌면 framework 효과보다 stochastic variation이 큰 것으로 보고 H0를 기각하지 않는다.
- 세 Seed에서 같은 방향의 gap이 반복되면 H1을 검토할 재현성 근거로 사용한다.
- 특정 single-factor branch에서만 gap이 크게 줄면 해당 요소를 주요 원인 후보로 본다.
- 어느 branch에서도 줄지 않으면 변수 간 interaction 또는 framework 내부 연산 차이를 검토한다.
- 이 규칙은 자동으로 인과성을 확정하지 않으며, Seed별 분산과 layer 진단을 함께 해석한다.

### 17. Hypothesis Evaluation

<!-- HYPOTHESIS_RESULTS_START -->
| Hypothesis | Result | Evidence |
|---|---|---|
| H0 | Pending | - |
| H1 | Pending | - |
| H2 | Pending | - |
| H3 | Pending | - |
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

유력 변수 조합의 factorial 실험, 모든 조건을 함께 맞춘 Full Alignment, 더 많은 Seed/architecture/dataset, 시계열 얼굴 표현을 위한 Conv3D 후속 연구로 확장할 수 있다.

### 21. Project Structure

```text
common/                 dataset/seed/result/alignment 유틸리티
experiments/00...08/    독립 성능 실험: keras.py, pytorch.py, compare.py, README.md
experiments/09.../      initial, one-step, trained, 3-seed layer 진단
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

학습할 때만 대상 `keras.py` 또는 `pytorch.py` 상단의 `RUN_TRAINING = False`를 `True`로 바꾼다. 먼저 00의 Keras/PyTorch를 실행하고 `compare.py`, 이후 01–08을 각각 실행한다. 한 파일은 세 Seed를 순차 학습한다.

```bash
.venv-metal/bin/python experiments/00_baseline_final_cnn_3seed/compare.py
.venv-metal/bin/python summary/compare_all_experiments.py
.venv-metal/bin/python summary/update_readme.py
```

결과가 없으면 비교 도구는 `Experiment results were not found. Run the training scripts first.`를 출력하며 가짜 값을 만들지 않는다. `update_readme.py`는 위 네 marker 내부만 바꾸고 연구 서술은 보존한다.
