# Final Research Report

## 1. Executive Summary

### 연구 제목

**엄격한 통제 조건에서 TensorFlow/Keras와 PyTorch 간 CNN 학습 수치 분기의 발생 위치와 누적 양상에 대한 실증 분석**

Working title: *Tracing Cross-Framework CNN Training Divergence between TensorFlow/Keras and PyTorch under Strictly Controlled Conditions*

본 연구는 동일한 CNN을 TensorFlow/Keras와 PyTorch에서 가능한 한 같은 상태와 조건으로 실행했을 때 최초 수치 차이가 어디에서 관찰되고, 그 차이가 forward·backward·optimizer·state를 통해 어떻게 전파되어 장기 training trajectory separation으로 이어지는지를 분석했다. 연구의 대상은 감정분류 또는 CIFAR-10 성능 자체가 아니라 **deep-learning software의 cross-framework numerical reproducibility**다.

Phase 1은 native framework 조건의 성능 gap을 관찰하고 strict control의 필요성을 확인한 pilot이었다. Phase 2는 canonical W0, exact input/order/augmentation tensor, aligned loss와 manual loop를 사용해 Native Adam, Native BatchNorm, accumulated state와 layer-local propagation을 차례로 통제했다. External Validation은 CPU, ResNet18, BN-free CNN과 CIFAR-10 full workload로 관찰 범위를 점검했다.

핵심 결과는 다음과 같다.

- Tested GPU+BN 조건의 Custom CNN/Young AffectNet, ResNet18/Young AffectNet, Custom CNN/CIFAR-10에서 first observed non-zero difference가 첫 BN batch-mean reduction에 반복적으로 나타났다.
- 이 위치는 보편적이지 않았다. CPU에서는 Conv1, BN-free GPU에서는 GAP로 이동했다.
- CommonAdam과 CommonBN은 각각 초기 optimizer/BN-state discrepancy를 크게 줄였지만 장기 parameter separation을 제거하지 못했다.
- Full-state re-synchronization 뒤 late checkpoint에서도 새로 생성된 one-step divergence는 free-running accumulated distance보다 여러 orders of magnitude 작았다.
- V4 CIFAR-10에서 E30 global weight relative L2는 3-Seed 평균 `1.1565`였지만 train accuracy는 Keras/PyTorch `98.296/98.359%`, final test accuracy는 `76.160/76.243%`로 가까웠다.

따라서 결과는 매 step마다 큰 framework error가 새로 생긴다는 설명보다, 작은 numerical difference가 weights·optimizer·BN state를 조금씩 갈라놓고 그 state가 이후 계산의 입력이 되는 **accumulated state-dependent feedback**과 더 잘 일치한다. 이는 완전한 causal proof가 아니며 framework superiority나 특정 연산의 unique root cause를 뜻하지 않는다.

## 2. Research Motivation

같은 architecture와 hyperparameter 이름을 사용해도 framework마다 initialization, tensor layout, shuffle, augmentation, loss reduction, optimizer arithmetic, BatchNorm state semantics와 backend kernel이 다를 수 있다. Native implementation 결과만 비교하면 관찰된 performance gap을 framework 자체, stochasticity 또는 서로 다른 training state 중 어디에 귀속해야 하는지 알기 어렵다.

이 프로젝트는 application benchmark를 넘어 다음 software-engineering 문제를 다뤘다.

- Cross-framework experiment를 비교 가능한 상태로 만드는 control boundary는 무엇인가?
- Bitwise identity가 처음 깨지는 semantic operation은 어디인가?
- 초기 FP-scale discrepancy와 장기 parameter-space separation을 어떻게 연결할 수 있는가?
- Parameter reproducibility, training fit과 generalization outcome은 어떤 관계인가?

## 3. Research Questions

### Primary Research Question

> 동일한 CNN을 TensorFlow/Keras와 PyTorch에서 가능한 한 동일한 상태와 학습 조건으로 실행했을 때, 최초의 수치 차이는 어디에서 발생하고, 그 작은 차이가 학습 과정에서 어떻게 전파·누적되어 장기적인 training trajectory separation으로 이어지는가?

### Sub-questions

1. Exact W0와 model-boundary input을 공유하면 forward/loss/gradient/update가 bitwise identical한가?
2. Native Adam implementation 차이를 제거하면 초기 및 장기 divergence가 얼마나 감소하는가?
3. Native BatchNorm semantics를 제거하면 BN state와 장기 trajectory가 얼마나 가까워지는가?
4. Late checkpoint의 큰 free-running distance가 exact full-state sync 뒤 한 step에서 다시 생성되는가?
5. First entry와 largest forward/backward/gradient/update group은 동일한가?
6. Entry point와 magnitude가 device, architecture, BN presence 및 independent workload에서도 유지되는가?
7. Parameter-space separation은 training fit과 generalization difference를 직접 예측하는가?

## 4. Initial Hypotheses

Phase 1의 초기 가설은 다음과 같이 운영됐다.

- **H0:** 관찰된 performance gap은 Seed variation만으로 설명될 수 있다.
- **H1:** Framework별 차이가 여러 Seed에서 반복될 수 있다.
- **H2:** Input preprocessing, batch order와 augmentation 차이가 native gap에 기여한다.
- **H3:** 외부 조건을 정렬한 뒤에도 framework execution 내부의 작은 numerical difference가 남아 반복 학습에서 누적될 수 있다.

Phase 1 결과는 H2의 부분적 영향을 보였지만 exact shared W0가 아니었으므로 causal comparison으로 사용하지 않았다. Phase 2와 External Validation이 H3를 엄격하게 검토하는 중심 연구가 됐다.

## 5. Experimental Environment

| Item | Recorded condition |
|---|---|
| Hardware | Apple M1 Pro 중심 단일 hardware |
| OS/architecture | macOS arm64 |
| Python | 3.10.3 |
| TensorFlow | 2.18.1 |
| TensorFlow Metal | 1.2.0(V2 manifest) |
| PyTorch | 2.14.0 |
| NumPy | 2.0.2 |
| GPU execution | TensorFlow `GPU:0`, PyTorch `mps:0` |
| MPS fallback | disabled |
| Main dtype | float32 |
| Seeds | 42, 123, 2026 |

V1만 TensorFlow CPU와 PyTorch CPU를 사용했다. CPU/GPU 모두 서로 다른 framework-native backend를 사용하므로 동일 low-level kernel comparison은 아니다. Version과 device의 source-of-truth는 각 experiment의 environment/device manifest다.

## 6. Common Architecture

Phase 2의 common custom CNN은 128×128 RGB input과 8-class logits를 사용한다.

```text
[Conv3×3(bias=False) → BN → ReLU → MaxPool2] × 4
channels: 32 → 64 → 128 → 256
→ Global Average Pooling
→ Dense 256→128 → ReLU
→ Dense 128→8 raw logits
```

Trainable parameter는 `422,824`개다. V4는 동일 body를 32×32 RGB에 적용하고 classifier만 128→10으로 바꿔 `423,082` trainable parameters와 960 BN running-state values를 갖는다. V2 ResNet18은 framework별 `11,180,616` trainable parameters다. 모든 paired model은 semantic parameter map을 통해 Conv HWIO↔OIHW, Dense IO↔OI를 canonical layout으로 비교했다.

## 7. Control Methodology

Phase 2가 공유하거나 단계적으로 통제한 조건은 다음과 같다.

| Category | Control |
|---|---|
| Data | 동일 physical split, class mapping, sample identity |
| Input | 동일 float32 tensor와 integer label |
| Initialization | Seed별 canonical W0 exact load |
| Order | 동일 batch membership/order/within-batch order |
| Augmentation | 동일 sample별 transformation 결과 tensor |
| Objective | Raw logits, mean sparse cross-entropy |
| Loop | Manual training loop |
| Schedule | Batch 32, 30 epochs, fixed lr 0.001 |
| Disabled | Scheduler, early stopping, dropout, weight decay, clipping |
| Optimizer | Exp05부터 CommonAdam |
| BatchNorm | Exp06부터 CommonBN |
| State | Exp07–08 weights, Adam m/v/step, BN state exact re-sync |

`relative L2 = ||K−P||₂ / max(||K||₂, ||P||₂, 1e−12)`를 주요 distance로 사용했다. Exact equality, max absolute difference, loss, accuracy, Macro F1, hash/manifest와 NaN/Inf gate를 함께 기록했다.

Common semantics는 mathematical alignment이며 backend execution의 bitwise identity를 강제하지 않는다. First observed entry도 low-level root cause와 동일하지 않다.

## 8. Phase 1 — Preliminary Native Framework Comparison

Phase 1은 exact shared W0가 아닌 native implementation comparison이다. 따라서 strict causal evidence가 아니라 control 필요성을 드러낸 pilot로 해석한다.

| Experiment | Intervention | Keras Macro F1 | PyTorch Macro F1 | Signed P−K gap | Mean absolute paired gap |
|---|---|---:|---:|---:|---:|
| 00 Baseline | None | 45.84% | 50.08% | +4.23%p | 4.23%p |
| 01 Input | Input tensor preprocessing | 46.46% | 48.66% | +2.20%p | 3.91%p |
| 02 Batch | Exact batch order, valid Attempt 2 | 46.73% | 48.05% | +1.32%p | 3.17%p |
| 03 Augmentation | Sample-ID aligned augmentation | 43.15% | 45.46% | +2.31%p | 3.65%p |

Gap은 모든 branch에서 감소했지만 framework별 절대 성능 하락, Seed crossover와 변동성 증가가 함께 나타났다. 어떤 single factor도 native gap의 단독 원인으로 확정할 수 없었고, 이 결과가 Phase 2 strict-control design의 근거가 됐다.

## 9. Phase 2 — Experiments 04–08

### 9.1 Experiment 04: Common Initialization & Controlled Training

동일 W0/input/label/batch/order/augmentation에서 input과 Conv1 output은 exact였고 세 Seed 모두 BN1에서 약 `1e−7` 규모의 first non-zero difference가 관찰됐다. Initial gradient 방향은 거의 같았지만 native Adam update에서 더 큰 relative discrepancy가 나타났다.

Epoch 30 global weight relative L2 평균은 `1.1155`였다. Train accuracy 평균은 Keras/PyTorch `64.47/64.31%`로 가까웠고 best-validation Macro F1은 `47.41/47.39%`로 거의 같았다. Seed/checkpoint별 generalization 방향은 달랐다.

### 9.2 Experiment 05: Common Adam Optimizer Control

Native Adam만 동일 CommonAdam으로 교체했다. First-step global update relative L2의 3-Seed 평균은 `0.025385 → 0.008974`, 즉 `64.65%` 감소했다. 그러나 Step 100 감소율은 `15.96%`, E30 감소율은 `1.82%`로 줄었고 E30 global relative L2는 `1.0952`였다.

Native Adam은 initial divergence amplification contributor로 관찰됐지만 long-term separation의 sole explanation은 아니었다.

### 9.3 Experiment 06: Common BatchNorm Control

CommonAdam을 유지하고 native BN을 explicit CommonBN semantics로 바꿨다. First-step all-BN running-variance relative L2 평균은 `3.15e−7 → 1.83e−9`, 약 `99.42%` 감소했고 gradient/update divergence도 평균 `82.79%/56.03%` 감소했다.

그러나 Step 100 global distance는 Exp05/06 `0.0533/0.0601`, E30은 `1.0952/1.0976`이었다. Native BN state-update semantics를 정렬해도 long-term separation은 유지됐다.

### 9.4 Experiment 07: Multi-Step Divergence & State Re-Synchronization

Existing Exp06 checkpoints의 weights, CommonAdam m/v/step, BN state, input과 labels를 exact re-sync한 뒤 one step을 실행했다. E30 free-running global weight distance는 `1.097571 → 1.097597`이었다. Exact sync 뒤 새 E30 post-weight divergence 평균은 K-anchor `3.78e−7`, P-anchor `8.67e−9`였다.

Late state가 매 step free-running distance와 같은 규모의 error를 새로 만들지는 않았다. 관찰은 accumulated state-dependent feedback과 강하게 consistent하지만 유일한 causal mechanism을 증명하지 않는다.

### 9.5 Experiment 08: Layer-by-Layer Training Trajectory Analysis

Initial shared 3개, E1 Seed42 K/P anchor, E30 Seed123/2026 K/P anchor의 총 9 case를 추적했다. 모든 case에서:

```text
input exact → Conv1 exact → BN1 input exact
→ first non-zero: BN1 batch mean
```

Maximum forward relative L2는 `2.46e−7–5.82e−7`이며 BN4 x-hat/output, ReLU3 또는 ReLU4에서 나타났다. Largest backward, parameter-gradient와 update group은 case/state/anchor에 따라 달랐다. 따라서 first entry와 largest divergence를 동일시할 수 없다.

### 9.6 Phase 2 answer

CommonAdam/CommonBN intervention은 native implementation의 early/local contribution을 분리했지만 long-term divergence를 제거하지 못했다. Full-state re-sync는 accumulated state와 newly generated one-step discrepancy를 구분했다. 종합적으로 작은 difference가 weights/optimizer/BN state에 반영되고 다음 step에서 다른 computational state를 만드는 반복 feedback 설명이 현재 결과와 가장 잘 맞는다.

## 10. External Validation V1–V4

### V1 — CPU-only Execution Validation

Custom CNN+CommonBN+CommonAdam의 같은 9 cases를 CPU에서 실행했다. Within-framework repeatability는 exact였고 9/9 first non-zero는 Conv1이었다. GPU Phase 2의 BN1 entry와 달라 first observed entry가 execution-stack dependent임을 보였다.

### V2 — ResNet18 + BatchNorm Architecture Validation

Semantic-equivalent ResNet18을 직접 구현했다. Trainable parameters는 양 framework `11,180,616`개였다. 3/3 Seeds first non-zero는 `stem.bn.batch_mean`, largest forward는 `stage4.block2.bn2.x_hat`이었다. First BN entry는 tested GPU stack의 더 깊은 BN architecture에서도 유지됐지만 downstream sensitivity는 architecture/Seed에 의존했다.

### V3 — BatchNorm-Free Custom CNN Validation

BN을 완전히 제거하고 Phase 2 Conv/Dense W0를 exact 재사용했다. 3/3 Seeds에서 input부터 Conv4/ReLU4/Pool4까지 exact였고 first non-zero는 GAP이었다. BN 제거로 entry는 이동했지만 divergence 자체는 사라지지 않았다.

### V4 — CIFAR-10 Independent Workload Validation

V4는 dataset-only intervention이 아니다. Image domain, train size, 128→32 resolution, 8→10 classes, classifier, BN/GAP reduction geometry, augmentation policy와 optimizer update 수가 함께 달라진다. Stage A는 initial one-step localization, Stage B는 independent workload의 long-term trajectory를 검증했다.

## 11. V4 Stage B

### 11.1 Configuration

| Item | Value |
|---|---|
| Dataset | CIFAR-10 |
| Split | 45,000 train / 5,000 validation / 10,000 official test |
| Per class | 4,500 train / 500 validation |
| Seeds/frameworks | 42, 123, 2026 × Keras/PyTorch = 6 runs |
| Epochs/batch | 30 / 32, `drop_last=False` |
| Steps | 1,407/epoch, 42,210 total |
| Preprocessing | uint8 → float32 / 255 |
| Augmentation | None |
| Optimizer | CommonAdam, lr .001, β1 .9, β2 .999, ε 1e−7 |
| BN | CommonBN, ε 1e−3, population variance, update rate .01 |
| Devices | TensorFlow GPU:0 / Metal, PyTorch mps:0, fallback disabled |

### 11.2 Validity

Persisted preflight/training summaries report all required gates as valid:

```text
V4_DATASET_INTEGRITY = VALID
V4_SPLIT_CHECK = VALID
V4_ARCHITECTURE_EQUIVALENCE = VALID
V4_PARAMETER_EQUIVALENCE = VALID
V4_GPU_DEVICE_CHECK = VALID
V4_FIXED_BATCH_CHECK = VALID
V4_SHARED_W0_CHECK = VALID
V4_PHASE2_SHARED_BODY_W0 = VALID
V4_RESYNC_PRECHECK = VALID
V4_TRACE_EQUIVALENCE = VALID
V4_GPU_REPEATABILITY = VALID
V4_TRAINING_CONFIG_EQUIVALENCE = VALID
V4_EPOCH_ORDER_CHECK = VALID
V4_STEP_CHECKPOINT_CHECK = VALID
V4_EPOCH_CHECKPOINT_CHECK = VALID
V4_TRAINING_RUNS_COMPLETE = TRUE (6/6)
V4_TEST_EVALUATION_COMPLETE = TRUE
NAN_INF_FOUND = FALSE
```

일부 preflight의 `new_full_training_executed=false`와 final analysis의 `training_executed_by_analysis=false`는 그 command 자체가 새 학습을 실행하지 않았다는 의미다. 별도 training histories/checkpoints/test metrics에 보존된 완료된 Stage B 6 runs를 부정하지 않는다.

### 11.3 Stage A

3 Seeds 모두 exact shared W0/fixed batch에서 input과 Conv1은 exact, first non-zero는 `bn1.batch_mean`, exact prefix length는 3, largest forward는 `bn4.x_hat`이었다.

| Seed | Largest forward relative L2 | First/largest backward | Largest parameter gradient | Largest update |
|---:|---:|---|---|---|
| 42 | 6.431674e−7 | logits / conv1 | conv1/kernel | conv4/kernel |
| 123 | 6.417550e−7 | logits / conv1 | conv1/kernel | conv4/kernel |
| 2026 | 6.402945e−7 | logits / conv1 | conv1/kernel | conv4/kernel |

## 12. Epoch-Matched Analysis

Epoch-matched는 같은 횟수의 V4 workload pass를 비교한다.

| Epoch | Seed 42 | Seed 123 | Seed 2026 | Mean |
|---:|---:|---:|---:|---:|
| 1 | 0.4194 | 0.4103 | 0.4325 | 0.4207 |
| 5 | 0.8521 | 0.8441 | 0.8761 | 0.8575 |
| 10 | 1.0196 | 1.0160 | 1.0403 | 1.0253 |
| 20 | 1.1165 | 1.1118 | 1.1339 | 1.1207 |
| 30 | 1.1522 | 1.1475 | 1.1697 | 1.1565 |

세 Seed는 유사한 parameter-space separation trajectory를 보였다. Relative L2가 1보다 크다는 것은 normalized parameter-vector distance이며 performance가 100% 이상 다르다는 뜻이 아니다.

E30 convolution group의 3-Seed mean은 Conv1 `0.2880` < Conv2 `0.7847` < Conv3 `1.0903` < Conv4 `1.2481`이었다. FC128은 `0.6037`, classifier는 `0.3894`였다. Deeper convolution group에서 더 큰 separation이 반복됐지만 depth나 Conv4를 원인으로 확정하지 않는다.

CommonBN에서도 E30 running-mean distance는 BN1–4 `0.3550/0.3282/0.4215/0.8358`이었다. 이미 달라진 weights가 activations와 batch statistics를 바꾸면 동일 BN formula도 서로 다른 running state를 만든다. Large BN-state divergence는 cause뿐 아니라 already-diverged state의 downstream consequence일 수 있다.

## 13. Step-Matched Analysis

Phase 2는 321 steps/epoch, V4는 1,407 steps/epoch이므로 E30끼리만 비교하면 update budget이 다르다. 다음은 같은 optimizer-update count에서의 descriptive comparison이다.

| Global step | V4 global rel L2 | Phase 2 global rel L2 |
|---:|---:|---:|
| 0 | 0.0000 | 0.0000 |
| 321 | 0.1677 | 0.1823 |
| 1,605 | 0.4521 | 0.5522 |
| 3,210 | 0.6345 | 0.7759 |
| 6,420 | 0.8337 | 0.9915 |
| 9,630 | 0.9416 | 1.0976 |

Independent workload에서도 같은 update budget에서 substantial parameter separation이 나타났고 V4 magnitude는 Phase 2보다 작았다. Exact divergence magnitude는 workload-dependent였다. Step matching은 update-count confound만 부분적으로 제어하며 두 workload를 causal-equivalent하게 만들지 않는다.

## 14. Training, Validation and Test Analysis

### Training and validation trajectories

V4 E30 train accuracy 평균은 Keras `98.296%`, PyTorch `98.359%`였다. 90개 paired epoch의 mean absolute train accuracy gap은 `0.136%p`, validation accuracy gap은 `2.687%p`였다. Mean absolute train/validation loss gap도 각각 `0.00296/0.19352`였다.

같은 E30 global distance `1.1565`와 거의 같은 training fit이 공존했다. Validation trajectory는 framework/Seed에 더 민감했지만 numerical divergence가 generalization difference를 직접 만들었다는 causal proof는 아니다.

### Final Epoch 30 test

| Seed | Keras loss / Acc / F1 | PyTorch loss / Acc / F1 | Accuracy P−K |
|---:|---|---|---:|
| 42 | 1.18360 / .7766 / .77899 | 1.37918 / .7546 / .75593 | −.0220 |
| 123 | 1.26051 / .7723 / .76857 | 1.26611 / .7756 / .77398 | +.0033 |
| 2026 | 1.61352 / .7359 / .73997 | 1.44391 / .7571 / .75717 | +.0212 |

| Aggregate | Keras | PyTorch | Signed P−K | Mean absolute paired gap |
|---|---:|---:|---:|---:|
| Accuracy | .76160 ± .02236 | .76243 ± .01147 | +.00083 | .01550 |
| Macro F1 | .76251 ± .02020 | .76236 ± .01008 | −.00015 | .01523 |
| Loss | 1.35254 ± .22926 | 1.36307 ± .08998 | +.01052 | .12360 |

Seed별 방향이 교차했고 final mean은 거의 같았다. Consistent framework-level final generalization superiority는 관찰되지 않았다.

### Best validation-loss checkpoint

| Seed | Keras best epoch | PyTorch best epoch | Accuracy P−K | Macro F1 P−K |
|---:|---:|---:|---:|---:|
| 42 | 5 | 5 | −.0027 | −.00938 |
| 123 | 4 | 6 | +.0316 | +.03455 |
| 2026 | 3 | 9 | +.0562 | +.06325 |

Accuracy 평균은 Keras `.73030`, PyTorch `.75867`, signed P−K `.02837`이었다. Macro F1 signed P−K는 `.02948`이었다. 그러나 `n=3`, Seed42의 반대 방향과 framework별 서로 다른 selection epoch 때문에 superiority claim을 하지 않는다. 적절한 해석은 trajectory separation 뒤 validation-loss optimum timing이 framework- and seed-dependent였다는 것이다.

## 15. Integrated Interpretation

1. Exact W0/input/order/loss semantics를 공유해도 cross-framework bitwise identity는 보장되지 않았다.
2. Tested GPU+BN의 세 model/workload 조합에서 first BN batch-mean entry가 반복됐다.
3. CPU에서는 Conv1, BN-free GPU에서는 GAP로 이동해 entry point가 execution context와 operation composition에 의존했다.
4. FP-scale first difference는 forward/backward/update로 전달됐다.
5. CommonAdam은 early amplification을 줄였지만 long-term separation을 제거하지 못했다.
6. CommonBN은 native BN state-update discrepancy를 줄였지만 long-term separation을 제거하지 못했다.
7. Full-state re-sync 후 late one-step divergence가 작아진 결과는 continually regenerated large error보다 accumulated state-dependent feedback과 더 잘 맞았다.
8. CIFAR-10 independent workload에서도 same-update-budget 및 full 30-epoch parameter separation이 재현됐다.
9. Exact magnitude와 validation/generalization path는 workload와 Seed에 따라 달랐다.
10. Large parameter-space separation은 similar training fit과 nearly equal final mean performance와 공존할 수 있었다.

따라서 `parameter reproducibility != functional/training-fit equivalence`다. 동시에 비슷한 성능이 parameter trajectory reproducibility를 의미하지도 않는다.

## 16. Threats to Validity

- 주요 Seed가 3개여서 population distribution과 statistical significance를 추정하기 어렵다.
- Apple M1 Pro 단일 hardware와 TensorFlow Metal/PyTorch MPS가 중심이며 CUDA/NVIDIA를 검증하지 않았다.
- CPU validation도 TensorFlow/PyTorch의 서로 다른 native backend를 사용한다.
- Low-level reduction tree/order, kernel, graph fusion과 compiler lowering을 직접 계측하지 않았다.
- First non-zero는 bitwise identity break이며 practical significance와 동일하지 않다.
- V4는 data, resolution, classes, geometry, augmentation와 update budget이 함께 바뀌는 independent workload이며 pure dataset-only intervention이 아니다.
- Young AffectNet HQ는 age-transformed derivative workload다.
- Dataset image redistribution 권한 문제로 raw images를 repository에 공개하지 않는다.
- Phase 1은 exact shared W0가 아니므로 causal framework comparison에 사용할 수 없다.
- 본 연구는 framework superiority를 평가하도록 설계되지 않았다.
- Fixed batch/selected checkpoints의 layer diagnostic은 전체 training distribution을 완전히 대표하지 않는다.

## 17. Claim Boundaries

### Supported wording

- first observed numerical entry
- cross-framework numerical divergence
- parameter-space/trajectory separation
- controlled intervention result
- state-dependent accumulation과 consistent
- execution-stack dependent entry point
- workload-dependent magnitude
- seed-dependent generalization outcome
- no consistent framework superiority

### Unsupported wording

- BatchNorm, GAP, reduction, Metal, MPS 또는 Conv4가 root cause다.
- TensorFlow/Keras 또는 PyTorch가 더 안정적·정확·우수하다.
- State-feedback mechanism을 causally proven했다.
- 모든 CPU/GPU, architecture, network와 framework에 일반화된다.
- CIFAR-10 dataset 자체가 divergence를 만들었다.

## 18. Expected Paper Contributions

1. Exact W0/input/order/loss semantics를 포함하는 strict cross-framework controlled training pipeline.
2. CommonAdam/CommonBN intervention을 통한 native optimizer/normalization contribution 분리.
3. Full-state re-synchronization을 통한 newly generated one-step divergence와 accumulated state divergence의 구분.
4. Layer-wise forward/backward/gradient/update tracing에 의한 first-entry와 propagation localization.
5. CPU/ResNet18/BN-free/CIFAR-10 external-validity matrix.
6. Independent workload full training을 통한 long-term parameter trajectory separation 재검증.
7. Manifest/hash/validity gate/exact state mapping 중심의 reproducibility artifact package.

1차 투고 목표는 **Journal of KIISE (JOK, 정보과학회논문지)**다. AI application 성능 논문이 아니라 Software Engineering for AI, cross-framework numerical reproducibility와 empirical deep-learning software analysis 관점으로 구성한다. 국제 venue는 future possibility이며 현재 확정하지 않는다.

## 19. Final Status

| Block | Status |
|---|---|
| Phase 1 | Preliminary / Completed |
| Phase 2 | Completed / VALID |
| External Validation V1–V4 | Completed |
| Overall | **Research experiments completed** |

Experimental phase는 closed다. 이 repository에는 추가 experiment가 계획되어 있지 않다. CUDA/NVIDIA replication, operator microbenchmark, 추가 architecture/framework와 larger-seed study는 현재 scope 밖의 Future Work일 뿐 미완료 TODO가 아니다.

## 20. Artifact Index

### Project-level documents

- [Root README](../README.md)
- [Phase 2 strict-controlled conclusion](../PHASE2_STRICT_CONTROLLED_CONCLUSION.md)
- [Final results summary](FINAL_RESULTS_SUMMARY.md)
- [Experiment status](EXPERIMENT_STATUS.md)
- [External validation overview](../experiments/external_validation/README.md)

### Phase 1

- [Experiment 00](../experiments/00_baseline_final_cnn_3seed/README.md)
- [Experiment 01](../experiments/01_input_tensor_alignment/README.md)
- [Experiment 02](../experiments/02_batch_order_alignment/README.md)
- [Experiment 03](../experiments/03_augmentation_alignment/README.md)

### Phase 2

- [Experiment 04](../experiments/04_common_initialization_controlled_training/README.md)
- [Experiment 05](../experiments/05_gradient_optimizer_divergence/README.md)
- [Experiment 06](../experiments/06_batchnorm_state_divergence/README.md)
- [Experiment 07](../experiments/07_multistep_state_resynchronization/README.md)
- [Experiment 08](../experiments/08_layer_by_layer_trajectory/README.md)

### External Validation

- [V1 CPU-only](../experiments/external_validation/V1_cpu_only/README.md)
- [V2 ResNet18 + BN](../experiments/external_validation/V2_resnet18_bn/README.md)
- [V3 BN-free CNN](../experiments/external_validation/V3_bn_free_cnn/README.md)
- [V4 CIFAR-10](../experiments/external_validation/V4_cifar10/README.md)

### V4 source-of-truth artifacts

- [Diagnostic summary](../experiments/external_validation/V4_cifar10/results/summaries/diagnostic_summary.json)
- [Stage A cases](../experiments/external_validation/V4_cifar10/results/summaries/stage_a_case_summary.csv)
- [Epoch-matched trajectory](../experiments/external_validation/V4_cifar10/results/summaries/epoch_matched_trajectory_summary.csv)
- [Step-matched trajectory](../experiments/external_validation/V4_cifar10/results/summaries/step_matched_trajectory_summary.csv)
- [Training curves](../experiments/external_validation/V4_cifar10/results/summaries/training_curve_comparison.csv)
- [Final test summary](../experiments/external_validation/V4_cifar10/results/summaries/test_final_summary.csv)
- [Best-validation test summary](../experiments/external_validation/V4_cifar10/results/summaries/test_best_val_summary.csv)
- [Training validity](../experiments/external_validation/V4_cifar10/results/preflight/training_validity.json)
- [Environment manifest](../experiments/external_validation/V4_cifar10/results/manifests/environment.json)
- [Dataset split manifest](../experiments/external_validation/V4_cifar10/results/manifests/cifar10_split_manifest.json)
- [Checkpoint manifest](../experiments/external_validation/V4_cifar10/results/snapshots/checkpoint_manifest.csv)

Raw datasets and large binary checkpoints are not publication artifacts. Existing CSV, JSON, manifest, histories and selected summaries remain the numerical source-of-truth.
