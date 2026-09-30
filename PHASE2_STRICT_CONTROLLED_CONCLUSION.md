# Phase 2 - Strict Controlled Framework Comparison

> 이 문서는 완료된 Phase 2의 scoped conclusion이다. 이후 External Validation V1–V4도 완료됐으며 프로젝트 전체 결론은 [Final Research Report](docs/FINAL_RESEARCH_REPORT.md)를 따른다.

## 1. Research Question

> When the same CNN is trained in Keras and PyTorch under increasingly controlled conditions, where does the first numerical difference appear, how does it propagate, and what explains the large long-term trajectory divergence?

Phase 2는 TensorFlow/Keras Metal과 PyTorch MPS에서 동일 CNN을 가능한 한 같은 조건으로 시작시킨 뒤, model boundary부터 forward, loss, gradient, optimizer update, BatchNorm state와 multi-step trajectory를 순서대로 추적했다. 목적은 Framework 우열을 정하는 것이 아니라 cross-framework numerical divergence가 처음 관찰되는 위치와 장기 parameter-space separation으로 이어지는 양상을 분리하는 것이었다.

## 2. Why Phase 2 Was Needed

Experiments 00–03의 Phase 1은 input, batch order와 augmentation을 독립적으로 정렬한 exploratory OFAT였다. 이 실험들은 성능 Gap이 개별 조건과 Seed에 민감함을 보였지만, Framework-native 차이가 여러 곳에 동시에 남아 first numerical difference나 장기 trajectory separation을 직접 설명할 수 없었다.

Phase 2는 공통 초기 상태와 training schedule을 고정하고, native Adam, native BatchNorm, accumulated state와 layer-local computation을 차례로 격리했다. Accuracy/F1보다 `W0 → activation → loss → gradient → update → state → trajectory`를 우선 분석했다.

## 3. Controlled Conditions

Experiments 04–08이 공유하거나 단계적으로 유지한 핵심 조건은 다음과 같다.

| Category | Controlled condition |
|---|---|
| Model | 동일 CNN architecture, 422,824 trainable parameters |
| Initialization | Seed 42/123/2026별 canonical W0 |
| Data | 동일 physical 70/15/15 split, class mapping, RGB 128×128 |
| Input | 동일 canonical float32 tensor와 integer label |
| Training order | 동일 persisted batch order와 augmentation parameter schedule |
| Objective | Raw logits, mean sparse cross-entropy |
| Schedule | Batch 32, 30 epochs, epoch당 321 batches, 총 9,630 updates |
| Optimizer control | Experiment 05부터 동일 CommonAdam NumPy float32 implementation |
| BatchNorm control | Experiment 06부터 동일 Common BN mathematical semantics |
| State control | Experiment 07–08에서 weights, Adam m/v/step, BN state exact re-synchronization |
| Numerical scope | float32, TensorFlow/Keras Metal과 PyTorch MPS framework-native forward/backward |

동일 mathematical rule과 동일 input은 서로 다른 execution stack에서 bitwise-identical reduction이나 autograd 결과를 보장하지 않는다. 이 남은 차이가 Phase 2의 분석 대상이다.

## 4. Experiment Sequence

### Experiment 04 — Common Initialization & Controlled Training

동일 W0, input, label, batch/order와 augmentation에서 controlled baseline을 확립했다. Input과 Conv1 output은 exact였고, 첫 비영 차이는 BN1에서 floating-point 규모로 관찰됐다. 초기 gradient 방향은 거의 같았지만 non-zero difference가 있었고 native Adam update에서 더 큰 relative divergence가 나타났다.

반복 학습 후 Epoch 30 global weight relative L2 평균은 `1.1155`까지 증가했다. 반면 Epoch 30 train accuracy 평균은 Keras/PyTorch `64.47/64.31%`로 유사했다. Fixed Epoch 30 성능은 Seed 2026 PyTorch의 late-stage degradation 영향을 크게 받았고, best-validation Macro F1 평균은 `47.41/47.39%`로 거의 같았다.

### Experiment 05 — Common Adam Optimizer Control

Native Adam만 동일 CommonAdam으로 교체했다. First-step update relative L2는 3-Seed 평균 `0.025385→0.008974`, 즉 `64.65%` 감소했다. 그러나 감소율은 Step 100 `15.96%`, Epoch 30 `1.82%`로 작아졌고 Epoch 30 global relative L2는 `1.0952`로 Experiment 04의 `1.1155`와 비슷했다.

Native Adam implementation은 초기 divergence의 amplification contributor로 관찰됐지만 장기 trajectory divergence를 충분히 설명하지 못했다.

### Experiment 06 — Common BatchNorm Control

Common Adam을 유지하고 native BN을 explicit Common BN semantics로 교체했다. First-step all-BN running-variance relative L2 평균은 `3.15e-7→1.83e-9`, 약 `99.42%` 감소했다. First-step gradient/update divergence도 평균 `82.79%/56.03%` 감소했다.

그러나 감소는 몇 step 뒤 유지되지 않았다. Step 100 global weight relative L2는 Experiment 05 `.0533`, Experiment 06 `.0601`이었고 Epoch 30은 `1.0952/1.0976`으로 사실상 같았다. Native BatchNorm semantics만으로 장기 separation을 설명할 수 없었다. Common BN 이후 장기 BN state가 다시 분리된 현상은 원인뿐 아니라 이미 달라진 trajectory의 downstream consequence를 함께 반영할 수 있다.

### Experiment 07 — Multi-Step Divergence & State Re-Synchronization

Experiment 06 checkpoint를 read-only로 재사용해 free-running one step과 exact full-state re-synchronized one step을 비교했다. Epoch 30 free-running global weight divergence는 one step 전후 `1.097571→1.097597`이었다.

반면 full-state sync 뒤 새로 생성된 Epoch 30 post-weight divergence 평균은 K-anchor `3.78e-7`, P-anchor `8.67e-9`였다. Later checkpoint로 갈수록 synchronized one-step discrepancy가 일관되게 증가하지도 않았다. 즉 free-running의 큰 separation은 전체 state를 맞춘 단일 step에서 같은 규모로 재생성되지 않았다. 이는 accumulated state-dependent feedback과 강하게 consistent with하지만 유일한 causal mechanism을 확정하지 않는다.

### Experiment 08 — Layer-by-Layer Training Trajectory Analysis

Experiment 07의 synchronized one step 내부를 9개 selected case에서 추적했다. Initial shared-anchor 3개, Epoch 1 low-divergence control Seed 42의 K/P anchor, Epoch 30 Seed 123/2026의 K/P anchor를 사용했다.

모든 case에서 input과 Conv1 output은 exact였고, first observed non-zero numerical difference는 BN1 batch-mean reduction에서 반복됐다. Case별 maximum forward relative L2는 `2.46e-7–5.82e-7`이었으며 BN4 x̂/output, ReLU3 또는 ReLU4 등 deeper stage에서 나타났다. Backward, parameter-gradient와 Common Adam update의 최대 위치는 case와 anchor에 따라 달라졌다.

Experiment 08은 `LAYER_TRACE_PRECHECK=VALID`, `TRACE_EQUIVALENCE=VALID`, `SELECTED_CASES_COMPLETE=TRUE`이며 Experiment 07과 비교한 45개 global metric이 모두 exact였다. NaN/Inf와 새 Full Training은 없었다.

## 5. Main Findings

1. 동일 W0와 model-boundary input만으로 cross-framework training을 bitwise identical하게 만들 수는 없었다.
2. Strict controlled trace에서 재현된 first numerical entry point는 BN1 batch-mean reduction이었다.
3. Forward difference는 `1e-7` 규모로 작게 유지됐지만 gradient와 update에서 case-dependent local amplification이 관찰됐다.
4. Native Adam과 native BatchNorm semantics는 초기 difference의 일부를 증폭하거나 state discrepancy를 만들었지만 장기 separation을 각각 단독으로 설명하지 못했다.
5. Full-state re-synchronization 뒤 새로 생성된 one-step separation은 free-running accumulated separation보다 여러 orders of magnitude 작았다.
6. Late-stage K/P anchor에 따라 최대 backward/update 위치와 global sensitivity가 달라져 parameter-space location sensitivity가 관찰됐다.
7. 큰 parameter-space separation은 비슷한 training-set fitting과 공존했으며 validation/test 성능의 크기나 방향을 직접 예측하지 못했다.

## 6. Where the First Difference Appeared

Experiment 04는 동일 input과 Conv1 이후 BN1에서 첫 비영 차이를 관찰했다. Experiment 06의 explicit Common BN first-step trace는 BN1의 batch mean/variance와 output에 약 `1e-8–1e-7` relative difference가 남음을 보였다. Experiment 08은 이 위치를 더 세분화했다.

9개 synchronized case 전부에서 다음 순서가 반복됐다.

```text
Input                 exact
Conv1 output          exact
BN1 input             exact
BN1 batch mean        first observed non-zero numerical difference
```

따라서 BN1 batch-mean reduction은 테스트한 환경에서 **reproducible first observed numerical entry point**다. 이는 mathematically aligned BatchNorm semantics가 서로 다른 execution stack의 reduction을 bitwise identical하게 만들지는 않음을 보여준다. Low-level reduction order, kernel implementation 또는 backend mechanism 자체는 직접 격리하지 않았으므로 BN1을 unique root cause로 해석하지 않는다.

## 7. How the Difference Propagated

Experiment 08에서 BN1 batch mean의 first non-zero difference는 normalization과 후속 Conv/BN/ReLU/Pool block으로 전달됐다. Forward maximum relative L2는 `2.46e-7–5.82e-7`이었지만 stage를 따라 단조 증가하지 않았다. Largest forward stage도 BN4 x̂, BN4 output, ReLU3와 ReLU4로 달랐다.

Backward maximum은 Conv1, BN1, BN2 또는 BN3였고, largest parameter-gradient group은 Conv1 kernel, BN1 beta, BN2 beta, Conv2 kernel 또는 Conv4 kernel이었다. 단일 backward layer나 parameter group이 모든 case의 dominant amplification point가 되지는 않았다.

## 8. What Adam Explained

Experiment 04→05에서 Common Adam은 first-step update divergence를 평균 `64.65%` 줄였다. 이는 native Adam implementation이 early local amplification에 기여했음을 시사한다. Experiment 08 initial case에서도 일부 parameter group의 Common Adam update divergence가 gradient divergence보다 크게 변환됐다.

그러나 Epoch 30 global divergence 감소는 `1.82%`뿐이었고, Experiment 08의 largest update group도 Conv1/2/3 kernel 또는 BN1 beta로 case마다 달랐다. Optimizer transformation은 local difference를 증폭할 수 있지만 하나의 Adam implementation이나 하나의 parameter group이 장기 separation을 충분히 설명하지 않는다.

## 9. What BatchNorm Explained

Common BN은 native running-variance update discrepancy를 거의 제거하고 first-step gradient/update divergence를 평균적으로 줄였다. 그럼에도 BN1 reduction의 작은 difference는 남았고 long-term parameter divergence는 감소하지 않았다.

BatchNorm 관련 관찰은 두 층으로 구분해야 한다.

- Native BN semantics 차이는 초기 running-state discrepancy에 기여했다.
- Common BN에서도 reduction-level non-zero difference가 남았고, 이미 갈라진 trajectory에서는 BN state도 다시 크게 분리됐다.

따라서 native BN semantics는 제한된 역할을 설명하지만 장기 divergence의 단독 설명은 아니다. Experiment 08은 BN1 batch-mean reduction을 first observed entry point로 식별했을 뿐 BatchNorm을 unique root cause로 확정하지 않는다.

## 10. What State Re-Synchronization Revealed

Experiment 07의 가장 강한 관찰은 accumulated free-running state와 synchronized one-step generation을 분리한 것이다. Epoch 30 accumulated weight divergence는 약 `1.098`이었지만 exact sync 후 새로 생긴 post-weight divergence는 평균 `1e-9–1e-7` 규모였다.

이는 later checkpoint가 매 step마다 free-running separation과 같은 규모의 divergence를 새로 생성한다는 설명과 맞지 않는다. 현재 결과는 이전 step의 작은 weights/Adam/BN-state 차이가 다음 step의 input state가 되고, 그 상태에서 activation/gradient/update가 다시 달라지는 accumulated state-dependent feedback과 더 잘 맞는다.

## 11. Layer-by-Layer Findings

Experiment 08의 공식 case summary는 다음과 같다.

| Case | First forward difference | Largest forward | Largest backward | Largest parameter-gradient | Largest update |
|---|---|---|---|---|---|
| Initial 42 / shared | BN1 batch mean | BN4 x̂ | Conv1 | Conv1 kernel | Conv2 kernel |
| Initial 123 / shared | BN1 batch mean | ReLU4 | BN1 | Conv1 kernel | Conv2 kernel |
| Initial 2026 / shared | BN1 batch mean | ReLU4 | BN1 | BN2 beta | Conv3 kernel |
| Epoch 1, 42 / K | BN1 batch mean | BN4 output | BN2 | BN1 beta | BN1 beta |
| Epoch 1, 42 / P | BN1 batch mean | BN4 output | BN2 | BN1 beta | BN1 beta |
| Epoch 30, 123 / K | BN1 batch mean | BN4 x̂ | BN3 | Conv2 kernel | Conv3 kernel |
| Epoch 30, 123 / P | BN1 batch mean | ReLU3 | BN1 | Conv1 kernel | Conv1 kernel |
| Epoch 30, 2026 / K | BN1 batch mean | ReLU3 | BN2 | Conv4 kernel | Conv1 kernel |
| Epoch 30, 2026 / P | BN1 batch mean | BN4 x̂ | BN2 | Conv1 kernel | Conv1 kernel |

First forward entry point만 9개 case에서 반복됐고 이후 largest stage/group은 달라졌다. 특히 Epoch 30 Seed 123/2026의 K/P anchor 결과는 one-step numerical sensitivity가 step을 시작한 parameter-space location에 부분적으로 의존한다는 해석과 consistent with하다.

## 12. Parameter Divergence vs Training Performance

Experiments 04–06에서 Epoch 30 global weight relative L2는 약 `1.10`까지 증가했다. 동시에 Epoch 30 train accuracy 평균은 다음처럼 매우 가까웠다.

| Experiment | Keras train accuracy | PyTorch train accuracy |
|---|---:|---:|
| 04 Native Adam / Native BN | 64.47% | 64.31% |
| 05 Common Adam | 64.62% | 64.43% |
| 06 Common Adam / Common BN | 64.55% | 64.33% |

따라서 **large parameter-space separation does not necessarily imply large differences in training-set fitting**. 서로 다른 weight trajectory가 유사한 training objective와 accuracy에 도달할 수 있다.

Parameter distance와 performance gap도 단조 관계가 아니었다. Experiment 04에서 Epoch 30 global distance가 Seed별로 비슷했지만 Macro F1 paired gap의 크기와 방향은 크게 달랐다. Parameter-space divergence를 Framework performance superiority로 해석해서는 안 된다.

## 13. Generalization Behavior

Fixed Epoch 30과 best-validation checkpoint는 같은 질문이 아니다. Fixed epoch는 같은 update count를 비교하고, best-validation은 각 Framework trajectory에서 generalization optimum timing이 달랐음을 반영한다.

- Experiment 04 best-validation Macro F1 평균은 Keras/PyTorch `47.41/47.39%`로 거의 같았지만 Seed별 방향은 달랐다.
- Experiment 05에서도 Seed crossover가 있었고 Framework별 best epoch가 모두 달랐다.
- Experiment 06의 best-validation 평균은 이 조건에서 PyTorch 값이 높았지만 fixed Epoch 30의 Seed별 방향은 교차했다.
- Experiment 04 Seed 2026은 비슷한 train accuracy와 큰 late-stage validation/test degradation이 함께 나타났다.

Generalization outcome은 Seed와 checkpoint selection에 의존했다. 이 exploratory 3-seed study는 Framework-level performance superiority를 확립하도록 설계되지 않았다.

## 14. Answer to the Phase 2 Research Question

테스트한 TensorFlow/Keras Metal과 PyTorch MPS 환경에서 두 구현은 동일 model parameters와 identical model-boundary input에서 시작해도 floating-point 수준의 작은 numerical difference를 만들 수 있었다. Strict controlled trace에서 반복적으로 관찰된 최초의 비영 차이는 BN1 batch-mean reduction에 있었다.

Native Adam implementation은 초기 difference 일부를 증폭했지만 Common optimizer로 교체해도 장기 trajectory separation은 제거되지 않았다. Native BatchNorm semantics를 Common BN으로 교체하면 direct running-state update discrepancy 대부분은 제거됐지만 장기 divergence는 유지됐다.

Full-state re-synchronization은 later checkpoint가 free-running에서 관찰된 큰 separation을 한 step에 같은 규모로 계속 재생성하지 않음을 보였다. Layer-wise trace는 첫 numerical entry point가 반복 가능했지만 후속 forward, backward와 optimizer local amplification이 현재 model state와 anchor에 따라 달라짐을 보였다.

전체 결과는 하나의 optimizer, BatchNorm implementation 또는 고정 layer가 divergence 전체를 설명한다는 해석보다, 매우 작은 numerical difference가 weights, optimizer state와 BN state에 반영되고 이후 step에서 반복적으로 전달되는 **accumulated state-dependent feedback**과 가장 잘 맞는다. 이는 관찰과 consistent with한 종합 해석이며 proven causal chain은 아니다.

```text
Exact W0 + exact input
        ↓
Conv1 output exact
        ↓
BN1 batch-mean reduction
first observed floating-point-scale difference
        ↓
small forward difference
        ↓
gradient and optimizer-update difference
        ↓
slightly different weights / Adam state / BN state
        ↓
next step begins from different states
        ↓
state-dependent activation / gradient / update
        ↓
repeated accumulation
        ↓
large long-term parameter-space separation
```

## 15. What Phase 2 Answered—and Did Not Answer

Phase 2가 보여준 것:

- 테스트 환경에서 반복 가능한 first observed numerical entry point
- Native Adam의 early amplification contribution과 그 설명 범위
- Native BN semantics의 direct state-update contribution과 그 설명 범위
- 9,630 updates 동안의 large trajectory divergence
- Full-state re-synchronization 뒤 one-step divergence의 작은 규모
- Parameter-space location에 따른 layer/update sensitivity
- Parameter divergence와 training/generalization performance의 비직접적 관계

Phase 2가 증명하지 않은 것:

- 정확한 low-level backend kernel mechanism
- Reduction order가 유일한 원인이라는 주장
- Metal 또는 MPS가 원인이라는 주장
- TensorFlow/Keras 또는 PyTorch 계산의 상대적 정확성
- 특정 layer가 전체 divergence의 unique root cause라는 주장
- Framework-level performance superiority
- Statistical significance

## 16. Limitations

- Seeds는 42, 123, 2026 세 개뿐이다.
- 하나의 CNN architecture와 BatchNorm 포함 구조만 사용했다.
- 하나의 emotion dataset/domain과 physical split만 사용했다.
- Apple M1 Pro의 TensorFlow Metal과 PyTorch MPS execution stack에 한정된다.
- 모든 주요 계산은 float32다.
- Experiment 07–08은 하나의 fixed diagnostic batch와 selected checkpoint/case를 사용했다.
- Low-level kernel, reduction tree/order와 compiler operation을 직접 instrumentation하지 않았다.
- Result는 exploratory 3-seed evidence이며 statistical significance를 주장할 수 없다.

따라서 이 결론을 다른 hardware, backend, precision, architecture 또는 dataset으로 일반화하지 않는다.

## 17. Future Work / Optional Extensions

새 Experiment를 자동으로 만들지는 않는다. 가능한 후속 방향은 다음과 같다.

- CPU-only controlled comparison과 동일 CPU execution path 비교
- Selected diagnostic의 float64 반복
- Deterministic reduction과 reduction-order profiling
- BatchNorm이 없는 architecture 또는 alternative normalization 비교
- 다른 CNN architecture와 더 깊거나 얕은 network 비교
- Seed 수 확대와 confidence interval/statistical analysis
- 다른 dataset/domain과 split에서 재현성 확인
- Low-level TensorFlow/PyTorch kernel 및 reduction profiling

## 18. Phase 2 Status

| Experiment | Status |
|---|---|
| 04 Common Initialization & Controlled Training | Completed / VALID |
| 05 Common Adam Optimizer Control | Completed / VALID |
| 06 Common BatchNorm Control | Completed / VALID |
| 07 Multi-Step Divergence & State Re-Synchronization | Completed / VALID |
| 08 Layer-by-Layer Training Trajectory Analysis | Completed / VALID |

> **Phase 2 - Strict Controlled Framework Comparison: Completed.**

Experiment 09는 생성하지 않았다. 추가 연구는 위 Future Work / Optional Extensions로만 남긴다.
