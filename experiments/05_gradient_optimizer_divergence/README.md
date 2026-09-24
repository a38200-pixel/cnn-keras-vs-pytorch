# Experiment 05 - Common Adam Optimizer Control

## Purpose

Experiment 04에서 관찰된 parameter trajectory separation 중 native Adam implementation 차이를 제거하면 초기 update와 장기 trajectory divergence가 얼마나 달라지는지 검증한다. Experiment 04의 Forward/Loss 분석은 이미 layer trace와 native/reference CE 진단으로 수행됐으므로 별도 standalone 실험은 생략했다.

## Difference from Experiment 04

| Condition | Experiment 04 | Experiment 05 |
|---|---|---|
| Gradient | Framework-native backward | Framework-native backward |
| Adam hyperparameters | 동일 | 동일 |
| Adam implementation | Keras/PyTorch native | **동일 `CommonAdam` NumPy float32 구현** |
| Update path | Native optimizer | canonical gradient → CommonAdam → canonical weight reload |

04→05에서 의미 있게 바뀐 학습 조건은 **Adam implementation 하나뿐**이다. Gradient·BN·backend numerical difference는 그대로 남는다.

## Unchanged Controlled Conditions

동일 physical split과 class mapping, RGB 128×128, 422,824 trainable parameters, float32, Seed 42/123/2026 W0, canonical input/augmentation, persisted batch order, integer label/raw logits/mean CE, BN eps/momentum, batch 32, 30 epochs·321 batches·9,630 updates, LR .001, no callback/scheduler/clipping/weight decay를 Experiment 04에서 그대로 상속했다.

## Common Reference Adam Design

Keras `GradientTape`와 PyTorch `loss.backward()`가 계산한 gradient를 동일 canonical layout의 float32 NumPy array로 export한다. 양쪽 모두 [`common.reference_adam.CommonAdam`](../../common/reference_adam.py)을 호출한다.

```text
native backward → canonical float32 gradient
→ same CommonAdam(m, v, step, equation/epsilon/bias correction/order)
→ canonical updated trainable parameters → framework model reload
```

Adam 설정은 lr=.001, beta1=.9, beta2=.999, epsilon=1e−7, weight decay=0, AMSGrad off다. BN running state는 각 native forward에서 갱신되며 CommonAdam이 수정하지 않는다.

## Experiment Validity

> **Completed / VALID.** Keras와 PyTorch 모두 Seed 42/123/2026에서 30 epochs와 Seed당 9,630 updates를 완료했다. 6개 history는 각각 30행이며, 42개 checkpoint(`2 Frameworks × 3 Seeds × 7 시점`)의 config hash는 `450cbd41...ca7d`로 일치한다.

Preflight에서 dataset 10,251/2,194/2,203, 422,824 trainable parameters, TensorFlow GPU/PyTorch MPS, W0/input/label/Conv1 exact, CommonAdam class·step과 finite gradient/m/v/update, canonical checkpoint round-trip을 검증했다. Full run에서도 동일 physical split, class mapping, W0, input, labels, batch/augmentation schedule, architecture, BN 설정, raw logits/mean CE, batch size 32, 30 epochs, LR .001을 유지했다.

## First-Step Diagnostic

동일 첫 batch에서 Framework-native backward가 만든 gradient를 canonical float32 layout으로 변환하고 같은 CommonAdam에 전달했다. Gradient가 이미 non-zero difference를 가지므로 update exact match는 validity 조건이 아니다.

| Seed | Gradient rel-L2 | 04 Native update rel-L2 | 05 Common update rel-L2 | Reduction |
|---:|---:|---:|---:|---:|
| 42 | 9.62e−4 | 0.029430 | 0.023449 | 20.32% |
| 123 | 2.83e−5 | 0.024755 | 0.000633 | 97.44% |
| 2026 | 3.10e−5 | 0.021971 | 0.002840 | 87.08% |
| **3-Seed mean** | — | **0.025385** | **0.008974** | **64.65%** |

Common Adam 적용 후 세 Seed 모두 첫 update divergence가 감소했고, 특히 Seed 123/2026에서 감소 폭이 컸다. 이는 native Adam implementation 차이가 초기 optimization divergence의 **amplification factor**로 기여했을 가능성을 시사한다. Gradient 차이가 남아 있고 Seed별 감소 폭도 다르므로 Adam이 전체 차이의 원인이라는 결론은 아니다.

## Early-Step Trajectory

Full Training과 분리된 fresh diagnostic model에서 측정한 global trainable-weight relative L2의 3-Seed 평균이다.

| Step | Experiment 04 Native Adam | Experiment 05 Common Adam | Reduction |
|---:|---:|---:|---:|
| 0 | 0 | 0 | — |
| 1 | 0.000524 | 0.000185 | 64.65% |
| 2 | 0.000856 | 0.000343 | 59.91% |
| 5 | 0.002324 | 0.000923 | 60.29% |
| 10 | 0.005606 | 0.002893 | 48.40% |
| 20 | 0.013313 | 0.008313 | 37.55% |
| 50 | 0.035022 | 0.027867 | 20.43% |
| 100 | 0.063401 | 0.053285 | 15.96% |

Common Adam의 감소 효과는 초기 step에서 가장 컸다. 반복 update와 함께 양쪽 trajectory가 계속 분리되면서 04 대비 감소율은 대체로 작아졌고, Step 100에는 약 16%가 남았다. Seed별 Step 100 감소율은 42/123/2026에서 각각 17.28%, 26.37%, 0.59%로 효과가 Seed-dependent했다.

## Epoch-Level Parameter Trajectory

공식 canonical checkpoint에서 계산한 global trainable-weight relative L2의 3-Seed 평균이다. BN running mean/variance는 global trainable-weight 계산에서 제외하고 아래에서 별도로 비교한다.

| Checkpoint | Optimizer step | Experiment 04 | Experiment 05 | Reduction |
|---|---:|---:|---:|---:|
| after_first_step | 1 | 0.000524 | 0.000185 | 64.66% |
| epoch 1 | 321 | 0.186375 | 0.178498 | 4.23% |
| epoch 5 | 1,605 | 0.566178 | 0.541288 | 4.40% |
| epoch 10 | 3,210 | 0.794442 | 0.766426 | 3.53% |
| epoch 20 | 6,420 | 1.013096 | 0.987945 | 2.48% |
| epoch 30 | 9,630 | **1.115488** | **1.095162** | **1.82%** |

첫 update의 감소는 컸지만 Epoch 1에는 차이가 4.23%로 축소됐고 Epoch 30에는 04와 거의 비슷한 global distance가 관찰됐다. 따라서 native Adam은 초기 divergence amplification에 기여하지만 장기 trajectory divergence를 단독으로 설명하지 못한다.

## Layer-Wise and BatchNorm State Trajectory

Experiment 05 Epoch 30의 Conv kernel relative L2는 다음과 같다.

| Seed | Conv1 | Conv2 | Conv3 | Conv4 |
|---:|---:|---:|---:|---:|
| 42 | 0.3282 | 0.7687 | 1.1036 | 1.2683 |
| 123 | 0.3893 | 0.8779 | 1.1295 | 1.2516 |
| 2026 | 0.4075 | 0.7837 | 1.0743 | 1.2394 |
| **Mean** | **0.3750** | **0.8101** | **1.1025** | **1.2531** |

세 Seed 모두 `Conv1 < Conv2 < Conv3 < Conv4`였으며 Experiment 04의 canonical relative L2에서도 같은 순서가 관찰됐다. 더 깊은 Conv layer에서 더 큰 separation이 반복됐지만, 이것만으로 layer depth가 원인임을 뜻하지 않는다.

Common Adam 이후에도 Epoch 30 BN running-state relative L2가 남았다. 3-Seed 평균 running mean/variance는 BN1 `.665/.545`, BN2 `.534/.369`, BN3 `.447/.361`, BN4 `1.125/.234`였다. 이는 optimizer implementation을 통제한 뒤에도 BN state trajectory가 분리됐다는 관찰이며, BN이 원인이라는 결론은 아니다. Experiment 06에서 BN forward/statistics/state update를 직접 통제할 근거가 된다.

## Training Dynamics

Epoch 30 train accuracy 평균은 Keras `64.62%`, PyTorch `64.43%`로 `0.19%p` 차이에 불과했다. Seed별 Keras/PyTorch 값도 42 `64.95/65.09%`, 123 `64.53/64.16%`, 2026 `64.37/64.05%`였다. 큰 parameter trajectory separation이 곧 training-set fitting divergence를 의미하지 않는다.

## Fixed Epoch 30 Results — Primary

| Seed | Keras Acc. | PyTorch Acc. | Keras Macro F1 | PyTorch Macro F1 | Keras Loss | PyTorch Loss |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 45.12% | 41.44% | 42.31% | 36.06% | 1.6035 | 1.8504 |
| 123 | 49.34% | 47.80% | 45.39% | 45.57% | 1.4208 | 1.4076 |
| 2026 | 39.95% | 37.81% | 38.11% | 36.01% | 1.8742 | 2.1276 |
| **Mean ± sample std** | **44.80 ± 4.71%** | **42.35 ± 5.05%** | **41.94 ± 3.65%** | **39.21 ± 5.50%** | **1.6328 ± .2281** | **1.7952 ± .3632** |

Mean absolute paired gap은 Accuracy가 Experiment 04 `3.36%p`에서 05 `2.45%p`로 `0.91%p`(27.03%) 감소했고, Macro F1은 `4.78%p`에서 `2.84%p`로 `1.94%p`(40.51%) 감소했다. 이는 Framework 간 final gap이 감소했다는 관찰이지 Common Adam이 일반적으로 성능을 향상시키거나 특정 Framework가 더 낫다는 의미가 아니다.

## Best Validation Checkpoint — Secondary

| Seed | Best epoch K/P | Keras Acc. | PyTorch Acc. | Keras Macro F1 | PyTorch Macro F1 | Keras Loss | PyTorch Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 17 / 22 | 47.71% | 49.30% | 44.68% | 45.33% | 1.3468 | 1.3646 |
| 123 | 11 / 20 | 44.94% | 47.12% | 43.05% | 43.68% | 1.3771 | 1.3984 |
| 2026 | 27 / 19 | 50.70% | 48.62% | 49.76% | 45.78% | 1.2993 | 1.3425 |
| **Mean ± sample std** | — | **47.78 ± 2.88%** | **48.34 ± 1.11%** | **45.83 ± 3.50%** | **44.93 ± 1.10%** | **1.3411 ± .0392** | **1.3685 ± .0281** |

Best checkpoint의 mean absolute paired gap은 Accuracy `1.95%p`, Macro F1 `1.75%p`였다. Accuracy/F1 우위 방향은 Seed 42/123과 2026에서 바뀌어 일관된 Framework 우열이 없었다. 모든 Seed에서 Keras와 PyTorch의 best epoch도 달랐으므로, Common Adam 이후에도 generalization optimum timing은 Framework별로 달랐다. Fixed Epoch 30과 Best Validation 결과는 서로 다른 질문이므로 분리해 해석한다.

## Answer to the Experiment 05 Research Question

Common Adam을 사용하자 첫 optimizer update에서의 Keras–PyTorch 차이는 3-Seed 평균 `64.65%` 감소했다. 이는 native Adam implementation 차이가 초기 optimization divergence를 증폭시키는 요인으로 기여했음을 시사한다.

그러나 반복 학습이 진행되면서 parameter trajectory는 다시 분리됐다. 04 대비 global distance 감소율은 Epoch 1 `4.23%`, Epoch 30 `1.82%`에 그쳤고, Epoch 30 global relative L2도 Experiment 04 `1.1155`, Experiment 05 `1.0952`로 비슷했다.

따라서 native Adam implementation은 초기 divergence의 **amplification contributor**이지만 장기적인 cross-framework trajectory divergence를 단독으로 설명하지 못한다. Gradient, BatchNorm state, Metal/MPS backend numerical difference 등 다른 요인이 반복 학습 과정에서 계속 누적될 가능성이 남아 있으며 추가 격리가 필요하다.

## Connection to Experiment 04 and Next Experiment

Experiment 04에서는 동일 W0/input에서 Conv1까지 exact였고 BN1에서 약 `1e−6`의 첫 비영 차이, 이후 gradient 차이, native Adam update 확대와 장기 trajectory separation을 관찰했다. Experiment 05에서는 Common Adam으로 초기 update 차이가 줄었지만 장기 separation의 대부분이 다시 나타났다. 현재 결과는 Framework difference를 하나의 optimizer 구현으로 설명하기보다 작은 numerical/gradient/state difference가 반복 학습을 통해 누적되는 과정으로 보는 해석과 더 잘 맞는다.

다음 **Experiment 06 - BatchNorm State Divergence**의 질문은 다음과 같다.

> Common Adam까지 적용한 상태에서 BatchNorm forward/statistics/state update까지 통제하면 남아 있는 trajectory 및 generalization divergence가 얼마나 감소하는가?

## Result Artifacts

- `results/keras_common_adam_results.csv`, `pytorch_common_adam_results.csv`
- `results/controlled_comparison_summary.json`, `diagnostic_comparison_04_vs_05.json`
- `results/first_step/diagnostic_3seed_summary.json`
- `results/early_steps/`, `results/trajectory/`, `results/history/`
- `results/checkpoints/checkpoint_manifest.csv`

## Limitations

- CommonAdam은 native optimizer 차이만 제거하며 native backward·BN·Metal/MPS 차이는 남긴다.
- NumPy로 gradient와 parameter를 왕복하므로 runtime은 성능 비교 지표로 사용하지 않는다.
- 3 Seeds와 하나의 CNN/dataset/split에 제한된다.
- Initial/trajectory divergence 감소와 performance gap의 관계는 단조롭지 않다.
- BN state distance는 관찰값이며 원인 판정에는 직접 통제 실험이 필요하다.
