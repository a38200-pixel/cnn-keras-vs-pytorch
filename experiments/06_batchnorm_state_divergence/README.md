# Experiment 06 - BatchNorm State Divergence

## Research Question

> Common Adam 조건에서 BatchNorm forward/statistics/state update까지 동일하게 통제하면 Keras–PyTorch trajectory 및 generalization divergence가 얼마나 감소하는가?

## Difference from Experiment 05

| Condition | Experiment 05 | Experiment 06 |
|---|---|---|
| Gradient | Framework-native backward | Framework-native backward |
| Optimizer | CommonAdam | CommonAdam |
| BatchNorm | Framework-native BN | **Explicit Common BN semantics** |
| BN autograd | Framework-native | Framework-native tensor/autograd, identical mathematical rule |

05→06에서 의미 있게 바뀐 조건은 **Native BN → Common BN**뿐이다. Same W0/input/batch/augmentation/labels, physical dataset split, class mapping, architecture, raw-logit mean CE, float32, batch size 32, LR `.001`, CommonAdam과 Seed 42/123/2026을 유지했다. Backend reduction과 autograd의 floating-point difference는 남아 있다.

## Experiment Validity

> **Completed / VALID.** Keras와 PyTorch 모두 세 Seed에서 30 epochs와 Seed당 9,630 optimizer updates를 완료했다.

- 6개 history CSV가 각각 30개 epoch를 포함한다.
- `2 Frameworks × 3 Seeds × 7 시점 = 42`개 canonical checkpoint가 생성됐다.
- 동일 config hash `30027b92...036c`, 422,824 trainable parameters, CommonAdam implementation과 Common BN semantics를 사용했다.
- W0/input/labels/initial BN state와 Conv1 output은 세 Seed 모두 exact였다.
- Native BN layer가 제거됐고 gamma/beta gradient flow는 non-None·finite였다.
- Checkpoint와 final result CSV를 바탕으로 생성한 `controlled_comparison_summary.json`의 최종 상태는 `VALID`다.

따라서 05→06 비교는 BN semantics 통제 효과를 해석할 수 있는 유효한 controlled experiment다.

## Common BN Semantics

Keras NHWC는 axes `[0,1,2]`, PyTorch NCHW는 axes `[0,2,3]`에서 channel별 통계를 계산한다.

```text
batch_mean     = mean(x over N,H,W)
batch_variance = mean((x - batch_mean)^2 over N,H,W)  # population, ddof=0
x_hat          = (x - batch_mean) / sqrt(batch_variance + 1e-3)
output         = gamma * x_hat + beta

running_mean     = 0.99 * running_mean     + 0.01 * batch_mean
running_variance = 0.99 * running_variance + 0.01 * batch_variance
```

Evaluation에서는 running state만 사용하고 state를 갱신하지 않는다. Batch statistics는 TensorFlow/PyTorch tensor 연산으로 계산되어 backward graph에 남으며 trainable gamma/beta와 non-trainable running mean/variance를 분리한다.

## First-Step BN State

세 Seed 모두 첫 non-zero forward difference는 BN1에서 관찰됐지만, BN1 output relative L2는 약 `1e−7`, max absolute difference는 최대 `9.54e−7` 수준이었다.

| Seed | BN1 batch mean rel-L2 | BN1 batch variance rel-L2 | BN1 output rel-L2 | All-BN running mean rel-L2 | All-BN running variance rel-L2 |
|---:|---:|---:|---:|---:|---:|
| 42 | 5.76e−8 | 3.56e−8 | 9.80e−8 | 9.27e−8 | 2.74e−9 |
| 123 | 2.67e−8 | 7.35e−8 | 8.40e−8 | 8.84e−8 | 0 |
| 2026 | 2.06e−8 | 8.30e−8 | 9.26e−8 | 1.10e−7 | 2.74e−9 |

All-BN running-variance relative L2의 3-Seed 평균은 Experiment 05 `3.15e−7`에서 Experiment 06 `1.83e−9`로 약 `99.42%` 감소했고, Seed 123은 exact였다. 이는 Common BN이 native BN의 running-state update semantics 차이를 실제로 거의 제거했음을 보여준다. 다만 BN1 output difference와 running-mean difference는 남았으므로 BN 차이가 완전히 사라졌다고 해석하지 않는다.

Experiment 05는 별도 forward tensor CSV를 저장하지 않았으므로 05 BN1 output 비교값은 pre-update forward path·W0·input이 같은 Experiment 04 first-step trace를 사용했다. Running state와 gradient/update 비교값은 Experiment 05 artifact에서 직접 읽었다.

## First-Step Gradient and Update

| Seed | Gradient rel-L2 05 → 06 | Update rel-L2 05 → 06 | Weight rel-L2 05 → 06 |
|---:|---:|---:|---:|
| 42 | 9.62e−4 → 8.49e−5 | .023449 → .002830 | .000484 → .000058 |
| 123 | 2.83e−5 → 7.85e−5 | .000633 → .006167 | .000013 → .000127 |
| 2026 | 3.10e−5 → 1.24e−5 | .002840 → .002840 | .000059 → .000059 |
| **3-Seed mean** | **3.41e−4 → 5.86e−5** | **.008974 → .003946** | **.000185 → .0000815** |
| **Mean reduction** | **82.79%** | **56.03%** | **56.03%** |

Common BN은 first-step gradient/update/weight divergence를 평균적으로 줄였다. 그러나 Seed 42의 update divergence는 크게 감소한 반면 Seed 123은 증가했고 Seed 2026은 거의 같았다. 즉 **평균 감소는 관찰됐지만 Seed-consistent한 효과는 아니었다.**

## Early-Step Trajectory

Fresh diagnostic model에서 측정한 global trainable-weight relative L2의 3-Seed 평균이다.

| Step | Experiment 05 | Experiment 06 | 06 vs 05 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | — |
| 1 | .000185 | .0000815 | 56.03% 감소 |
| 2 | .000343 | .000196 | 42.82% 감소 |
| 5 | .000923 | .000944 | 2.28% 증가 |
| 10 | .002893 | .003233 | 11.75% 증가 |
| 20 | .008313 | .010483 | 26.09% 증가 |
| 50 | .027867 | .033101 | 18.78% 증가 |
| 100 | **.053285** | **.060059** | **12.71% 증가** |

Step 1–2에서는 06의 divergence가 더 작았지만 Step 5부터 감소 효과가 사라졌다. Step 100의 Seed별 05→06 변화도 42 `.067122→.066475`, 123 `.044793→.065426`, 2026 `.047939→.048277`로 일관된 감소가 아니었다. Common BN의 초기 divergence reduction은 몇 step 이후 유지되지 않았다.

## Epoch-Level Parameter Trajectory

공식 canonical checkpoint에서 계산한 global trainable-weight relative L2의 3-Seed 평균이다.

| Checkpoint | Optimizer step | Experiment 05 | Experiment 06 | 06 vs 05 |
|---|---:|---:|---:|---:|
| after first step | 1 | .000185 | .0000815 | 56.02% 감소 |
| epoch 1 | 321 | .178498 | .182288 | 2.12% 증가 |
| epoch 5 | 1,605 | .541288 | .552206 | 2.02% 증가 |
| epoch 10 | 3,210 | .766426 | .775916 | 1.24% 증가 |
| epoch 20 | 6,420 | .987945 | .991490 | 0.36% 증가 |
| epoch 30 | 9,630 | **1.095162** | **1.097571** | **0.22% 증가** |

Epoch 30 global relative L2는 05 `1.0952`, 06 `1.0976`으로 사실상 같은 수준이었다. Native BN semantics를 제거해도 long-term parameter-space separation은 감소하지 않았다.

## Layer-Wise and BN-State Trajectory

Experiment 06 Epoch 30 Conv kernel relative L2는 다음과 같다.

| Seed | Conv1 | Conv2 | Conv3 | Conv4 |
|---:|---:|---:|---:|---:|
| 42 | .2361 | .7799 | 1.1248 | 1.2622 |
| 123 | .3979 | .8721 | 1.1475 | 1.2726 |
| 2026 | .3703 | .7254 | 1.0658 | 1.2279 |
| **Mean** | **.3348** | **.7925** | **1.1127** | **1.2542** |

세 Seed 모두 `Conv1 < Conv2 < Conv3 < Conv4`였고 이 순서는 Experiments 04, 05, 06에서 반복됐다. 이는 deeper convolution blocks가 일관되게 더 큰 cross-framework parameter-space separation을 보였다는 관찰이며 Conv4가 원인이라는 뜻은 아니다.

Common BN을 사용한 뒤에도 Epoch 30 BN1–BN4 running mean relative L2 평균은 각각 `.624/.599/.455/1.098`, running variance는 `.579/.402/.388/.256`이었다. 첫-step native state-update discrepancy를 거의 제거했는데도 장기 BN state는 다시 분리됐다. 따라서 큰 BN-state divergence는 native BN semantics 단독 원인의 증거라기보다, 이미 분리된 training trajectory의 downstream consequence를 일부 반영할 수 있다.

## Training-Set Fitting

Epoch 30 train accuracy 평균은 Keras `64.55%`, PyTorch `64.33%`로 `0.22%p` 차이에 그쳤다. Seed별 값도 Keras/PyTorch가 42 `64.94/64.61%`, 123 `63.53/63.61%`, 2026 `65.18/64.75%`로 유사했다. 큰 parameter-space divergence가 큰 training-set fitting divergence를 의미하지 않는다.

## Fixed Epoch 30 Results — Primary

| Seed | Keras Acc. | PyTorch Acc. | Keras Macro F1 | PyTorch Macro F1 | Keras Loss | PyTorch Loss |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 41.90% | 49.93% | 38.53% | 48.78% | 1.7063 | 1.3733 |
| 123 | 46.62% | 45.07% | 41.71% | 41.65% | 1.6556 | 1.5642 |
| 2026 | 44.76% | 39.40% | 43.63% | 36.88% | 1.6770 | 1.7557 |
| **Mean ± sample std** | **44.42 ± 2.38%** | **44.80 ± 5.27%** | **41.29 ± 2.58%** | **42.44 ± 5.99%** | **1.6796 ± .0254** | **1.5644 ± .1912** |

Signed mean gap `(PyTorch − Keras)`은 Accuracy `+0.38%p`, Macro F1 `+1.15%p`로 작지만, Seed별 방향이 상쇄된 결과다. Mean absolute paired gap은 Accuracy `4.98%p`, Macro F1 `5.69%p`로 Experiment 05의 `2.45%p`, `2.84%p`보다 증가했다. 따라서 Common BN은 final generalization gap을 일관되게 줄이지 않았다. Seed별 방향이 서로 달라 Framework 우열로 해석하지 않는다.

## Best Validation Checkpoint — Secondary

| Seed | Best epoch K/P | Keras Acc. | PyTorch Acc. | Keras Macro F1 | PyTorch Macro F1 | Keras Loss | PyTorch Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 19 / 25 | 46.62% | 49.25% | 45.45% | 47.98% | 1.3897 | 1.3310 |
| 123 | 21 / 25 | 46.98% | 49.52% | 45.04% | 46.21% | 1.3718 | 1.3341 |
| 2026 | 19 / 25 | 49.39% | 51.52% | 46.83% | 49.75% | 1.3179 | 1.2916 |
| **Mean ± sample std** | — | **47.66 ± 1.50%** | **50.10 ± 1.24%** | **45.78 ± .93%** | **47.98 ± 1.77%** | **1.3598 ± .0374** | **1.3189 ± .0237** |

Experiment 06 조건에서는 세 paired best-validation checkpoint 모두 PyTorch의 test Accuracy와 Macro F1이 더 높았다. 다만 하나의 CNN/dataset과 3 Seeds만으로 Framework-level superiority를 확립할 수 없다. Best epoch도 Keras `19/21/19`, PyTorch `25/25/25`로 달라 Adam과 BN semantics 통제 후에도 generalization optimum timing이 Framework별로 달랐다. Fixed Epoch 30과 Best Validation 결과는 서로 다른 질문이므로 분리해 해석한다.

## Answer to the Experiment 06 Research Question

Common BN은 first-step running-state update discrepancy의 대부분을 제거했다. 또한 first-step gradient와 update divergence를 3-Seed 평균 기준 각각 `82.79%`, `56.03%` 줄였다.

그러나 이 감소는 첫 몇 optimization step 이후 유지되지 않았다. Step 100의 global weight relative L2는 Experiment 05 `.0533`보다 Experiment 06 `.0601`이 컸고, Epoch 30에는 05 `1.0952`, 06 `1.0976`으로 거의 같은 global parameter divergence가 관찰됐다.

따라서 native BatchNorm semantics는 장기 Keras–PyTorch trajectory divergence를 충분히 설명하지 못한다. 남은 결과는 reduction, autograd와 backend execution에서 발생하는 작은 numerical difference가 반복 update를 통해 누적되는 과정과 일치하지만, 특정 backend 연산이 원인으로 입증된 것은 아니며 추가 격리가 필요하다.

## Connection Across Experiments 04–06

```text
04: Same W0/Input
    → BN1에서 첫 미세 non-zero difference
    → gradient difference
    → Native Adam에서 update 차이 확대
    → long-term trajectory divergence

05: Common Adam
    → initial update divergence 감소
    → long-term divergence 대부분 재발

06: Common Adam + Common BN
    → BN running-state implementation difference 대부분 제거
    → first-step divergence 평균 감소
    → long-term divergence는 사실상 유지
```

현재 결과에서 native Adam implementation과 native BatchNorm semantics 어느 하나도 장기 cross-framework trajectory divergence를 단독으로 설명하지 못한다.

## Next Research Direction

후속 [Experiment 07 - Multi-Step Divergence & State Re-Synchronization](../07_multistep_state_resynchronization/README.md)은 **Diagnostic Completed / VALID**다. Exact full-state sync 후 새로 생성된 one-step divergence는 후반 checkpoint에서 매우 작았지만 free-running accumulated divergence는 크게 유지됐다. 이는 state-dependent feedback과 강하게 일치하며, 남은 anchor-local sensitivity는 Experiment 08의 layer-wise isolation으로 연결한다.

## Result Artifacts and Metadata Note

- `results/keras_common_bn_results.csv`, `pytorch_common_bn_results.csv`
- `results/controlled_comparison_summary.json`
- `results/first_step/diagnostic_3seed_summary.json`
- `results/early_steps/`, `results/trajectory/`, `results/history/`
- `results/checkpoints/checkpoint_manifest.csv`
- `results/diagnostic_comparison_05_vs_06.json`

`diagnostic_comparison_05_vs_06.json`의 `full_training_executed: false`는 full training 이전에 생성된 diagnostic artifact의 시점 정보다. 현재 실험 상태의 source of truth는 final result CSV와 `controlled_comparison_summary.json`이며, 이에 따라 Experiment 06의 최종 상태는 **Completed / VALID**다.

## Limitations

- 동일 mathematical semantics는 bitwise-identical backend reduction/autograd를 보장하지 않는다.
- BN1의 첫 non-zero 차이는 약 `1e−7` relative L2이며 실용적으로 유의한 성능 분기점으로 단정하지 않는다.
- First-step 평균 감소는 Seed-consistent하지 않았고 long-term 성능과 단조 관계를 보이지 않았다.
- 3 Seeds, 하나의 CNN, dataset과 physical split에 제한된다.
- BN state distance는 원인과 결과가 함께 반영될 수 있으며, 관찰만으로 인과성을 확정하지 않는다.
