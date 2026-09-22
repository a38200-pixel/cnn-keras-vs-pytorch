# Experiment 05 - Common Adam Optimizer Control

## Purpose

Experiment 04에서 관찰된 parameter trajectory separation 중 native Adam implementation 차이를 제거하면 초기 update와 이후 trajectory가 얼마나 달라지는지 검증한다. Experiment 04의 Forward/Loss 분석은 이미 layer trace와 native/reference CE 진단으로 수행됐으므로 별도 standalone 실험은 생략했다.

## Difference from Experiment 04

| Condition | Experiment 04 | Experiment 05 |
|---|---|---|
| Gradient | Framework-native backward | Framework-native backward |
| Adam hyperparameters | 동일 | 동일 |
| Adam implementation | Keras/PyTorch native | **동일 `CommonAdam` NumPy float32 구현** |
| Update path | Native optimizer | canonical gradient → CommonAdam → canonical weight reload |

04→05에서 의미 있게 바뀌는 학습 조건은 **Adam implementation 하나뿐**이다. Gradient·BN·backend numerical difference는 그대로 남는다.

## Unchanged Controlled Conditions

동일 physical split과 class mapping, RGB 128×128, 422,824 trainable parameters, float32, Seed 42/123/2026 W0, canonical input/augmentation, persisted batch order, integer label/raw logits/mean CE, BN eps/momentum, batch 32, 30 epochs·321 batches·9,630 updates, LR .001, no callback/scheduler/clipping/weight decay를 Experiment 04에서 그대로 상속한다.

## Common Reference Adam Design

Keras `GradientTape`와 PyTorch `loss.backward()`가 계산한 gradient를 동일 canonical layout의 float32 NumPy array로 export한다. 양쪽 모두 [`common.reference_adam.CommonAdam`](../../common/reference_adam.py)을 호출한다.

```text
native backward → canonical float32 gradient
→ same CommonAdam(m, v, step, equation/epsilon/bias correction/order)
→ canonical updated trainable parameters → framework model reload
```

Adam 설정은 lr=.001, beta1=.9, beta2=.999, epsilon=1e−7, weight decay=0, AMSGrad off다. BN running state는 각 native forward에서 갱신되며 CommonAdam이 수정하지 않는다.

## Preflight Validation

`COMMON_ADAM_PRECHECK=VALID`, `FULL_TRAINING_READY=TRUE`다. Parent 04 validity, dataset 10,251/2,194/2,203, 422,824 trainable parameters, fixed schedule, TensorFlow GPU/PyTorch MPS, W0/input/Conv1, CommonAdam class와 step, finite gradient/m/v/update, Seed별 step 0/1 canonical checkpoint round-trip을 모두 확인했다. 차이가 존재하는 사실은 FAIL이 아니며 mapping·shape·implementation 불일치나 NaN/Inf가 FAIL이다.

## First-Step Diagnostic

Seed별 동일 첫 batch에서 gradient/update/m/v/W1 relative L2를 기록하고 Experiment 04 native Adam update relative L2와 직접 비교한다. Gradient가 다르므로 CommonAdam의 delta 자체는 exact일 필요가 없다. 양쪽이 같은 클래스·dtype·수식·operation order를 호출하는지가 validity 조건이다.

| Seed | Gradient rel-L2 | 04 native update rel-L2 | 05 common update rel-L2 | 05 / 04 ratio |
|---:|---:|---:|---:|---:|
| 42 | 9.62e−4 | 0.02943 | 0.02345 | 79.68% |
| 123 | 2.83e−5 | 0.02476 | 0.000633 | 2.56% |
| 2026 | 3.10e−5 | 0.02197 | 0.002840 | 12.92% |

세 Seed 모두 first-step update relative divergence가 04보다 감소했다. 감소 폭은 Seed별로 크게 달랐으므로 native Adam 제거가 trajectory amplification에 기여했을 가능성을 시사하지만 전체 차이의 단독 원인이라는 뜻은 아니다.

## Early-Step Diagnostic

Full Training과 분리된 fresh model에서 step `0/1/2/5/10/20/50/100`의 global/layer weight, gradient, update, m/v distance와 canonical checkpoint를 저장한다. 이후 04의 early-step trajectory와 비교할 수 있다.

| Seed | Step 1 | Step 2 | Step 5 | Step 10 | Step 20 | Step 50 | Step 100 | 04 Step 100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | .000484 | .000841 | .002026 | .005288 | .013282 | .037648 | .067122 | .081144 |
| 123 | .000013 | .000046 | .000225 | .001192 | .004049 | .018774 | .044793 | .060836 |
| 2026 | .000059 | .000143 | .000518 | .002198 | .007609 | .027179 | .047939 | .048222 |

Step 100 weight relative L2는 04 대비 Seed 42에서 17.28%, Seed 123에서 26.37%, Seed 2026에서 0.59% 감소했다. Common Adam 이후에도 divergence는 누적됐으며, Seed 2026에서는 Step 100 차이가 거의 유지됐다. Gradient·BN·backend 차이가 남는다는 점과 optimizer 효과가 Seed-dependent하다는 점을 함께 보여준다.

## Expected Comparison with Experiment 04

- Step 1 update divergence
- Step 100 weight divergence
- Epoch 1/5/10/20/30 global 및 Conv1–4 trajectory
- Fixed Epoch 30과 Best Validation 성능

05에서 divergence가 감소해도 “Adam이 전체 차이의 원인”이라고 결론내리지 않는다. 허용 가능한 해석은 native Adam implementation이 관찰된 trajectory divergence의 amplification에 기여했을 가능성이다. 추가 격리는 이후 분석이 필요하다.

## Training Status

> **Training Pending.** `RUN_TRAINING=False`이며 3-Seed × 30-epoch 학습과 Test 평가는 실행하지 않는다.

검증 순서:

```bash
.venv-metal/bin/python experiments/05_gradient_optimizer_divergence/first_step_trace.py
.venv-metal/bin/python experiments/05_gradient_optimizer_divergence/early_step_trace.py
.venv-metal/bin/python experiments/05_gradient_optimizer_divergence/preflight_validate.py
.venv-metal/bin/python experiments/05_gradient_optimizer_divergence/compare_controlled.py
```

실제 학습은 사용자가 두 training 파일의 `RUN_TRAINING=True`를 직접 설정한 뒤에만 실행한다.

```bash
.venv-metal/bin/python experiments/05_gradient_optimizer_divergence/keras_controlled_train.py
.venv-metal/bin/python experiments/05_gradient_optimizer_divergence/pytorch_controlled_train.py
.venv-metal/bin/python experiments/05_gradient_optimizer_divergence/compare_controlled.py
```

## Limitations

- CommonAdam은 native optimizer 차이만 제거하며 native backward·BN·Metal/MPS 차이는 남긴다.
- NumPy로 gradient와 parameter를 왕복하므로 runtime은 성능 비교 지표로 사용하지 않는다.
- 3 Seeds와 하나의 CNN/dataset/split에 제한된다.
- 초기 divergence 감소가 장기 성능 차이 감소를 보장하지 않는다.
