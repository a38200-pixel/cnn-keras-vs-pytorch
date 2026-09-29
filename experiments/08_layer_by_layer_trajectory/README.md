# Experiment 08 - Layer-by-Layer Training Trajectory Analysis

## Research Question

> Common Adam과 Common BN 조건에서 full state를 exact re-synchronize했을 때, 매우 작은 Keras–PyTorch one-step difference는 forward, loss, backward, optimizer update의 어느 stage에서 처음 관찰되고 어디에서 국소적으로 확대되는가?

Experiment 07은 exact full-state sync 뒤 새로 생성되는 global one-step divergence가 free-running accumulated divergence보다 매우 작음을 보였다. Experiment 08은 같은 one-step 안을 layer 단위로 열어 first non-zero, largest divergence와 amplification 위치를 구분한다. 어느 위치도 이 실험만으로 root cause로 해석하지 않는다.

## Validity

> **Diagnostic Completed / VALID.** `LAYER_TRACE_PRECHECK=VALID`, `TRACE_EQUIVALENCE=VALID`, `SELECTED_CASES_COMPLETE=TRUE`, `NAN_INF_FOUND=FALSE`다.

- Experiment 06 checkpoint는 read-only로 사용하고 Experiment 07의 full-state re-synchronization utility를 재사용했다.
- 총 9개 selected trace case를 완료했다.
- 각 case 시작 전에 weights, Adam `m/v/step`, BN gamma/beta/running mean/variance, input과 label을 exact하게 맞췄으며 Input/Label conversion도 exact였다.
- Experiment 06부터의 Common Adam과 Common BN 조건을 그대로 유지했다.
- Experiment 07의 고정 batch 32개를 그대로 사용했다. Tensor hash는 `c3ef4b5c6cb842154ddb03fa6cdc175dd5ab4e69a1e02f2bbc5493c3e2a48e7c`로 exact match였다.
- 각 case는 fresh checkpoint load 후 **1 forward → 1 loss → 1 backward → 1 Common Adam update**만 수행했다. BN은 forward에서 한 번만 실행됐다.
- Experiment 07과 loss, global gradient/update/post-weight를 대조한 45개 항목이 모두 exact match였다. 허용 tolerance는 `atol=1e-12`, `rtol=1e-7`이었지만 실제 비교는 전부 exact였다.
- 새로운 Full Training, Test-set model selection, dataset 변경과 checkpoint overwrite는 수행하지 않았다.

## Selected Cases

| Case | Step | Anchor | Selection reason |
|---|---:|---|---|
| Initial Seed 42 | 0 | shared | Initial synchronized baseline |
| Initial Seed 123 | 0 | shared | Initial synchronized baseline |
| Initial Seed 2026 | 0 | shared | Initial synchronized baseline |
| Epoch 1 Seed 42 | 321 | K / P | Low-divergence control |
| Epoch 30 Seed 123 | 9,630 | K / P | Late-stage anchor-sensitive case |
| Epoch 30 Seed 2026 | 9,630 | K / P | Late-stage anchor-sensitive case |

Epoch 1 control은 두 anchor의 synchronized post-weight relative L2 평균이 가장 작은 Seed로 정했다. 평균은 Seed 42 `1.5092e-8`, Seed 123 `1.9052e-8`, Seed 2026 `1.9608e-8`이어서 Seed 42를 선택했다.

## Metrics and Trace Order

Forward는 Input → 각 Conv/BN 내부/BN output/ReLU/Pool → GAP → FC128 pre-activation/ReLU → Logits → Loss 순서다. Common BN에서는 같은 forward에서 생성된 input, batch mean, population variance, normalized `x_hat`, affine output을 기록했다. PyTorch NCHW activation은 NHWC로 변환했고 Conv kernel은 OIHW→HWIO, Dense는 IO canonical layout으로 비교했다.

각 tensor의 relative L2는 다음과 같다.

```text
||Keras - PyTorch||₂ / max(||Keras||₂, ||PyTorch||₂, 1e-12)
```

Backward summary의 순서는 Logits에서 입력 방향으로 역행한다. 따라서 `first_nonzero_backward_stage=logits`는 loss에 가장 가까운 traced gradient에서 이미 non-zero가 관찰됐다는 뜻이며, forward의 최초 발생 위치를 뜻하지 않는다. `update_to_gradient_ratio`는 gradient relative L2가 `1e-12`보다 클 때만 계산한다. Shape가 달라지는 연속 stage의 amplification ratio와 이 ratio는 모두 descriptive metric이다.

## Case Results

| Case | First non-zero forward | Largest forward relL2 | Largest activation-gradient relL2 | Largest parameter-gradient | Largest update |
|---|---|---|---|---|---|
| Initial 42 / shared | BN1 batch mean | BN4 x̂ `5.82e-7` | Conv1 `6.53e-3` | Conv1 kernel `1.83e-3` | Conv2 kernel `3.48e-2` |
| Initial 123 / shared | BN1 batch mean | ReLU4 `5.81e-7` | BN1 `4.39e-3` | Conv1 kernel `8.91e-4` | Conv2 kernel `2.48e-2` |
| Initial 2026 / shared | BN1 batch mean | ReLU4 `5.77e-7` | BN1 `2.26e-3` | BN2 beta `1.12e-3` | Conv3 kernel `1.94e-4` |
| Epoch 1, 42 / K | BN1 batch mean | BN4 output `2.51e-7` | BN2 `6.03e-4` | BN1 beta `5.54e-5` | BN1 beta `2.64e-5` |
| Epoch 1, 42 / P | BN1 batch mean | BN4 output `2.46e-7` | BN2 `3.56e-3` | BN1 beta `7.20e-5` | BN1 beta `3.54e-5` |
| Epoch 30, 123 / K | BN1 batch mean | BN4 x̂ `3.59e-7` | BN3 `2.70e-4` | Conv2 kernel `2.77e-4` | Conv3 kernel `9.41e-5` |
| Epoch 30, 123 / P | BN1 batch mean | ReLU3 `3.71e-7` | BN1 `3.03e-4` | Conv1 kernel `1.35e-5` | Conv1 kernel `5.50e-6` |
| Epoch 30, 2026 / K | BN1 batch mean | ReLU3 `3.74e-7` | BN2 `4.55e-3` | Conv4 kernel `1.30e-3` | Conv1 kernel `4.38e-4` |
| Epoch 30, 2026 / P | BN1 batch mean | BN4 x̂ `3.68e-7` | BN2 `7.53e-5` | Conv1 kernel `1.11e-5` | Conv1 kernel `1.51e-5` |

모든 case에서 input과 Conv1 output은 exact였고, first observed numerical divergence는 BN1의 batch-mean reduction에서 `4.43e-8–7.62e-8` relative L2로 반복됐다. 이후 forward 차이는 단조롭게 증가하지 않았지만 대체로 deeper block에서 `2.46e-7–5.82e-7`의 case 최대값에 도달했다. Initial에서는 최대가 BN4 x̂/ReLU4, Epoch 1에서는 BN4 output, Epoch 30에서는 BN4 x̂ 또는 ReLU3였다.

Across all nine synchronized trace cases, the first observed non-zero numerical difference repeatedly appeared at the BN1 batch-mean reduction. 따라서 이는 테스트 환경에서 repeatable한 **reproducible first observed numerical entry point**로 기술할 수 있다. Forward numerical differences remained small, while their largest observed location shifted across deeper stages.

Loss는 Initial 3개와 Epoch 1 K/P에서 exact였고, Epoch 30 네 case의 absolute difference는 `5.96e-8–1.19e-7`이었다. 첫 forward non-zero 위치와 loss non-zero 여부를 동일시할 수 없다.

## Backward and Common Adam

모든 case에서 backward order의 첫 traced non-zero는 `dLoss/dLogits`였다. 최대 activation-gradient stage는 Initial의 Conv1/BN1, Epoch 1의 BN2, Epoch 30의 BN1–BN3로 case마다 달랐다. 최대 parameter-gradient group도 Conv1, BN2 beta, BN1 beta, Conv2 또는 Conv4 kernel로 바뀌었다. 즉 특정 단일 layer가 모든 parameter-space location에서 일관된 backward amplification point였다고 볼 수 없다.

There was no single parameter group or backward layer that consistently acted as the dominant amplification point.

Initial에서는 일부 parameter group의 update divergence가 gradient divergence보다 크게 변환됐다. 최대 `update_to_gradient_ratio`는 Seed 42 Conv3 `34.8`, Seed 123 Conv4 `135.2`, Seed 2026 Conv3 `86.2`였다. 반면 Epoch 1과 Epoch 30 selected case에서는 최대 ratio가 대체로 `0.48–1.37` 범위였다. 이는 Common Adam transformation의 local amplification이 initial state에서 관찰됐음을 시사하지만 optimizer를 root cause로 확정하지 않는다.

Optimizer transformation can locally amplify gradient differences, but Experiment 08 does not identify a single parameter group that consistently dominates this amplification.

Global 결과는 Experiment 07과 exact하게 동일하다. Initial global gradient/update/post-weight relative L2는 Seed별로 각각 `3.72e-4–5.33e-4`, `9.86e-5–1.43e-2`, `2.04e-6–2.95e-4`였다. Epoch 1 post-weight는 `1.07e-8/1.95e-8`이었고, Epoch 30은 K-anchor에서 Seed 123 `1.19e-7`, Seed 2026 `1.01e-6`, P-anchor에서 각각 `8.31e-9`, `8.23e-9`였다.

## BN Internal Trace

Common BN을 사용해도 identical Conv1 input에서 BN1 batch mean이 모든 case의 first observed numerical divergence였다. BN1 x̂/affine output은 대체로 `7.04e-8–1.12e-7`, deeper BN x̂/output은 최대 약 `5.82e-7`이었다. Post-step running-state relative L2는 매우 작았고 일부 variance 비교는 exact였다. 전체 BN running mean은 최대 약 `1.06e-7`, variance는 최대 약 `1.03e-8` 수준이었다.

이 결과는 backend reduction 이후 작은 차이가 normalization과 후속 block으로 propagation되는 패턴과 consistent with하다. 그러나 BN1은 first non-zero entry point일 뿐 성능 차이가 시작된 지점이나 root cause로 단정하지 않는다.

The repeated BN1 batch-mean difference shows that mathematically aligned BatchNorm semantics do not guarantee bitwise-identical reduction results across the two execution stacks. 이 관찰은 low-level reduction/numerical execution 차이와 consistent with하지만 Experiment 08은 backend mechanism 자체를 격리하지 않았다.

## Initial vs Late-Stage and Anchor Sensitivity

First non-zero forward stage는 initial과 late-stage 모두 BN1 batch mean으로 동일했다. Forward 최대값 역시 모두 `1e-7` 규모였고 대체로 deeper block에 위치했다. 반면 backward와 update의 최대 stage/group은 checkpoint와 anchor에 따라 달랐다.

Epoch 1 Seed 42의 K/P anchor는 최대 forward와 parameter/update group이 같았지만 global gradient/update 크기는 달랐다. Epoch 30에서는 anchor sensitivity가 더 뚜렷했다.

- Seed 123: K-anchor global gradient/update/post-weight `1.14e-4 / 4.45e-5 / 1.19e-7`, P-anchor `9.26e-6 / 8.03e-7 / 8.31e-9`
- Seed 2026: K-anchor `7.34e-4 / 4.07e-4 / 1.01e-6`, P-anchor `9.17e-6 / 8.53e-7 / 8.23e-9`

K/P anchor에 따라 largest forward, backward, parameter-gradient 및 update group도 달라졌다. 이는 도달한 parameter-space location에 따른 one-step numerical sensitivity와 consistent with하지만, K-anchor 자체나 특정 Framework가 원인이라는 뜻은 아니다.

## Hypothesis Evaluation

- **H1:** 지지된다. 9개 synchronized case 모두 first observed numerical divergence가 BN1 batch mean에서 반복됐다.
- **H2:** 부분적으로 지지된다. Forward는 `1e-7` 규모였지만 activation/parameter gradient relative L2는 case에 따라 더 커졌다. 다만 공통 단일 backward layer는 없었다.
- **H3:** Initial case에서 강한 local update/gradient ratio가 관찰됐지만 Epoch 1/30에서는 동일 패턴이 지속되지 않았다. Common Adam은 state와 parameter group에 따라 작은 gradient difference를 다르게 변환했다.
- **H4:** 지지된다. 특히 Epoch 30 Seed 123/2026에서 K/P anchor의 global 크기와 최대 stage/group이 달랐다.

## Answer to the Experiment 08 Research Question

Exact synchronized state와 동일 fixed batch에서 Conv1 output까지는 exact였고, 9개 case 모두 최초의 비영 수치 차이는 Common BN1의 batch-mean reduction에서 관찰됐다. 이 작은 차이는 후속 BN normalization과 deeper block으로 전파됐지만 forward relative L2는 계속 `1e-7` 규모였고 단조 증가하지 않았다.

Backward에서는 더 큰 relative divergence가 관찰됐으나 최대 stage와 parameter group은 Seed, checkpoint와 anchor에 따라 달랐다. Common Adam은 특히 initial case의 일부 parameter group에서 gradient difference를 더 큰 update relative difference로 변환했지만, 이 패턴은 late-stage 전반에 일관되지 않았다. Epoch 30의 K/P-anchor 차이는 one-step sensitivity가 학습으로 도달한 parameter-space location에 의존할 가능성을 시사한다.

따라서 Experiment 08은 **BN1 batch-mean reduction을 재현 가능한 first observed numerical divergence entry point로 식별**하고, 작은 차이가 forward/backward/update를 거치며 state-dependent하게 전파·국소 확대되는 위치를 기록했다. 이를 BN, backend reduction, autograd 또는 optimizer가 단독 root cause라는 증거로 해석하지 않는다. Experiment 07의 큰 free-running separation과 연결하면, 매 step의 작은 local difference가 이미 달라진 state에 반복적으로 피드백되는 누적 과정이 현재 04–08 결과와 가장 잘 맞는다.

Under exact full-state synchronization, all nine analyzed cases produced identical input and Conv1 output. 이후 first observed numerical difference는 BN1 batch-mean reduction에서 일관되게 나타났고, forward maximum은 `1e-7` order에 머물렀다. Backward와 optimizer-update divergence는 더 강한 case/anchor dependence를 보였으며 모든 case를 지배하는 단일 layer, parameter 또는 update group은 없었다. 따라서 BN1 batch mean은 테스트 환경의 reproducible numerical entry point이지만 후속 amplification은 current parameter-space state에 의존한다. 이 결과는 BatchNorm, autograd, optimizer 또는 backend implementation 중 하나를 unique root cause로 확정하지 않는다.

## Limitations

- 고정 diagnostic batch 하나와 selected checkpoint 9개만 분석했다.
- 같은 Apple GPU에서도 TensorFlow Metal과 PyTorch MPS backend를 사용한다.
- Relative metric은 absolute norm이 작은 gradient에서 커질 수 있으므로 각 CSV에 양쪽 norm을 함께 기록했다.
- First non-zero, largest divergence와 amplification은 서로 다른 개념이며 어느 것도 단독으로 root cause를 입증하지 않는다.
- Experiments 04–08의 종합 결론과 Phase 2 종료 판단은 [Phase 2 final conclusion](../../PHASE2_STRICT_CONTROLLED_CONCLUSION.md)에 기록했다.

## Artifacts and Reproduction

주요 결과는 `results/preflight/`, `results/forward/`, `results/backward/`, `results/optimizer/`, `results/bn/`, `results/summaries/`, `results/manifests/`에 있다. 기존 결과를 보존하기 위해 artifact writer는 내용이 다른 파일을 overwrite하지 않는다.

```bash
.venv-metal/bin/python experiments/08_layer_by_layer_trajectory/layer_trace_preflight.py
.venv-metal/bin/python experiments/08_layer_by_layer_trajectory/layer_trace.py
.venv-metal/bin/python experiments/08_layer_by_layer_trajectory/summarize_trace.py
```

세 명령은 selected one-step diagnostic만 수행하며 Full Training을 시작하지 않는다. 이전 미실행 설계는 `archive/pre_redesign/`에 보존되어 있고 공식 Phase 2 결과로 사용하지 않는다.

## Phase 2 Status

Experiment 08은 **Completed / VALID**이며, Experiments 04–08을 포함한 **Phase 2 - Strict Controlled Framework Comparison은 Completed**다. Experiment 09는 생성하지 않았다.
