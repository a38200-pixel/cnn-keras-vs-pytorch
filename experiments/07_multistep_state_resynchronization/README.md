# Experiment 07 - Multi-Step Divergence & State Re-Synchronization

## Research Question

> Common Adam과 Common BN까지 통제한 상태에서 장기 trajectory divergence는 매 step 새롭게 발생하는 cross-framework numerical difference 때문인가, 아니면 이미 발생한 parameter/optimizer/BN-state 차이가 다음 step에 다시 입력되면서 증폭되는 feedback과 더 일치하는가?

## Motivation from Experiments 04-06

- Experiment 04는 Same W0/input과 Conv1 exact 이후 BN1에서 첫 미세 non-zero difference, gradient difference, native Adam update amplification과 장기 trajectory separation을 관찰했다.
- Experiment 05는 Common Adam으로 first-step divergence를 크게 줄였지만 장기 divergence 대부분이 다시 나타났다.
- Experiment 06은 Common Adam과 Common BN으로 running-state implementation 차이를 대부분 제거했지만, 초기 감소 효과가 몇 step 뒤 사라졌고 Epoch 30 parameter divergence는 05와 거의 같았다.

## Difference from Experiment 06

Experiment 07은 새로운 30-epoch training이 아니다. Experiment 06의 `initial`, `after_first_step`, `epoch_001/005/010/020/030` canonical checkpoint를 read-only로 clone하고, 모든 checkpoint에 동일한 fixed diagnostic batch를 넣어 checkpoint-local one step만 수행한다.

## Hypotheses

- **H1:** Synchronized one-step divergence는 매우 작은데 free-running divergence만 커진다면, 이전에 누적된 state difference가 이후 step에 재입력되는 feedback amplification과 일치하는 결과다.
- **H2:** Re-synchronization 이후에도 후반 checkpoint에서 one-step gradient/update divergence가 크게 증가한다면, 동일 state에서 새로 생성되는 framework-native numerical/autograd difference가 도달한 parameter-space region에 의존할 가능성을 시사한다.

두 항목은 사전 가설이며 어느 mechanism도 결과만으로 원인으로 확정하지 않는다.

## Experimental Design

### Free-Running Probe

각 Framework가 Experiment 06에서 실제 도달한 `W/Adam m,v/BN state`를 그대로 load한다. 같은 batch로 training-mode one step을 수행해 pre/post global weight, Adam m/v, BN state divergence와 loss/gradient/update difference를 측정한다.

### Full-State Re-Synchronized Probe

Model parameters, Adam m/v와 step, BN gamma/beta/running mean/variance를 양쪽 Framework에 exact하게 주입한다. 동일 batch에서 native TensorFlow `GradientTape`와 PyTorch autograd로 한 step을 수행하며, 이 결과를 **state-controlled one-step divergence**로 해석한다.

### Keras / PyTorch Anchor Strategy

이미 분리된 중간 checkpoint에서 한 Framework만 기준으로 삼지 않는다. 각 checkpoint마다 Keras state를 양쪽에 주입한 K-anchor와 PyTorch state를 양쪽에 주입한 P-anchor를 모두 실행하고 별도 행으로 저장한다. K/P state가 exact인 initial은 shared anchor 한 번만 실행한다.

## Fixed Diagnostic Batch

`probe_id=exp07_fixed_probe_v1`로 Train split에서 32개 sample ID를 hash 기반으로 고정 선택한다. 고정 augmentation key를 사용한 float32 image/label을 모든 Seed/checkpoint/anchor가 공유한다. 이는 실제 training continuation batch가 아니라 checkpoint-local numerical probe다.

## Re-Synchronization State

One-step 실행 전에 model weights, Adam m/v/step, BN gamma/beta/running state, input과 label의 exact equality를 확인한다. 하나라도 실패하면 `RESYNC_PRECHECK=INVALID`로 probe를 중단한다.

## Metrics

- Forward: logits max-abs/relative L2, loss absolute difference
- Backward: global gradient relative L2, cosine similarity, max-abs difference
- CommonAdam: update relative L2, Adam m/v relative L2
- State: pre/post global weight relative L2와 delta, BN running mean/variance relative L2
- Safety: NaN/Inf

Experiment 07은 global mechanism 분리에 집중한다. 상세 layer-by-layer activation/gradient/update trace는 Experiment 08에 남긴다.

## Checkpoints

| Checkpoint | Optimizer step |
|---|---:|
| initial | 0 |
| after_first_step | 1 |
| epoch_001 | 321 |
| epoch_005 | 1,605 |
| epoch_010 | 3,210 |
| epoch_020 | 6,420 |
| epoch_030 | 9,630 |

## Validity Criteria

- Experiment 06 summary와 42개 checkpoint integrity가 VALID여야 한다.
- Fixed batch의 sample ID/image/label과 NHWC↔NCHW roundtrip이 exact여야 한다.
- 모든 39개 Seed/checkpoint/anchor re-synchronization case에서 전체 state가 exact여야 한다.
- 모든 free-running/resynchronized probe가 finite여야 한다.
- Experiment 06 checkpoint는 수정하거나 overwrite하지 않는다.

## Experiment Validity

> **Diagnostic Completed / VALID.** `RESYNC_PRECHECK=VALID`, `PROBE_READY=TRUE`다.

- Seeds 42/123/2026의 `initial`, `after_first_step`, `epoch_001/005/010/020/030`을 사용했다.
- Experiment 06의 Keras/PyTorch checkpoint 42개를 read-only로 검증했다.
- Fixed diagnostic batch는 32개이며 NHWC↔NCHW canonical max-abs difference가 `0`이었다.
- 39개 Seed/checkpoint/anchor case에서 model, Adam m/v/step과 전체 BN state가 exact였다.
- K-anchor와 P-anchor를 모두 실행했으며 initial만 shared anchor를 사용했다.
- Free-running 21개와 re-synchronized 39개 probe가 모두 finite였다.
- 새로운 30-epoch full training과 Test-set model selection은 실행하지 않았다.

## Diagnostic Results

Fixed batch는 32개 unique sample, float32 `(32,128,128,3)`, tensor hash `c3ef4b5c...48e7c`로 고정됐다. 아래 표는 실제 comparison artifact에서 읽은 3-Seed 평균 global metric이다. `Sync Grad/Update/Post Weight`는 K-anchor와 P-anchor를 각각 표시한다.

| Checkpoint | Free pre-weight | Free post-weight | Sync gradient K / P | Sync update K / P | Sync post-weight K / P |
|---|---:|---:|---:|---:|---:|
| initial | 0 | 1.35e−4 | 4.61e−4 / shared | 6.53e−3 / shared | 1.35e−4 / shared |
| after first step | 8.15e−5 | 1.96e−4 | 8.19e−5 / 9.63e−5 | 1.68e−4 / 3.04e−4 | 2.40e−6 / 4.34e−6 |
| epoch 1 | .182288 | .182820 | 1.74e−5 / 1.88e−5 | 5.17e−6 / 2.43e−6 | 2.24e−8 / 1.35e−8 |
| epoch 5 | .552206 | .552409 | 3.84e−5 / 5.43e−5 | 3.96e−5 / 3.13e−5 | 1.51e−7 / 1.29e−7 |
| epoch 10 | .775916 | .776013 | 8.79e−6 / 3.49e−5 | 1.30e−6 / 8.78e−5 | 1.13e−8 / 2.61e−7 |
| epoch 20 | .991490 | .991515 | 1.04e−4 / 1.21e−5 | 2.32e−4 / 3.68e−6 | 6.92e−7 / 1.60e−8 |
| epoch 30 | **1.097571** | **1.097597** | **2.85e−4 / 8.41e−6** | **1.51e−4 / 8.77e−7** | **3.78e−7 / 8.67e−9** |

Epoch 1–30의 free-running gradient relative L2 평균은 `.778→1.052`, update relative L2는 `1.064→1.296` 범위까지 커진 반면, full-state sync 후 post-weight divergence는 대체로 `1e−8–1e−6` 수준이었다. Epoch 30에서도 accumulated pre-weight divergence `1.097571`은 새로 생성된 K/P-anchor post-weight divergence보다 여러 orders of magnitude 컸다. 이는 이미 누적된 parameter/optimizer/BN-state 차이가 이후 계산에 재입력되는 feedback amplification과 **consistent with**하지만, feedback이 원인으로 증명됐다는 뜻은 아니다.

Synchronized divergence는 checkpoint에 따라 단조 증가하지 않았다. 다만 K/P anchor sensitivity는 일부 checkpoint와 Seed에서 분명했다. Epoch 30의 K-anchor 평균 post-weight divergence `3.78e−7`은 P-anchor `8.67e−9`보다 컸고, 주로 K-anchor Seed 123/2026의 spike가 반영됐다. 따라서 도달한 parameter/state region에 따른 local numerical sensitivity 가능성이 남으며, anchor 결과를 평균 하나로 합치지 않는다.

Free-running one step은 이미 형성된 separation을 크게 바꾸지 않았다. 예를 들어 Epoch 30 weight divergence는 `1.097571→1.097597`, delta는 약 `2.57e−5`였다. 따라서 Epoch 30의 큰 separation은 이 probe step 하나가 새로 만든 현상이라기보다 이전 9,630 updates 동안 누적된 state-dependent difference와 일치한다.

## Hypothesis Evaluation

### H1 — Accumulated State-Dependent Feedback

결과는 H1과 전반적으로 강하게 일치한다. Epoch 30 free-running weight divergence는 약 `1.0976`이지만 exact full-state sync 후 새로 생성된 post-step weight divergence는 K-anchor `3.78e−7`, P-anchor `8.67e−9`였다. 동일한 full training state에서 한 step을 다시 시작하면 free-running의 큰 separation이 재현되지 않았다.

이는 이전 step에서 생성된 작은 weights/optimizer/BN-state difference가 다음 step의 입력으로 반복 전달되는 accumulated state-dependent feedback과 **consistent with**하다. 다만 Experiment 07만으로 feedback을 유일한 인과 mechanism으로 확정하지 않는다.

### H2 — Training-Location-Dependent New Divergence

Global metric은 H2를 강하게 지지하지 않는다. Initial shared-anchor post-weight divergence는 `1.35e−4`였지만 Epoch 1 이후 값은 대부분 `1e−8–1e−7` 규모였고, 후반 checkpoint로 갈수록 synchronized one-step divergence가 일관되게 증가하지 않았다.

다만 Epoch 20/30 일부 K-anchor에는 local spike가 있었다. 특히 Epoch 30 Seed 123/2026 K-anchor는 parameter/state location에 따른 local sensitivity 후보이지만, 그 절대 크기는 accumulated free-running divergence보다 훨씬 작다. 이 case는 Experiment 08에서 layer-wise로 격리한다.

## Interpretation Rules

- Free-running과 synchronized 결과를 checkpoint별로 나란히 비교한다.
- K-anchor와 P-anchor를 평균으로 숨기지 않고 각각 보고한다.
- 작은 synchronized divergence와 큰 accumulated divergence는 feedback amplification과 **consistent with**라고만 해석한다.
- 후반 synchronized divergence 증가는 parameter-space location dependence 가능성을 시사하지만 backend/autograd 원인을 확정하지 않는다.
- Accuracy, Test selection 또는 Framework 우열을 평가하지 않는다.

## Answer to the Experiment 07 Research Question

Full-state re-synchronization은 후반 checkpoint에서 새로 생성되는 one-step parameter divergence를 매우 작은 규모로 낮췄지만, free-running model은 크게 분리된 상태로 남았다. 즉 weights, optimizer state와 BN state를 동일하게 맞춘 한 번의 framework-native training step에서는 장기 free-running separation이 재현되지 않았다.

이 관찰은 학습 초기에 생성된 작은 numerical difference가 이후 model/optimizer/BN state를 바꾸고, 변경된 state가 다음 update로 반복 전달되는 accumulated state-dependent feedback과 강하게 일치한다. Native Adam이나 native BatchNorm semantics 어느 하나도 장기 divergence를 단독으로 설명하지 못한다는 Experiments 05–06의 결과와도 연결된다.

그러나 Experiment 07은 유일한 인과 mechanism을 확정하지 않는다. Anchor-dependent local spike의 forward/backward/update 위치는 후속 Experiment 08의 layer-wise isolation에서 분석했다.

## Limitations

- Fixed batch 한 개의 local sensitivity를 측정하므로 전체 training distribution을 대표하지 않을 수 있다.
- Checkpoint state는 Experiment 06의 한 architecture, dataset split과 3 Seeds에 한정된다.
- Re-synchronization은 causal intervention에 가깝지만 backend reduction/autograd의 개별 연산을 분리하지 않는다.
- CommonAdam은 CPU NumPy float32 경로이며 Framework-native backward만 비교한다.

## Current Status

**Diagnostic Completed / VALID.** Free-running 21개와 full-state re-synchronized 39개 one-step probe가 NaN/Inf 없이 완료됐다. Experiment 06 checkpoint는 read-only로 사용했으며 새로운 full training이나 Test-set model selection은 수행하지 않았다.

주요 artifact는 `results/probe_batch_manifest.csv`, `results/preflight/resync_precheck.json`, `results/manifests/resync_state_manifest.csv`, `results/free_running/`, `results/resynchronized/`, `results/comparison/free_vs_resync_summary.csv`, `results/comparison/diagnostic_3seed_summary.json`이다.

## Follow-up Experiment

후속 [Experiment 08 - Layer-by-Layer Training Trajectory Analysis](../08_layer_by_layer_trajectory/README.md)는 아래 representative case의 layer-wise diagnostic을 **Completed / VALID**로 마쳤다.

1. Initial / shared anchor — 상대적으로 큰 synchronized 초기 divergence baseline
2. Epoch 1 — representative low-divergence control
3. Epoch 30 / Seed 123 / K-anchor — late-stage local spike
4. Epoch 30 / Seed 2026 / K-anchor — late-stage local spike
5. 동일 Epoch 30 Seed의 P-anchor — anchor control

각 case의 Conv → BN → ReLU → Pool → FC → Logits → Loss와 layer-wise gradient/update를 추적한 결과, 9개 case 모두 Conv1까지 exact였고 최초 비영 차이는 BN1 batch mean에서 관찰됐다. Forward 차이는 계속 `1e-7` 규모였지만 backward/update의 최대 위치는 checkpoint와 anchor에 따라 달랐다. 이는 Experiment 07의 accumulated state-dependent feedback 해석과 연결되지만 특정 layer를 root cause로 확정하지 않는다.

Experiments 04–08의 종합 해석과 종료 상태는 [Phase 2 final conclusion](../../PHASE2_STRICT_CONTROLLED_CONCLUSION.md)에 기록했다.
