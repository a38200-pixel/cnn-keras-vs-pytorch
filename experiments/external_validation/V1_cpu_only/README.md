# V1 - CPU-only 실행 검증

## 상태

**Completed / VALID**

## 연구질문

> 동일하게 동기화된 Keras와 PyTorch state를 Metal/MPS GPU 경로 대신 CPU에서 실행하면 최초 수치 차이는 어디에서 나타나며, 그 전파 양상은 Experiment 08과 어떻게 다른가?

V1은 테스트한 GPU 실행 경로를 제거해도 Phase 2의 수치 패턴이 유지되는지 검증한다. TensorFlow CPU와 PyTorch CPU도 서로 다른 framework-native kernel을 사용할 수 있으므로, 이 결과만으로 공통 backend 사용, GPU의 단독 원인 여부 또는 수치 정확성의 우열을 판단할 수 없다.

## 변경 조건과 고정 조건

| 구분 | V1 조건 |
|---|---|
| 변경 요인 | TensorFlow Metal → TensorFlow CPU, PyTorch MPS → PyTorch CPU |
| Dataset/model | 동일한 Young AffectNet HQ 고정 split과 custom CNN+Common BN |
| Optimizer | 동일한 CommonAdam float32 구현 |
| State | Weight, BN affine/running state, Adam m/v/step exact synchronization |
| Input | Experiment 07/08과 동일한 고정 diagnostic batch |
| Case | Experiment 08과 동일한 9개 selected case |
| Trace | 동일한 semantic naming, canonical layout 및 metric 정의 |
| Thread | TensorFlow intra/inter-op 1, PyTorch intra/inter-op 1 |
| 범위 | Synchronized one-step diagnostic만 수행, 학습·평가 없음 |

`src/v1_config.py`의 `RUN_FULL_TRAINING = False`를 유지한다. V1 실행 파일에는 Full Training loop가 없다.

## 실행환경

[실행환경 manifest](results/manifests/cpu_environment.json)에 TensorFlow 2.18.1/Keras 3.12.4의 실제 장치 `/device:CPU:0`과 PyTorch 2.14.0의 실제 장치 `cpu`를 기록했다. TensorFlow에서 보이는 GPU는 0개였고 PyTorch MPS tensor는 사용되지 않았다. TensorFlow와 PyTorch의 intra/inter-op thread 수는 모두 1로 고정했다. `tensorflow-metal` package는 설치된 상태였지만 연산에는 사용하지 않았다.

## 검증 사례 선택

Experiment 08 case manifest를 새로 선정하지 않고 그대로 재사용했다.

- Initial shared-anchor: Seeds 42, 123, 2026
- Epoch 1 low-divergence control: Seed 42의 K-anchor와 P-anchor
- Epoch 30 sensitivity: Seeds 123, 2026의 K-anchor와 P-anchor

Experiment 06 checkpoint, Experiment 07 synchronization/batch utility 및 Experiment 08 trace utility는 V1 내부의 얇은 adapter를 통해 읽기 전용으로 사용했다.

## 사전 검증

Trace 실행 전에 CPU 배치, 기존 fixed-batch identity와 전체 hash, NHWC↔NCHW 복원, case 선택 및 9개 case의 weight, BN gamma/beta/running mean/running variance, CommonAdam m/v/step, input, label exact synchronization을 확인했다.

모든 gate가 통과했다.

- `CPU_ONLY_DEVICE_CHECK = VALID`
- `V1_RESYNC_PRECHECK = VALID`
- Experiment 07/08 batch manifest와 재구성한 float32 값이 exact였으며 전체 hash는 `c3ef4b5c6cb842154ddb03fa6cdc175dd5ab4e69a1e02f2bbc5493c3e2a48e7c`다.
- 9개 case 모두 Experiment 08의 case ID, source-state hash, checkpoint step 및 full-state synchronization 결과와 일치했다.

상세 근거는 [장치 검사](results/preflight/cpu_device_check.json), [동기화 사전 검사](results/preflight/resync_precheck.json), [case manifest](results/manifests/v1_case_manifest.csv)에 있다.

## CPU 반복 재현성

각 framework에서 동일한 fresh state로 독립적인 traced one-step을 두 번 실행했다. Logits, loss, global gradient/update, post-step weight 및 BN running state를 비교했다.

`CPU_REPEATABILITY = VALID`: 9개 case와 두 framework에서 수행한 126개 비교가 모두 exact였다.

## 계층 추적 계측 동등성

각 case에서 fresh state로 실행한 non-instrumented synchronized CPU one-step과 layer-traced one-step을 비교했다. Loss, global gradient, CommonAdam update, post-step weight 및 BN post-state가 일치해야 layer 결과를 유효하게 사용하도록 했다.

`CPU_TRACE_EQUIVALENCE = VALID`: 126개 plain-vs-traced 비교가 모두 exact였다. 따라서 trace는 측정 대상 one-step에 두 번째 forward 또는 추가 BN state update를 만들지 않았다.

## 순전파 결과

모든 case에서 canonical input은 exact였다. Experiment 08 GPU와 달리 CPU의 최초 비영 stage는 **9/9 case 모두 Conv1**이었다. Conv1 relative L2는 약 `1.49e-7–2.02e-7`로 작지만 0은 아니었으므로 최초 진입점이 앞 단계로 이동했다. 최대 forward 차이는 6개 case에서 BN1 batch variance, 3개 case에서 BN2 batch variance였고 relative L2는 `1.70e-5–5.33e-5`였다.

| 검증 사례 | 최초 비영 stage | 최대 순전파 차이 (relative L2) | 최대 역전파 stage | 최대 parameter gradient | 최대 update |
|---|---|---:|---|---|---|
| Initial 42 / shared | Conv1 | BN1 batch variance (`3.29e-5`) | Conv1 | Conv1 kernel | Conv2 kernel |
| Initial 123 / shared | Conv1 | BN1 batch variance (`3.28e-5`) | Conv1 | BN1 beta | Conv3 kernel |
| Initial 2026 / shared | Conv1 | BN1 batch variance (`2.88e-5`) | Conv1 | BN1 beta | Conv1 kernel |
| Epoch 1, 42 / K | Conv1 | BN1 batch variance (`1.70e-5`) | Conv1 | BN3 beta | Conv3 kernel |
| Epoch 1, 42 / P | Conv1 | BN1 batch variance (`2.96e-5`) | Conv1 | BN3 beta | BN3 beta |
| Epoch 30, 123 / K | Conv1 | BN2 batch variance (`5.33e-5`) | BN1 | BN1 beta | BN1 beta |
| Epoch 30, 123 / P | Conv1 | BN2 batch variance (`5.19e-5`) | BN1 | BN1 beta | BN1 beta |
| Epoch 30, 2026 / K | Conv1 | BN2 batch variance (`3.97e-5`) | BN1 | BN1 beta | BN1 beta |
| Epoch 30, 2026 / P | Conv1 | BN1 batch variance (`4.67e-5`) | Conv1 | BN1 beta | BN1 beta |

전체 stage metric은 `results/forward/`에 있으며 공식 요약은 [case_summary.csv](results/summaries/case_summary.csv)다.

## 역전파 결과

최대 activation-gradient 차이는 6개 case에서 Conv1, 3개 case에서 BN1이었으며 relative L2는 `2.90e-2–1.07e-1`이었다. GPU의 최대 지점에는 Conv1, BN1, BN2, BN3가 포함됐으므로 CPU의 case별 최대 backward 위치는 GPU와 동일하게 유지되지 않았다.

최대 parameter-gradient group은 BN1 beta 6개, BN3 beta 2개, Conv1 kernel 1개였다. Global gradient relative L2는 `1.76e-4–1.86e-3`였다. 이들은 관찰된 민감 지점이며 고유 원인으로 해석하지 않는다.

## 옵티마이저 결과

최대 CommonAdam update group은 BN1 beta 4개, Conv3 kernel 2개였고 Conv1 kernel, Conv2 kernel, BN3 beta가 각각 1개였다. Global update relative L2는 `1.81e-4–2.98e-2`, post-weight relative L2는 `5.98e-7–6.16e-4`였다.

CPU에서도 K/P anchor sensitivity가 유지됐다. Epoch 1 Seed 42의 K/P 최대 forward relative L2는 `1.70e-5/2.96e-5`였고 최대 update group도 Conv3 kernel에서 BN3 beta로 이동했다. Epoch 30 Seed 2026에서는 K-anchor의 최대 forward/backward 위치가 BN2 variance/BN1이었지만 P-anchor에서는 BN1 variance/Conv1이었다. Seed 123도 최대 group은 같았지만 global update 크기가 anchor에 따라 달랐다.

## CPU와 GPU 비교

[CPU-vs-GPU summary](results/summaries/cpu_vs_gpu_summary.csv)의 직접 비교 결과는 다음과 같다.

- GPU Experiment 08: 9/9 case에서 BN1 batch mean이 최초 비영 stage였다.
- CPU V1: 9/9 case에서 Conv1이 최초 비영 stage였다.
- CPU maximum-forward relative L2는 paired GPU 값의 `49.9×–148.6×`, 평균 `97.0×`였다. CPU 범위는 `1.70e-5–5.33e-5`, GPU 범위는 `2.46e-7–5.82e-7`이다.
- 최대 backward, parameter-gradient 및 update 위치는 case별로 그대로 유지되지 않았다. 일부 case는 일치했지만 다수는 다른 stage/group으로 이동했다.
- CPU에서도 anchor에 따른 크기 차이가 나타났고 일부 case에서는 최대 위치도 달라졌다.

이 비교는 서로 다른 두 framework-native CPU 경로와 서로 다른 두 GPU 경로 사이의 비교다. Backend의 수치 정확성 순위를 의미하지 않는다.

## V1 연구질문에 대한 답

Phase 2 GPU의 최초 진입점은 CPU에서 동일하게 유지되지 않았다. Metal/MPS 실행 경로를 제거하자 모든 selected case의 반복 가능한 최초 수치 차이가 BN1 batch-mean reduction에서 Conv1으로 이동했고, 이 진단에서 CPU forward 최대 차이는 GPU보다 훨씬 컸다. Backward와 optimizer의 민감 지점도 달라졌다.

따라서 V1은 Phase 2 관찰의 적용 범위를 **보완하고 좁힌다(qualify and narrow)**. BN1 batch mean은 테스트한 Metal/MPS stack에서 반복 재현된 진입점이지만, 아키텍처에만 의존하거나 장치와 무관한 보편적 근거는 아니다. 다만 exact synchronized model-boundary input과 state에서도 작은 framework-native 수치 차이가 발생하고 one-step 계산을 따라 전파될 수 있다는 더 넓은 관찰은 유지된다.

이 결과는 execution-stack sensitivity와 일치하지만 GPU가 단독 원인임을 보이거나 CPU의 정확성이 더 높거나 낮음을 의미하지 않는다.

## 한계

- CPU-only는 Metal/MPS 경로를 제거하지만 두 framework에 동일한 CPU 구현을 제공하지 않는다.
- 동일 fixed batch와 selected checkpoint는 직접 비교 가능성을 높이지만 workload 범위를 제한한다.
- One-step 동작만으로 장기 trajectory 또는 generalization을 예측할 수 없다.
- Low-level CPU reduction tree와 kernel 구현을 직접 계측하지 않았다.

## 다음 검증

이후 계획했던 V2–V4는 모두 완료됐다. V1의 실행 당시에는 후속 validation을 실행하지 않았다는 provenance를 유지하며, 최종 상태는 상위 [External Validation README](../README.md)에서 관리한다.

## 최종 검증 상태

```text
CPU_ONLY_DEVICE_CHECK = VALID
V1_RESYNC_PRECHECK = VALID
CPU_REPEATABILITY = VALID
CPU_TRACE_EQUIVALENCE = VALID
V1_SELECTED_CASES_COMPLETE = TRUE
NAN_INF_FOUND = FALSE
NEW_FULL_TRAINING_EXECUTED = FALSE
V1 STATUS = Completed / VALID
```
