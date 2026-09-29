# 외부 타당성 검증 연구 계획

## 목적

Phase 2(Experiments 04–08)는 Young AffectNet HQ, custom CNN+BN, Metal/MPS, float32 조건에서 완료됐으며 VALID하다. External Validation은 주요 요인을 하나씩 변경했을 때 관련 수치 분기 관찰이 유지되는지 검증하는 별도 연구다. Phase 2를 다시 열거나 기존 결론을 수정하지 않는다.

## 전체 연구질문

> Phase 2에서 관찰된 수치 분기 패턴은 테스트한 GPU 실행 스택, 아키텍처, BatchNorm 기반 설계 및 dataset에만 해당하는가? 아니면 해당 조건을 변경해도 관련 패턴이 유지되는가?

## 검증 순서와 요인 격리

| 순서 | 검증 | 주요 변경 요인 | 초기 범위 | 상태 |
|---:|---|---|---|---|
| V1 | CPU-only 실행 | 실행 장치/경로 | Experiment 08과 동일한 9개 synchronized one-step case | **Completed / VALID** |
| V2 | ResNet18 + BatchNorm | 아키텍처 | Diagnostic 전 설계·정렬 preflight | **Next** |
| V3 | BatchNorm 없는 CNN | 정규화 아키텍처 | Diagnostic 전 설계·정렬 preflight | Planned |
| V4 | CIFAR-10 | Dataset/workload | Diagnostic 전 계획 및 data manifest | Planned |

각 validation에서는 가능한 한 하나의 요인만 변경한다. 피할 수 없는 부수 차이는 limitation으로 명시한다.

## V1 수행 절차

1. TensorFlow와 PyTorch 연산을 CPU로 강제하고 지원되는 intra/inter-op thread를 1로 설정한다.
2. Probe 전에 실제 연산 장치를 검증한다.
3. 기존 Experiment 07/08 fixed batch를 persisted ID, label, augmentation metadata, float32 tensor 및 전체 hash와 비교한다. 새 batch를 선택하지 않는다.
4. Experiment 08과 정확히 동일한 9개 case 및 Experiment 06 checkpoint를 읽기 전용으로 재사용한다.
5. Canonical weight, BN affine/running state, CommonAdam m/v/step, input 및 label을 exact synchronization한다.
6. Fresh non-instrumented one-step과 layer-traced one-step을 비교한다.
7. 각 framework에서 독립적인 traced 실행을 두 번 반복한다.
8. Experiment 08과 같은 정의로 forward, backward, parameter-gradient, optimizer, BN 및 global metric을 기록한다.
9. V1과 Experiment 08 case summary를 직접 연결해 비교한다.
10. 학습, validation/test 평가 및 checkpoint 선택은 수행하지 않는다.

## V1 유효성 검사 기준

```text
CPU_ONLY_DEVICE_CHECK = VALID
V1_RESYNC_PRECHECK = VALID
CPU_REPEATABILITY = VALID
CPU_TRACE_EQUIVALENCE = VALID
V1_SELECTED_CASES_COMPLETE = TRUE
NAN_INF_FOUND = FALSE
NEW_FULL_TRAINING_EXECUTED = FALSE
```

Gate가 실패하면 diagnostic 근거는 보존하되 V1을 invalid/incomplete로 표시하고, 유효하지 않은 layer trace는 연구결론에 사용하지 않는다.

## 해석 원칙

V1은 테스트한 GPU 실행 경로와 CPU 실행을 구분하지만 TensorFlow와 PyTorch에 동일 backend를 제공하지는 않는다. 진입점이 유지되면 테스트 조건에서 장치 간 재현성을 지지한다. 진입점이 바뀌면 execution-stack sensitivity를 시사한다. 거의 또는 완전히 exact한 결과가 나온다면 제거된 GPU 경로가 해당 환경의 one-step 차이에 실질적으로 기여했을 가능성을 시사하지만, 고유 원인을 증명하지는 않는다.

V2–V4는 구현 전에 각각의 preflight 명세를 확정해야 한다. 현재 승인된 External Validation 범위에는 Full Training이 없다.

## V1 결과

9개의 CPU-only case는 모든 validity gate를 통과했다. CPU의 최초 비영 forward stage는 9/9 case에서 Conv1이었고, Experiment 08 GPU에서는 9/9 case에서 BN1 batch mean이었다. CPU maximum-forward relative L2는 paired GPU 값보다 `49.9×–148.6×` 컸다.

따라서 V1은 Phase 2의 최초 진입점 관찰 범위를 테스트한 execution stack으로 좁힌다. 동시에 통제된 framework 간 수치 차이의 전파를 외부 조건에서도 검증해야 한다는 더 넓은 연구 동기는 유지된다. 실제 layer/group 결과는 [V1 보고서](../experiments/external_validation/V1_cpu_only/README.md)에 정리했다.
