# 외부 타당성 검증 연구 계획

## 목적

Phase 2(Experiments 04–08)는 Young AffectNet HQ, custom CNN+BN, Metal/MPS, float32 조건에서 완료됐으며 VALID하다. External Validation은 주요 요인을 하나씩 변경했을 때 관련 수치 분기 관찰이 유지되는지 검증하는 별도 연구다. Phase 2를 다시 열거나 기존 결론을 수정하지 않는다.

## 전체 연구질문

> Phase 2에서 관찰된 수치 분기 패턴은 테스트한 GPU 실행 스택, 아키텍처, BatchNorm 기반 설계 및 dataset에만 해당하는가? 아니면 해당 조건을 변경해도 관련 패턴이 유지되는가?

## 검증 순서와 요인 격리

| 순서 | 검증 | 주요 변경 요인 | 초기 범위 | 상태 |
|---:|---|---|---|---|
| V1 | CPU-only 실행 | 실행 장치/경로 | Experiment 08과 동일한 9개 synchronized one-step case | **Completed / VALID** |
| V2 | ResNet18 + BatchNorm | 아키텍처 | 동일 GPU stack의 3-Seed initial synchronized one-step | **Completed / VALID** |
| V3 | BatchNorm 없는 CNN | 정규화 아키텍처 | Diagnostic 전 설계·정렬 preflight | **Next** |
| V4 | CIFAR-10 | Dataset/workload | Diagnostic 전 계획 및 data manifest | **Planned** |

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

V3–V4는 구현 전에 각각의 preflight 명세를 확정해야 한다. 현재 승인된 External Validation 범위에는 Full Training이 없다.

## V1 결과

9개의 CPU-only case는 모든 validity gate를 통과했다. CPU의 최초 비영 forward stage는 9/9 case에서 Conv1이었고, Experiment 08 GPU에서는 9/9 case에서 BN1 batch mean이었다. CPU maximum-forward relative L2는 paired GPU 값보다 `49.9×–148.6×` 컸다.

따라서 V1은 Phase 2의 최초 진입점 관찰 범위를 테스트한 execution stack으로 좁힌다. 동시에 통제된 framework 간 수치 차이의 전파를 외부 조건에서도 검증해야 한다는 더 넓은 연구 동기는 유지된다. 실제 layer/group 결과는 [V1 보고서](../experiments/external_validation/V1_cpu_only/README.md)에 정리했다.

## V2 결과

동일 Metal/MPS GPU 실행환경과 fixed batch/Common Adam/Common BN을 유지하고 architecture만 직접 구현한 semantic-equivalent ResNet18로 변경했다. Framework별 trainable parameter는 `11,180,616`개였고 3개 Seed의 canonical state mapping 306/306개가 exact였다. 3개 Seed 모두 stem Conv는 exact였고 최초 비영 stage는 `stem.bn.batch_mean`이었다. Initial one-step maximum-forward relative L2는 `4.09e-6–4.23e-6`이었고, Custom CNN의 대응 maximum보다 Seed별 `7.03×–7.34×` 컸다. Residual Add는 최초 진입점이 아니었으며 downstream maximum의 위치와 크기는 architecture 및 Seed에 민감했다. 상세 결과와 metric 정의는 [V2 보고서](../experiments/external_validation/V2_resnet18_bn/README.md)에 정리했다.

## V1–V2 중간 결론과 다음 격리 질문

V1 CPU custom CNN은 9/9 case에서 Conv1, Phase 2 GPU custom CNN은 9/9 case에서 BN1 batch mean, V2 GPU ResNet18은 3/3 Seed에서 stem BN batch mean이 최초 비영 stage였다. 따라서 exact entry point는 execution stack에 민감하며, 테스트한 GPU의 BN batch-mean entry가 두 architecture에서 반복된 것과 downstream propagation이 architecture-dependent한 것을 함께 보고해야 한다. 어느 관찰도 특정 연산이나 backend의 보편적 인과성을 확정하지 않는다.

V3의 질문은 **“Custom CNN과 GPU 실행환경을 그 밖에는 유지한 채 BatchNorm을 제거하면 최초 수치 차이는 어디로 이동하는가?”**이다. V3는 **Next**, 독립 dataset/workload를 다루는 V4는 **Planned** 상태를 유지한다.
