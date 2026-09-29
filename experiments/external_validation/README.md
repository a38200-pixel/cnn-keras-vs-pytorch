# 외부 타당성 검증 연구

## 연구 동기

Phase 2에서는 Young AffectNet HQ, BatchNorm을 사용하는 custom CNN, float32, TensorFlow/Keras Metal 및 PyTorch MPS라는 하나의 엄격한 통제 환경에서 반복 가능한 최초 수치 차이 지점을 찾고 그 전파 과정을 추적했다. 해당 결과는 **Completed / VALID** 상태이지만, 같은 패턴이 실행 장치·아키텍처·정규화 설계·워크로드를 바꿔도 일반화된다고 볼 수는 없다.

External Validation은 Phase 2 결론을 다시 열거나 사후 변경하지 않고 그 외적 타당성을 검증하는 별도 연구 단계다.

## Phase 2와의 관계

Experiments 04–08의 코드, checkpoint, 결과 artifact와 [Phase 2 종합 결론](../../PHASE2_STRICT_CONTROLLED_CONCLUSION.md)은 읽기 전용 근거로 유지한다. 검증 결과는 기존 관찰을 지지하거나 적용 범위를 좁히고 보완할 수 있지만, 기존 결과를 대체하지 않는다.

## 연구질문

> Phase 2에서 관찰된 수치 분기 패턴은 테스트한 GPU 실행 스택, 아키텍처, BatchNorm 기반 설계 및 dataset에만 해당하는가? 아니면 해당 조건을 변경해도 관련 패턴이 유지되는가?

- **V1 — CPU-only 실행:** Metal/MPS GPU 실행 경로를 제거하면 최초 수치 차이는 어디에서 나타나는가?
- **V2 — ResNet18 + BatchNorm:** 표준 residual architecture에서도 관련 수치 분기 진입점이 관찰되는가?
- **V3 — BatchNorm 없는 Small CNN:** BatchNorm이 없을 때 최초 수치 진입점은 어디로 이동하는가?
- **V4 — CIFAR-10:** 독립적인 dataset/workload에서도 관련 수치 분기 패턴이 관찰되는가?

## 검증 구성

| 검증 | 변경 요인 | 고정 요인 | 핵심 질문 | 상태 |
|---|---|---|---|---|
| [V1 CPU-only 실행](V1_cpu_only/README.md) | 실행 장치 | Young AffectNet HQ, custom CNN+BN, Common Adam, Common BN, 동일 state/batch/case | Metal/MPS 없이도 수치 분기가 유지되는가? | **Completed / VALID** |
| [V2 ResNet18 + BN](V2_resnet18_bn/README.md) | 아키텍처 | BN 사용 및 framework 간 통제 방법론 | Residual architecture에서도 패턴이 유지되는가? | **Next** |
| [V3 BN-free CNN](V3_bn_free_cnn/README.md) | 정규화 아키텍처 | Small CNN 계열 및 통제 방법론 | BN이 없을 때 최초 진입점은 어디인가? | Planned |
| [V4 CIFAR-10](V4_cifar10/README.md) | Dataset/workload | 앞 단계에서 검증된 아키텍처 및 통제 방법론 | 독립 workload에서도 패턴이 유지되는가? | Planned |

각 validation에서는 가능한 한 한 종류의 요인만 변경한다. V2–V4는 한 번에 여러 요인이 바뀌지 않도록 앞 단계에서 검증된 설계를 기준으로 순차 확정한다.

## 해석 원칙

- 실제 첫 비영(non-zero) stage를 보고하며 BN1이라고 가정하지 않는다.
- CPU-only는 TensorFlow CPU와 PyTorch CPU의 framework-native 구현을 의미한다. 두 framework가 동일 kernel/backend를 사용한다는 뜻이 아니다.
- 최초 진입점의 이동은 실행 스택·아키텍처·정규화·워크로드 민감성을 시사하지만, 하나의 고유 원인을 확정하지 않는다.
- 진입점이 유지되더라도 테스트한 조건에서의 재현성을 지지할 뿐 보편적 인과성을 증명하지 않는다.
- Parameter divergence와 성능 우열은 별개의 질문이다.
- GPU, Metal, MPS, BatchNorm, TensorFlow 또는 PyTorch가 본질적으로 잘못됐거나 우월하다고 표현하지 않는다.

## 타당성 위협과 한계

- V1은 Experiment 08과 동일한 단일 진단 batch 및 선택된 9개 case를 사용한다.
- TensorFlow CPU와 PyTorch CPU는 여전히 서로 다른 실행 스택이다.
- 초기 검증 범위는 synchronized one-step이며 장기 학습이나 일반화 성능이 아니다.
- 결과는 float32, 3개 초기 Seed 및 사용 가능한 checkpoint로 제한된다.
- V2–V4 실행 전에는 추가 정렬 조건과 불가피한 혼란 변수를 별도로 문서화해야 한다.

## 진행 순서

1. V1 CPU-only synchronized one-step을 수행하고 Experiment 08과 case별로 직접 비교한다.
2. 검증된 절차를 V2 ResNet18 + BN에 적용한다.
3. V3에서 BatchNorm 존재 여부를 격리한다.
4. 모델 측 검증 이후 V4에서 dataset/workload를 변경한다.

이번 단계에서는 V1 Full Training과 V2–V4 구현·실행을 수행하지 않았다.

## V1 결과

V1은 CPU 배치, 기존 batch/state의 exact synchronization, framework 내부 반복 재현성, trace 동등성, case 완전성 및 유한값 검사를 모두 통과했다. Experiment 08 GPU의 9개 case 모두에서 BN1 batch mean이 최초 차이였던 것과 달리 CPU에서는 9개 모두 Conv1에서 최초 차이가 나타났다. CPU의 paired maximum-forward relative L2는 GPU 값보다 `49.9×–148.6×` 컸다.

따라서 GPU의 최초 진입점은 실행 스택에 민감한 관찰로 적용 범위를 좁혀야 한다. 반면 exact synchronized boundary state에서도 작은 framework 간 수치 차이가 발생하고 한 step 내부로 전파될 수 있다는 더 넓은 관찰은 유지된다.
