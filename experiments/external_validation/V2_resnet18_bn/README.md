# V2 - ResNet18 + BatchNorm 검증

## 상태

**Next**

## 연구질문

동기화된 조건에서 BatchNorm을 사용하는 표준 ResNet18 형태의 residual architecture에도 관련된 framework 간 최초 수치 진입점이 나타나는가?

## 변경 요인 격리 계획

- **변경 요인:** Phase 2의 custom sequential CNN에서 정렬된 ResNet18 구현으로 아키텍처만 변경한다.
- **가능한 한 고정:** dataset/workload, precision, input/batch identity, canonical initialization, Common Adam, Common BN semantics, synchronized one-step 방법론, trace metric 및 V2 전에 확정할 실행 조건을 유지한다.
- 진단 전에 stem, shortcut/downsample 규칙, padding, BN 위치, activation 순서, pooling, classifier, parameter layout 및 loss를 framework 간 정렬한다.

## 수집할 근거

Preflight에서 canonical state/input exact synchronization, architecture/parameter parity, instrumentation equivalence, framework 내부 반복 재현성 및 residual block 단위 semantic trace를 확인한다. BN1이 최초 stage일 것이라고 가정하지 않고 V1 및 Phase 2 결과와 비교한다.

현재 단계에서는 코드, checkpoint, 학습 및 diagnostic을 생성하거나 실행하지 않았다.
