# V3 - BatchNorm 없는 CNN 검증

## 상태

**Next**

## 연구질문

정렬된 small CNN에서 BatchNorm을 제거하면 framework 간 최초 수치 차이는 어디에서 나타나며 이후 연산으로 어떻게 전파되는가?

## 변경 요인 격리 계획

- **변경 요인:** 정규화 아키텍처, 구체적으로 BatchNorm 존재 여부만 변경한다.
- **가능한 한 고정:** small CNN의 깊이/channel 구성, dataset/workload, precision, input/batch identity, canonical initialization, Common Adam, 실행 조건, synchronization 및 trace metric을 유지한다.
- 정규화 제거가 여러 아키텍처 변경과 섞이지 않도록 구현 전에 BN-free 설계와 parameter capacity 정렬 규칙을 확정한다.

## 수집할 근거

Trace에는 convolution, activation, pooling, GAP, dense/logits, loss, gradient, CommonAdam state/update 및 post-step weight를 포함한다. 최초 진입점은 Conv1, 후속 reduction, logits 또는 전체 exact 중 어느 결과도 가능하며 미리 가정하지 않는다.

현재 단계에서는 코드, 학습 및 diagnostic을 생성하거나 실행하지 않았다.
