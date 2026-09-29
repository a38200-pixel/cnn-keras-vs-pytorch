# V4 - CIFAR-10 Workload 검증

## 상태

**Planned**

## 연구질문

Young AffectNet HQ 대신 CIFAR-10을 사용해도 동기화된 framework 간 수치 분기 패턴이 유지되는가?

## 변경 요인 격리 계획

- **변경 요인:** dataset/workload만 변경한다.
- **가능한 한 고정:** V1–V3에서 검증된 하나의 아키텍처와 정규화 설계, precision, optimizer, canonical initialization, synchronization, 실행 조건, batch 구성 규칙 및 trace metric을 유지한다.
- 진단 전에 CIFAR-10 preprocessing, split, class mapping, batch identity 및 augmentation을 고정하고 hash로 기록한다.

## 수집할 근거

먼저 synchronized one-step trace를 수행한다. Multi-step 또는 Full Training은 diagnostic validity를 확인한 뒤 별도로 결정해야 하며 현재 범위에는 포함하지 않는다.

현재 단계에서는 dataset 다운로드, 코드, 학습 및 diagnostic을 생성하거나 실행하지 않았다.
