# 외부 타당성 검증 연구

> **Study Status: Completed.** V1–V4의 승인된 진단과 V4 Stage B full training이 모두 종료됐다.

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
- **V4 — CIFAR-10 Independent Workload Validation:** 독립 workload에서 initial divergence와 장기 trajectory separation이 함께 관찰되는가?

## 검증 구성

| 검증 | 변경 요인 | 고정 요인 | 핵심 질문 | 상태 |
|---|---|---|---|---|
| [V1 CPU-only 실행](V1_cpu_only/README.md) | 실행 장치 | Young AffectNet HQ, custom CNN+BN, Common Adam, Common BN, 동일 state/batch/case | Metal/MPS 없이도 수치 분기가 유지되는가? | **Completed / VALID** |
| [V2 ResNet18 + BN](V2_resnet18_bn/README.md) | 아키텍처 | Young AffectNet HQ, GPU stack, Common Adam/BN, fixed batch | Residual architecture에서도 패턴이 유지되는가? | **Completed / VALID** |
| [V3 BN-free CNN](V3_bn_free_cnn/README.md) | BatchNorm presence | Phase 2 custom CNN geometry, Young AffectNet HQ, GPU stack, Common Adam, fixed batch/W0 | BN만 제거하면 최초 진입점은 어디로 이동하는가? | **Completed / VALID** |
| [V4 CIFAR-10](V4_cifar10/README.md) | Independent workload | CommonBN/CommonAdam, shared body W0, dtype, Seeds, paired order | Initial entry와 full-training trajectory separation이 유지되는가? | **Completed / VALID** |

V1–V3는 주요 변경 요인을 가능한 한 격리했다. V4는 dataset-only intervention이 아니며 image domain, input resolution, dataset size, class count, classifier, spatial reduction geometry, update 수와 augmentation policy가 함께 달라지는 **Independent Workload Validation**이다.

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
- V1–V3의 검증 범위는 synchronized one-step이며 장기 학습이나 일반화 성능이 아니다. V4만 별도의 Stage B full training을 완료했다.
- 결과는 float32, 3개 초기 Seed 및 사용 가능한 checkpoint로 제한된다.
- V4의 추가 정렬 조건과 불가피한 workload 차이는 [V4 문서](V4_cifar10/README.md)에 고정했다.

## 진행 순서

1. V1 CPU-only synchronized one-step을 수행하고 Experiment 08과 case별로 직접 비교한다.
2. 검증된 절차를 V2 ResNet18 + BN에 적용한다.
3. V3에서 BatchNorm 존재 여부를 격리한다.
4. V4 CIFAR-10 Stage A와 6-run Stage B를 완료하고 epoch/step-matched trajectory를 분석했다.

V2에서는 기존 GPU 실행환경을 다시 고정하고 custom CNN을 직접 구현한 semantic-equivalent ResNet18로만 변경했다. Initial shared state 3개 synchronized one-step만 수행했으며 Full Training은 실행하지 않았다.

## V1 결과

V1은 CPU 배치, 기존 batch/state의 exact synchronization, framework 내부 반복 재현성, trace 동등성, case 완전성 및 유한값 검사를 모두 통과했다. Experiment 08 GPU의 9개 case 모두에서 BN1 batch mean이 최초 차이였던 것과 달리 CPU에서는 9개 모두 Conv1에서 최초 차이가 나타났다. CPU의 paired maximum-forward relative L2는 GPU 값보다 `49.9×–148.6×` 컸다.

따라서 GPU의 최초 진입점은 실행 스택에 민감한 관찰로 적용 범위를 좁혀야 한다. 반면 exact synchronized boundary state에서도 작은 framework 간 수치 차이가 발생하고 한 step 내부로 전파될 수 있다는 더 넓은 관찰은 유지된다.

## V2 결과

V2는 동일 Metal/MPS GPU stack에서 custom CNN만 직접 구현한 semantic-equivalent ResNet18로 바꿨다. Framework별 trainable parameter는 각각 `11,180,616`개였고, 3개 Seed의 canonical state mapping 306/306개가 exact였다. 3/3 Seed 모두 stem Conv까지 exact였고 최초 차이는 `stem.bn.batch_mean`에서 나타나 custom CNN의 대응 진입점이 유지됐다. Initial one-step maximum-forward relative L2는 `4.09e-6–4.23e-6`이며 paired custom CNN보다 `7.03×–7.34×` 컸다. Residual Add는 최초 진입점이 아니었고, 최대 backward/gradient/update 위치는 architecture와 Seed에 따라 달라졌다. 따라서 최초 진입점보다 downstream propagation magnitude와 sensitivity pattern에서 architecture dependence가 더 뚜렷했다.

## V3 결과

V3는 Phase 2 custom CNN의 Conv/Pool/GAP/Dense geometry, Conv `bias=False`, persisted Conv/Dense W0, fixed batch, Metal/MPS GPU와 CommonAdam을 유지하고 BN operation·gamma/beta·running state만 제거했다. Framework별 trainable parameter는 `421,864`개였고 3개 Seed의 24/24 parameter mapping 및 Phase 2 shared W0가 exact였다.

3/3 Seed 모두 `input → Conv1–4 → ReLU1–4 → Pool1–4`의 13개 stage가 exact였고 first non-zero는 `gap`으로 이동했다. Maximum-forward relative L2는 `1.22e-7–2.21e-7`로 paired BN model의 `0.210×–0.382×`였다. First backward는 3/3 `logits`였으며 gradient/update 차이는 남았다. 따라서 tested GPU custom CNN의 first BN entry는 BN presence에 의존하지만, BN 제거가 cross-framework divergence 자체를 없애지는 않았다.

## V4 결과

V4는 `32×32` CIFAR-10 custom CNN+CommonBN을 직접 구현하고 Phase 2 body W0와 새 10-class canonical classifier를 Keras/PyTorch에 exact mapping했다. Official train은 고정 stratified split으로 45,000/5,000, test는 official 10,000을 유지했다. Batch 32에서 1 epoch은 1,407 updates(마지막 batch 8), 30 epochs는 42,210 updates였다.

Stage A 3/3 Seeds에서 input과 Conv1은 exact였고 first non-zero는 `bn1.batch_mean`, largest forward는 `bn4.x_hat`이었다. Stage B 6/6 runs와 모든 validity gate가 완료됐다. Global weight relative L2 평균은 E1 `0.4207`에서 E30 `1.1565`로 증가했고, 같은 9,630-update budget에서는 V4 `0.9416`, Phase 2 `1.0976`이었다. V4 E30 train accuracy는 Keras/PyTorch `98.296/98.359%`, final test accuracy는 `.76160/.76243`으로 가까웠다. Exact magnitude와 validation timing은 workload/Seed dependent였으며 consistent framework superiority는 관찰되지 않았다.

## V1–V3 중간 종합 해석

| 조건 | 모델 | 최초 비영 forward stage |
|---|---|---|
| Phase 2 GPU | Custom CNN + BN | BN1 batch mean (9/9 case) |
| V1 CPU | Custom CNN + BN | Conv1 (9/9 case) |
| V2 GPU | ResNet18 + BN | Stem BN batch mean (3/3 Seed) |
| V3 GPU | Custom CNN, BN 없음 | GAP (3/3 Seed) |
| V4 GPU | Custom CNN + BN | BN1 batch mean (3/3 Seed) |

Exact first entry point는 execution stack과 BN presence에 민감했다. 테스트한 GPU stack의 BN batch-mean entry는 두 BN architecture에서 반복됐고, BN을 제거하자 GAP로 지연됐다. 이후 전파 크기와 최대 민감 위치도 architecture와 Seed에 따라 달랐다. 이는 관찰 범위를 구체화하는 결과이며 특정 backend, BatchNorm, GAP 또는 architecture를 root cause로 확정하지 않는다.

## Future Work: Reduction Operator Isolation

Phase 2/V2의 GPU+BN 조건에서는 BN batch-mean reduction이, V3 BN-free GPU 조건에서는 Conv/ReLU/Pool이 모두 exact인 뒤 GAP가 first observed numerical difference였다. 이 공통점은 tested GPU stack에서 floating-point mean/summation reduction operation이 first numerical entry point와 관련될 **가능성**을 후속 가설로 남긴다.

후속 operator-level microbenchmark에서는 동일한 exact tensor를 TensorFlow/Keras와 PyTorch에 입력하고 다음을 분리 비교할 수 있다.

1. Spatial-axis `reduce_mean`
2. `reduce_sum`
3. BatchNorm batch mean
4. Global Average Pooling

Shape, reduction axis, tensor size, dtype을 통제하고 exact equality, max absolute difference 및 relative L2를 기록한다. 필요하면 CPU와 GPU execution stack을 나누어 비교한다. 다만 현재 실험은 reduction order, kernel implementation, graph fusion 또는 compiler lowering을 직접 계측하지 않았으므로 reduction이 root cause이거나 BN과 GAP가 동일한 내부 원인을 공유한다고 결론내리지 않는다.

External Validation V1–V4는 완료됐다. Reduction microbenchmark, CUDA/NVIDIA replication과 추가 architecture/Seed는 현재 repository scope 밖의 Future Work이며 미완료 실험이 아니다.
