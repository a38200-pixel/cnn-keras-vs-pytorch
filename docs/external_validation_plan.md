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
| V3 | BatchNorm 없는 CNN | BatchNorm presence | Phase 2 geometry/W0와 GPU stack을 유지한 3-Seed initial synchronized one-step | **Completed / VALID** |
| V4 | CIFAR-10 Independent Workload Validation | Workload 전환 | 3-Seed initial one-step + 3-Seed×2-framework full training | **Completed / VALID** |

V1–V3는 가능한 한 하나의 주요 요인을 격리한다. V4는 dataset-only intervention이 아니며 image domain, resolution, sample/class 수, classifier shape, spatial reduction geometry, augmentation policy와 update 수가 함께 달라지는 independent workload validation이다.

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

V4의 preflight와 training 명세는 구현에 고정됐고 6개 Full Training run과 후속 분석이 완료됐다. 이 계획 문서는 완료된 설계 기록으로 보존한다.

## V1 결과

9개의 CPU-only case는 모든 validity gate를 통과했다. CPU의 최초 비영 forward stage는 9/9 case에서 Conv1이었고, Experiment 08 GPU에서는 9/9 case에서 BN1 batch mean이었다. CPU maximum-forward relative L2는 paired GPU 값보다 `49.9×–148.6×` 컸다.

따라서 V1은 Phase 2의 최초 진입점 관찰 범위를 테스트한 execution stack으로 좁힌다. 동시에 통제된 framework 간 수치 차이의 전파를 외부 조건에서도 검증해야 한다는 더 넓은 연구 동기는 유지된다. 실제 layer/group 결과는 [V1 보고서](../experiments/external_validation/V1_cpu_only/README.md)에 정리했다.

## V2 결과

동일 Metal/MPS GPU 실행환경과 fixed batch/Common Adam/Common BN을 유지하고 architecture만 직접 구현한 semantic-equivalent ResNet18로 변경했다. Framework별 trainable parameter는 `11,180,616`개였고 3개 Seed의 canonical state mapping 306/306개가 exact였다. 3개 Seed 모두 stem Conv는 exact였고 최초 비영 stage는 `stem.bn.batch_mean`이었다. Initial one-step maximum-forward relative L2는 `4.09e-6–4.23e-6`이었고, Custom CNN의 대응 maximum보다 Seed별 `7.03×–7.34×` 컸다. Residual Add는 최초 진입점이 아니었으며 downstream maximum의 위치와 크기는 architecture 및 Seed에 민감했다. 상세 결과와 metric 정의는 [V2 보고서](../experiments/external_validation/V2_resnet18_bn/README.md)에 정리했다.

## V3 결과

Phase 2 custom CNN geometry와 Conv `bias=False`, Metal/MPS GPU, fixed batch, CommonAdam 및 Seed별 persisted Conv/Dense W0를 유지하고 BN operation·affine parameter·running state를 제거했다. Keras/PyTorch trainable parameter는 각각 `421,864`개였고 24/24 Seed-parameter mapping과 Phase 2 shared W0가 exact였다.

3개 Seed 모두 Conv1–4/ReLU1–4/Pool1–4까지 exact였고 first non-zero는 `gap`이었다. Exact prefix는 input 포함 13개 stage이며 maximum-forward relative L2는 `1.22e-7–2.21e-7`, paired BN model 대비 `0.210×–0.382×`였다. Backward first entry는 3/3 `logits`였고 gradient/CommonAdam update 차이는 남았다. 상세 결과는 [V3 보고서](../experiments/external_validation/V3_bn_free_cnn/README.md)에 정리했다.

## V1–V3 중간 결론과 다음 검증

V1 CPU custom CNN은 9/9 case에서 Conv1, Phase 2 GPU custom CNN은 9/9 case에서 BN1 batch mean, V2 GPU ResNet18은 3/3 Seed에서 stem BN batch mean, V3 GPU BN-free custom CNN은 3/3 Seed에서 GAP가 최초 비영 stage였다. 따라서 exact entry point는 execution stack과 BN presence에 민감하며, downstream propagation은 architecture와 Seed에도 민감하다. 어느 관찰도 특정 연산이나 backend의 보편적 인과성을 확정하지 않는다.

V3 결과는 tested GPU BN entry가 BN presence에 의존한다는 가설을 지지하지만, divergence 자체는 GAP 이후 남았으므로 BN을 유일 원인으로 보는 해석을 한정한다. 독립 workload V4도 완료됐으며 first entry와 장기 separation이 모두 관찰됐다.

## V4 설계와 실행 계획

V4의 Primary RQ는 independent CIFAR-10 workload로 옮겼을 때 initial numerical divergence pattern과 subsequent training trajectory separation이 테스트한 GPU stack에서 모두 관찰되는지다.

### Stage A

- Split 확정 뒤 sorted training indices의 첫 32개를 diagnostic 전용 fixed batch로 사용한다.
- Seed 42/123/2026의 shared W0에서 forward/loss/backward/CommonAdam 1 update를 trace한다.
- CommonBN 내부 reduction을 포함한 최초 비영 forward stage, backward, parameter gradient, update와 post-weight distance를 기록한다.
- Plain/traced equivalence 및 fresh-run repeatability gate를 통과해야 layer-level 결과를 해석한다.

### Stage B

- Official train 50,000을 split seed 42로 class별 4,500/500, 전체 45,000/5,000으로 고정하며 official test 10,000은 마지막 평가에만 쓴다.
- Seed 42/123/2026 × Keras/PyTorch 총 6개 run, 30 epochs, batch 32, constant lr 0.001이다.
- CommonBN/CommonAdam, float32, mean sparse cross-entropy, exact paired W0와 Common NumPy epoch order를 사용한다.
- Augmentation이나 dataset mean/std normalization은 사용하지 않고 `uint8 → float32 / 255`만 적용한다.
- 45,000 samples에서 1 epoch은 1,407 updates, 마지막 batch는 8개이며 30 epochs는 42,210 updates다.

### Initialization과 trajectory 비교

Phase 2와 shape가 같은 Conv1–4, BN affine/running state와 FC128 W0를 exact 재사용한다. 128→10 classifier만 Seed별 deterministic Glorot-uniform canonical W0를 생성해 양 framework에 exact load한다. 구현상 trainable parameter는 각각 423,082개, BN running state는 960 values다.

Epoch 0/1/5/10/20/30 snapshot은 같은 workload pass 수를 비교한다. Global step 0/321/1,605/3,210/6,420/9,630 snapshot은 Phase 2와 같은 optimizer-update budget을 기술적으로 비교한다. Step-matched 결과도 workload를 causal-equivalent하게 만들지는 않으며 classifier는 8-class와 10-class라 cross-workload direct paired comparison에서 제외한다.

Checkpoint는 update 완료 후 증가한 global step과 CommonAdam internal step이 exact하게 일치할 때만 저장한다. 분석 단계에서 metadata와 snapshot completeness를 다시 검증한다. Minimum validation loss로 best checkpoint를 정하고, Epoch 30 final과 best-validation checkpoint의 official test loss/accuracy/macro F1을 30 epochs 완료 뒤 평가한다.

최종 결과에서 Stage A 3/3 Seeds의 first non-zero는 `bn1.batch_mean`이었고 Stage B E30 global relative L2 평균은 `1.1565`였다. Same-update-budget Step 9,630은 V4 `0.9416`, Phase 2 `1.0976`이었다. 6/6 runs, checkpoint/test 및 NaN/Inf gate가 모두 통과해 V4는 `Completed / VALID`다.

## Future Work 가설: Reduction Operator Isolation

현재 GPU 관찰은 다음과 같다.

- Phase 2/V2 + BN: first difference가 BN batch-mean reduction에서 반복 관찰됨
- V3 BN-free: Conv/ReLU/Pool은 exact이고 first difference가 GAP에서 관찰됨

따라서 floating-point mean/summation reduction operation과 first numerical entry point의 관련 가능성을 후속 가설로 둔다. 검증한다면 동일한 exact tensor를 양 framework에 입력하고 spatial-axis `reduce_mean`, `reduce_sum`, BatchNorm batch mean 및 Global Average Pooling을 operator-level microbenchmark로 비교한다. Shape, axis, tensor size, dtype을 고정하고 exact equality, max absolute difference, relative L2를 기록하며 CPU/GPU execution stack을 분리할 수 있다.

이는 계획 후보일 뿐 현재 결과가 아니다. Reduction order, backend kernel implementation, graph fusion 및 compiler lowering은 아직 직접 계측하지 않았으며, reduction이 root cause이거나 BN과 GAP에 동일한 내부 원인이 있음을 증명하지 않았다.
