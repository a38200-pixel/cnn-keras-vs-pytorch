# Tracing Cross-Framework CNN Training Divergence

> **Project Status: Research experiments completed**
>
> **Experimental phase: Closed**
>
> No additional experiments are planned for this repository.

## 연구 제목

**엄격한 통제 조건에서 TensorFlow/Keras와 PyTorch 간 CNN 학습 수치 분기의 발생 위치와 누적 양상에 대한 실증 분석**

Working title: *Tracing Cross-Framework CNN Training Divergence between TensorFlow/Keras and PyTorch under Strictly Controlled Conditions*

연구 분야는 Software Engineering for AI, cross-framework deep-learning reproducibility, numerical reproducibility of neural-network training과 empirical analysis of deep-learning software다. 이 프로젝트는 감정분류나 CIFAR-10의 성능 우위를 비교하는 연구가 아니다.

## 연구 문제

> 동일한 CNN을 TensorFlow/Keras와 PyTorch에서 가능한 한 동일한 상태와 학습 조건으로 실행했을 때, 최초의 수치 차이는 어디에서 발생하고, 그 작은 차이가 학습 과정에서 어떻게 전파·누적되어 장기적인 training trajectory separation으로 이어지는가?

Architecture 이름과 hyperparameter가 같아도 initialization, tensor/order, augmentation, loss reduction, optimizer, BatchNorm state와 backend execution은 다를 수 있다. 이 연구는 변수를 단계적으로 통제하고 `W0 → forward → loss → backward → update → state → long-term trajectory`를 추적했다.

## 연구 구조와 최종 상태

| Block | Experiments | Role | Status |
|---|---|---|---|
| Phase 1 | 00–03 | Preliminary Native Framework Comparison | Completed; pilot evidence only |
| Phase 2 | 04–08 | Strict Controlled Experiments | **Completed / VALID** |
| External Validation | V1–V4 | Device/architecture/BN/workload validation | **Completed** |

Phase 1은 exact shared W0가 아니므로 causal framework comparison으로 해석하지 않는다. Input, batch order와 augmentation alignment가 native performance gap에 영향을 주는 것을 확인하고 strict control의 필요성을 제시한 pilot이다.

Phase 2는 canonical W0, exact input/order/augmentation tensor, raw logits, mean sparse cross-entropy와 manual training loop를 사용했다. CommonAdam, CommonBN, full-state re-synchronization과 layer trace를 통해 초기 차이의 위치와 장기 누적을 분석했다.

External Validation은 V1 CPU, V2 ResNet18+BN, V3 BN-free custom CNN, V4 CIFAR-10 independent workload로 관찰 범위를 확장했다. 상세 상태는 [Experiment Status](docs/EXPERIMENT_STATUS.md)에 있다.

## Experimental Roadmap

| ID | Experiment | Main question | Final status |
|---|---|---|---|
| 00 | [Baseline](experiments/00_baseline_final_cnn_3seed/README.md) | Native Keras/PyTorch gap은 반복되는가? | Preliminary / Completed |
| 01 | [Input Tensor Alignment](experiments/01_input_tensor_alignment/README.md) | Input preprocessing 영향은 무엇인가? | Preliminary / Completed |
| 02 | [Batch Order Alignment](experiments/02_batch_order_alignment/README.md) | Batch order 영향은 무엇인가? | Preliminary / Completed; Attempt 2 VALID |
| 03 | [Augmentation Alignment](experiments/03_augmentation_alignment/README.md) | Augmentation schedule 영향은 무엇인가? | Preliminary / Completed / VALID |
| 04 | [Common Initialization & Controlled Training](experiments/04_common_initialization_controlled_training/README.md) | Exact W0/input에서 divergence는 어떻게 시작·누적되는가? | Completed / VALID |
| 05 | [Common Adam Optimizer Control](experiments/05_gradient_optimizer_divergence/README.md) | Native Adam 차이를 제거하면 얼마나 감소하는가? | Completed / VALID |
| 06 | [Common BatchNorm Control](experiments/06_batchnorm_state_divergence/README.md) | Native BN semantics를 제거하면 얼마나 감소하는가? | Completed / VALID |
| 07 | [Multi-Step & State Re-Synchronization](experiments/07_multistep_state_resynchronization/README.md) | Accumulated state와 새 one-step difference를 분리할 수 있는가? | Completed / VALID |
| 08 | [Layer-by-Layer Trace](experiments/08_layer_by_layer_trajectory/README.md) | First entry와 propagation 위치는 어디인가? | Completed / VALID |
| V1 | [CPU-only](experiments/external_validation/V1_cpu_only/README.md) | Entry point는 execution stack에 의존하는가? | Completed / VALID |
| V2 | [ResNet18 + BN](experiments/external_validation/V2_resnet18_bn/README.md) | BN entry가 다른 architecture에서도 유지되는가? | Completed / VALID |
| V3 | [BN-free CNN](experiments/external_validation/V3_bn_free_cnn/README.md) | BN이 없으면 first entry는 어디로 이동하는가? | Completed / VALID |
| V4 | [CIFAR-10 Independent Workload](experiments/external_validation/V4_cifar10/README.md) | Initial entry와 long-term separation이 독립 workload에서도 관찰되는가? | Completed / VALID |

## 핵심 결과

### First observed entry matrix

| Study | Architecture | BN | Workload | Execution | First observed entry |
|---|---|---:|---|---|---|
| Phase 2 | Custom CNN | Yes | Young AffectNet HQ | GPU Metal/MPS | BN1 batch mean |
| V1 | Custom CNN | Yes | Young AffectNet HQ | CPU | Conv1 |
| V2 | ResNet18 | Yes | Young AffectNet HQ | GPU Metal/MPS | stem BN batch mean |
| V3 | Custom CNN | No | Young AffectNet HQ | GPU Metal/MPS | GAP |
| V4 | Custom CNN | Yes | CIFAR-10 | GPU Metal/MPS | BN1 batch mean |

Tested GPU+BN의 세 architecture/workload 조합에서 first BatchNorm batch-mean entry가 반복됐다. 그러나 CPU와 BN-free GPU에서는 entry가 이동했다. 따라서 BN batch mean은 테스트한 GPU+BN 조건의 reproducible **first observed entry point**지만 universal root cause는 아니다. First entry는 largest divergence나 low-level mechanism과도 동일하지 않다.

### Phase 2 strict-control findings

- **Exp04:** Input과 Conv1 exact, BN1에서 약 `1e−7` first difference. E30 global weight relative L2 평균 `1.1155`.
- **Exp05:** CommonAdam으로 first-update divergence 평균 `64.65%` 감소. E30 `1.0952`로 장기 separation은 유지.
- **Exp06:** CommonBN으로 initial running-variance difference 약 `99.42%` 감소. E30 `1.0976`.
- **Exp07:** E30 full-state sync 뒤 new one-step post-weight divergence는 평균 K-anchor `3.78e−7`, P-anchor `8.67e−9`.
- **Exp08:** 9/9 synchronized cases에서 input/Conv1 exact, first non-zero는 BN1 batch mean. 후속 maximum은 state/anchor에 따라 달라짐.

Native Adam은 early amplification contributor였고 native BN semantics는 initial state discrepancy에 기여했지만 어느 하나도 long-term separation의 sole explanation은 아니었다. Full-state re-sync 결과는 large free-running separation이 매 step 같은 크기로 새로 생성되기보다, 작은 차이가 weights·optimizer·BN state를 바꾸고 이후 계산에 재입력되는 **accumulated state-dependent feedback**과 더 잘 일치했다. 이는 causal proof가 아니다.

### Long-term comparison

| Study | Workload | Update budget | Global parameter separation | Generalization |
|---|---|---:|---:|---|
| Phase 2 | Young AffectNet HQ | 9,630 | 약 `1.10` | Seed/checkpoint dependent |
| V4 step-matched | CIFAR-10 | 9,630 | `0.9416` | 이 step에서 final test 미평가 |
| V4 E30 | CIFAR-10 | 42,210 | `1.1565` | Final mean K/P performance nearly equal |

V4 same-update-budget distance는 Step 321/1,605/3,210/6,420/9,630에서 각각 `0.1677/0.4521/0.6345/0.8337/0.9416`이었다. 대응 Phase 2 값은 `0.1823/0.5522/0.7759/0.9915/1.0976`이었다. 독립 workload에서도 substantial separation이 나타났지만 exact magnitude는 workload-dependent였다.

V4 E30 convolution distance는 3-Seed 평균 Conv1 `0.2880` < Conv2 `0.7847` < Conv3 `1.0903` < Conv4 `1.2481`이었다. 이 pattern은 deeper convolution group의 더 큰 relative separation을 기술할 뿐 Conv4나 depth를 원인으로 확정하지 않는다.

### Parameter separation과 performance

V4 E30 train accuracy는 Keras `98.296%`, PyTorch `98.359%`였지만 global relative L2는 `1.1565`였다. 30 epochs 전체의 mean absolute train accuracy gap은 `0.136%p`, validation accuracy gap은 `2.687%p`였다. Large parameter-space separation은 similar training fit과 공존했고 validation trajectory는 더 민감했다.

| V4 Final E30 test | Keras | PyTorch | Signed P−K | Mean absolute paired gap |
|---|---:|---:|---:|---:|
| Accuracy | .76160 ± .02236 | .76243 ± .01147 | +.00083 | .01550 |
| Macro F1 | .76251 ± .02020 | .76236 ± .01008 | −.00015 | .01523 |

Seed별 방향은 교차했다. Best-validation loss checkpoint도 Seed별 K/P epoch가 `5/5`, `4/6`, `3/9`로 달랐다. 따라서 parameter reproducibility, training-fit similarity와 generalization optimum timing은 구분해야 하며 consistent framework superiority는 관찰되지 않았다.

## Final Conclusion

Exact shared initialization과 semantic alignment는 cross-framework bitwise numerical identity를 보장하지 않았다. FP-scale의 first difference는 forward/backward/update를 통해 state에 반영됐고 장기 parameter trajectories는 크게 분리됐다. Common optimizer와 BN semantics도 이를 완전히 제거하지 못했고, full-state re-sync 뒤 새 one-step discrepancy가 다시 작아졌다는 사실은 accumulated state-dependent feedback 설명과 가장 잘 맞았다.

Entry point는 execution context와 operation composition에 의존했고 magnitude와 generalization path는 workload/Seed에 따라 달랐다. 큰 parameter-space distance가 큰 performance gap을 뜻하지 않았고 final mean performance도 거의 같았다. 이 연구는 특정 framework, BatchNorm, GAP, reduction 또는 backend의 우열·root cause를 증명하지 않는다.

## Claim Boundary

말할 수 있는 범위:

- first observed numerical entry와 controlled intervention 결과
- cross-framework parameter/trajectory separation
- state-dependent accumulation과 consistent한 evidence
- execution-stack dependent entry와 workload-dependent magnitude
- seed/checkpoint-dependent generalization 및 no consistent framework superiority

말할 수 없는 범위:

- BN, GAP, reduction, Metal/MPS 또는 특정 layer가 unique root cause라는 주장
- TensorFlow/Keras 또는 PyTorch가 본질적으로 더 정확·안정적이라는 주장
- State-feedback causal mechanism의 완전한 증명
- 모든 hardware/network/framework에 대한 일반화
- CIFAR-10 dataset 자체가 divergence를 만들었다는 주장

## Threats to Validity

- 주요 Seeds 3개와 Apple M1 Pro 단일 hardware 중심
- TensorFlow Metal과 PyTorch MPS 비교이며 CUDA/NVIDIA 미검증
- Low-level reduction tree/kernel/compiler lowering 미계측
- First non-zero는 bitwise break이며 practical significance와 동일하지 않음
- V4는 pure dataset-only intervention이 아님
- Young AffectNet HQ는 age-transformed derivative workload이며 raw image redistribution을 피해야 함
- Phase 1은 exact shared W0가 아니며 causal comparison이 아님
- Framework superiority 평가가 연구 목적이 아님

## Expected Contributions and Publication Direction

기여는 strict controlled pipeline, CommonAdam/CommonBN intervention, full-state re-synchronization, layer-wise propagation trace, CPU/ResNet18/BN-free/CIFAR-10 external-validity matrix, independent-workload full training, manifest/hash 기반 reproducibility package다.

현재 1차 논문 투고 목표는 **Journal of KIISE (JOK, 정보과학회논문지)**다. Software Engineering for AI와 empirical deep-learning software analysis 관점으로 구성하며 국제 venue는 future possibility로만 둔다.

## Documentation and Artifact Navigation

- [Final Research Report](docs/FINAL_RESEARCH_REPORT.md): 논문 작성 전 source-of-truth 해석 문서
- [Final Results Summary](docs/FINAL_RESULTS_SUMMARY.md): 핵심 수치 중심 요약
- [Experiment Status](docs/EXPERIMENT_STATUS.md): 전체 실험 상태표
- [Phase 2 Final Conclusion](PHASE2_STRICT_CONTROLLED_CONCLUSION.md): Experiments 04–08 상세 결론
- [External Validation Overview](experiments/external_validation/README.md): V1–V4 종합
- [V4 Final Report](experiments/external_validation/V4_cifar10/README.md): CIFAR-10 Stage A/B 결과와 artifact index

각 experiment의 `results/` 아래 CSV/JSON/manifest/history가 numerical source-of-truth다. Raw dataset과 대형 runtime/resume/checkpoint binary는 Git 공개 대상이 아니다. Dataset을 재배포하지 않는다.

## Repository Closure

연구 실험 단계는 종료됐다. CUDA/NVIDIA replication, reduction-operator microbenchmark, 추가 architecture/framework와 larger-seed study는 이 repository scope 밖의 Future Work이며 현재 프로젝트의 미완료 TODO가 아니다. 기존 결과를 보존하기 위해 training entry point의 `RUN_TRAINING`/`RUN_FULL_TRAINING` 기본값은 비활성 상태로 유지한다.
