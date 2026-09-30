# Final Results Summary

## 연구 제목

**엄격한 통제 조건에서 TensorFlow/Keras와 PyTorch 간 CNN 학습 수치 분기의 발생 위치와 누적 양상에 대한 실증 분석**

Working title: *Tracing Cross-Framework CNN Training Divergence between TensorFlow/Keras and PyTorch under Strictly Controlled Conditions*

## 한 문장 결론

Exact W0·입력·순서·loss semantics를 맞춰도 tested TensorFlow/Keras Metal과 PyTorch MPS stack의 bitwise identity는 유지되지 않았고, 작은 초기 차이는 state-dependent feedback과 일치하는 방식으로 누적되어 큰 parameter-space separation으로 이어졌지만, 일관된 framework 성능 우위로 연결되지는 않았다.

## First observed entry matrix

| Study | Architecture | BN | Workload | Execution | First observed entry |
|---|---|---:|---|---|---|
| Phase 2 | Custom CNN | Yes | Young AffectNet HQ | GPU Metal/MPS | BN1 batch mean |
| V1 | Custom CNN | Yes | Young AffectNet HQ | CPU | Conv1 |
| V2 | ResNet18 | Yes | Young AffectNet HQ | GPU Metal/MPS | stem BN batch mean |
| V3 | Custom CNN | No | Young AffectNet HQ | GPU Metal/MPS | GAP |
| V4 | Custom CNN | Yes | CIFAR-10 | GPU Metal/MPS | BN1 batch mean |

First observed entry는 bitwise identity가 처음 깨진 관찰 위치다. Largest divergence나 low-level root cause와 동일하지 않다.

## Strict-control interventions

| Evidence | Main result | Bounded interpretation |
|---|---|---|
| Exp04 shared W0/input | Conv1 exact, BN1에서 약 FP-scale first difference; E30 global rel L2 `1.1155` | 작은 차이가 장기 trajectory separation과 공존 |
| Exp05 CommonAdam | First-update divergence 평균 `64.65%` 감소; E30 `1.0952` | Native Adam은 early amplification contributor, sole explanation 아님 |
| Exp06 CommonBN | Initial BN running-variance difference 약 `99.42%` 감소; E30 `1.0976` | Native BN semantics도 sole explanation 아님 |
| Exp07 full-state re-sync | E30 synchronized one-step post-weight divergence 평균 K-anchor `3.78e-7`, P-anchor `8.67e-9` | Large free-running distance는 accumulated state-dependent feedback과 더 잘 일치 |
| Exp08 layer trace | 9/9 input·Conv1 exact, first non-zero BN1 batch mean | Tested GPU+BN stack의 reproducible first observed entry |

## Long-term matrix

| Study | Workload | Update budget | Global parameter separation | Generalization |
|---|---|---:|---:|---|
| Phase 2 | Young AffectNet HQ | 9,630 | 약 `1.10` | Seed/checkpoint dependent |
| V4 step-matched | CIFAR-10 | 9,630 | `0.9416` | 이 step에서 final test 미평가 |
| V4 E30 | CIFAR-10 | 42,210 | `1.1565` | Final mean K/P performance nearly equal |

두 workload는 data, resolution, class 수, spatial geometry와 training policy가 달라 raw accuracy를 직접 비교하지 않는다.

## V4 핵심 수치

### Epoch-matched global relative L2, 3-Seed mean

| Epoch | Updates | Mean global relative L2 |
|---:|---:|---:|
| 1 | 1,407 | 0.4207 |
| 5 | 7,035 | 0.8575 |
| 10 | 14,070 | 1.0253 |
| 20 | 28,140 | 1.1207 |
| 30 | 42,210 | 1.1565 |

### Same-update-budget comparison, 3-Seed mean

| Global step | V4 CIFAR-10 | Phase 2 Young AffectNet HQ |
|---:|---:|---:|
| 321 | 0.1677 | 0.1823 |
| 1,605 | 0.4521 | 0.5522 |
| 3,210 | 0.6345 | 0.7759 |
| 6,420 | 0.8337 | 0.9915 |
| 9,630 | 0.9416 | 1.0976 |

Exact magnitude는 workload-dependent였지만 독립 workload에서도 substantial separation이 관찰됐다.

### V4 E30 layer/state distance, 3-Seed mean

| Group | Relative L2 |
|---|---:|
| Conv1 / Conv2 / Conv3 / Conv4 | 0.2880 / 0.7847 / 1.0903 / 1.2481 |
| FC128 / classifier | 0.6037 / 0.3894 |
| BN1–4 running mean | 0.3550 / 0.3282 / 0.4215 / 0.8358 |

Deeper convolution groups에서 더 큰 separation이 반복됐지만 Conv4나 network depth를 원인으로 확정하지 않는다.

### V4 performance

| Checkpoint | Keras Accuracy | PyTorch Accuracy | Signed P−K | Mean absolute paired gap |
|---|---:|---:|---:|---:|
| Final E30 | 0.76160 ± 0.02236 | 0.76243 ± 0.01147 | +0.00083 | 0.01550 |
| Best validation loss | 0.73030 ± 0.00930 | 0.75867 ± 0.02281 | +0.02837 | 0.03017 |

Final E30 Macro F1 평균은 Keras `0.76251`, PyTorch `0.76236`이었다. Final E30 Seed별 Accuracy 방향은 K/P/P로 교차했다. Best-validation epoch도 Seed 42 `5/5`, Seed 123 `4/6`, Seed 2026 `3/9`로 달랐다. `n=3`이므로 framework superiority를 주장하지 않는다.

E30 train accuracy는 Keras `98.296%`, PyTorch `98.359%`로 가까웠지만 global relative L2는 `1.1565`였다. 30 epochs 전체 mean absolute gap은 train accuracy `0.136%p`, validation accuracy `2.687%p`였다. 따라서 large parameter-space separation은 similar training fit과 공존할 수 있고 validation/generalization timing은 더 민감할 수 있다.

## 최종 해석

1. Exact initialization과 semantic alignment는 cross-framework bitwise identity를 보장하지 않았다.
2. GPU+BN의 세 architecture/workload 조합에서 first BN batch-mean entry가 반복됐지만, CPU에서는 Conv1, BN-free GPU에서는 GAP로 이동했다.
3. CommonAdam/CommonBN은 초기 차이 일부를 줄였으나 장기 separation을 제거하지 못했다.
4. Full-state re-sync 후 새 one-step divergence가 작아진 결과는 continuously generated large error보다 accumulated state-dependent feedback 설명과 더 잘 일치한다.
5. Parameter reproducibility와 training-fit/function-level similarity는 같은 개념이 아니다.
6. 결과는 framework 우열, BN/GAP/reduction/backend의 unique root cause 또는 보편적 일반화를 증명하지 않는다.

## 논문 기여와 투고 방향

기여는 strict controlled pipeline, CommonAdam/CommonBN intervention, full-state re-synchronization, layer-wise tracing, external-validity matrix, independent-workload full training, manifest/hash 기반 reproducibility package다. 1차 투고 목표는 **Journal of KIISE (JOK, 정보과학회논문지)**이며 Software Engineering for AI와 cross-framework numerical reproducibility 관점으로 정리한다.

상세 근거는 [Final Research Report](FINAL_RESEARCH_REPORT.md), 전체 상태는 [Experiment Status](EXPERIMENT_STATUS.md)를 참조한다.
