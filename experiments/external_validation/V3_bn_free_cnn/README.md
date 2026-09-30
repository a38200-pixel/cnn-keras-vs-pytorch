# V3 - BatchNorm-Free Custom CNN Validation

## Status

**Completed / VALID**

## Research Question

> Phase 2 custom CNN에서 나머지 구조와 실행 조건을 가능한 한 동일하게 유지하고 BatchNorm만 제거하면 최초 Keras–PyTorch 수치 차이는 어디에서 나타나는가?

## Motivation

Phase 2와 V2의 tested GPU 조건에서는 첫 비영 차이가 첫 BN batch-mean reduction에서 관찰됐다. V3는 같은 GPU stack과 custom CNN geometry에서 BN을 제거해 이 진입점이 어디로 이동하는지 격리한다. Conv1 또는 다른 stage를 사전에 가정하지 않는다.

## Changed and Fixed Conditions

- **변경:** BN1–BN4 operation, gamma/beta, running mean/variance 제거
- **고정:** Conv channel/kernel/stride/padding/bias, ReLU, Pool, GAP, Dense128, logits, dataset, fixed batch, augmentation tensor, labels, float32, Metal/MPS GPU, loss, Common Adam, Seeds 42/123/2026
- **Case:** Seed별 initial/shared 한 개, 총 3개 synchronized one-step
- **금지:** Epoch training, validation/test 평가, checkpoint 선택, V4 구현·실행

## BN-Free Architecture

```text
Input 128×128×3
→ [Conv3×3(bias=False) → ReLU → MaxPool2×2] × 4
  channels: 3→32→64→128→256
→ GAP → Dense 256→128(bias=True) → ReLU
→ Dense 128→8(bias=True) → raw logits
```

## Why Conv Bias Remains Disabled

Phase 2 Conv가 `bias=False`였으므로 V3도 유지한다. 일반적인 BN-free architecture 최적화를 위한 선택이 아니라 BN presence 외 변경을 최소화하기 위한 통제다.

## Shared Phase 2 Initialization

각 Seed의 Experiment 04 persisted canonical initial state에서 Conv1–4, FC128, logits의 8개 parameter array만 exact 재사용했다. 새 W0를 생성하지 않았고 BN state만 제외했다. V3 Keras state, V3 PyTorch state 및 대응 Phase 2 canonical W0는 3개 Seed 모두 element-wise exact였다.

## Parameter / Architecture Equivalence

| 검증 | 결과 |
|---|---:|
| Keras trainable parameters | `421,864` |
| PyTorch trainable parameters | `421,864` |
| Semantic parameter arrays | Seed당 8개, 총 24개 exact |
| Semantic forward stages | 17개 shape/order exact |
| BN layer/state | 양 framework 모두 0 |

Phase 2 BN model의 trainable `422,824`개보다 960개 적은 것은 BN gamma/beta 제거에 따른 의도된 차이다. Running mean/variance도 함께 제거돼 V3의 non-trainable BN state는 없다.

## Execution Environment

공식 환경은 `results/manifests/environment.json`에 기록했다.

| 항목 | 값 |
|---|---|
| OS / architecture | `macOS-26.6.2-arm64-arm-64bit` / `arm64` |
| Python / NumPy | `3.10.3` / `2.0.2` |
| TensorFlow / tensorflow-metal | `2.18.1` / `1.2.0` |
| TensorFlow actual Conv device | `/job:localhost/replica:0/task:0/device:GPU:0` |
| PyTorch | `2.14.0` |
| PyTorch actual Conv device | `mps:0` |
| MPS fallback | `PYTORCH_ENABLE_MPS_FALLBACK=0` |
| dtype | `float32` |
| Config SHA-256 | `bb4b9cbb063ad2a78bf8452b2c12abdbdf2f116bd1fb4e59268d782a1b07af20` |

공식 one-step은 Keras `training=True`, PyTorch `model.train()`에서 실행했다. BN이 없으므로 running-state update는 존재하지 않는다.

## Fixed Diagnostic Batch

Experiment 07/08, V1, V2와 같은 32개 NHWC float32 batch를 새로 생성하지 않고 읽었다. Sample ID, label, augmentation metadata, per-sample tensor 및 NCHW roundtrip을 검증했으며 full hash는 다음과 같이 exact였다.

```text
c3ef4b5c6cb842154ddb03fa6cdc175dd5ab4e69a1e02f2bbc5493c3e2a48e7c
```

## Validity Gates

```text
V3_ARCHITECTURE_EQUIVALENCE = VALID
V3_PARAMETER_EQUIVALENCE = VALID
V3_NO_BATCHNORM_CHECK = VALID
V3_GPU_DEVICE_CHECK = VALID
V3_FIXED_BATCH_CHECK = VALID
V3_SHARED_W0_CHECK = VALID
V3_PHASE2_SHARED_PARAMETER_W0 = VALID
V3_RESYNC_PRECHECK = VALID
V3_TRACE_EQUIVALENCE = VALID
V3_GPU_REPEATABILITY = VALID
V3_SELECTED_CASES_COMPLETE = TRUE
NAN_INF_FOUND = FALSE
NEW_FULL_TRAINING_EXECUTED = FALSE
V3 STATUS = Completed / VALID
```

Plain one-step과 traced one-step은 3 Seeds × 2 frameworks × 5 global arrays, 총 30개 비교에서 exact였다. 각 framework의 fresh-state traced run 1/2도 동일한 30개 비교에서 exact였다.

## Metric Definitions

Metric utility는 비교 array를 float64로 변환해 평탄화한 후 계산한다.

- `max_abs_diff = max |K-P|`
- `mean_abs_diff = mean |K-P|`
- `relative_l2 = ||K-P||₂ / max(||K||₂, ||P||₂, 1e-12)`
- `cosine_similarity = (K·P)/(||K||₂||P||₂)`; zero-norm exact arrays는 `1`, 그 밖의 zero-norm case는 `0`
- `sign_mismatch_fraction`: 절댓값 `≤1e-12`를 0으로 매핑한 뒤 부호가 다른 원소의 비율

First non-zero stage는 semantic order에서 `max_abs_diff > 0`을 처음 만족하는 행이다. Exact forward prefix length는 `input`부터 그 직전까지 exact인 stage 수이며 loss는 prefix에서 제외한다.

## Forward Trace

| Seed | First non-zero | Exact prefix | Largest forward stage | Largest relative L2 | Logits relative L2 | Loss abs diff |
|---:|---|---:|---|---:|---:|---:|
| 42 | `gap` | 13 | `logits` | `1.4833e-7` | `1.4833e-7` | `0` |
| 123 | `gap` | 13 | `fc128.preactivation` | `1.2208e-7` | `1.0613e-7` | `0` |
| 2026 | `gap` | 13 | `logits` | `2.2055e-7` | `2.2055e-7` | `0` |

세 Seed 모두 exact prefix는 다음과 같다.

```text
input → conv1 → relu1 → pool1 → conv2 → relu2 → pool2
→ conv3 → relu3 → pool3 → conv4 → relu4 → pool4
```

따라서 BN 제거 후 Conv1이 first non-zero가 되지 않았다. 모든 convolution/ReLU/pooling output이 exact였고 첫 차이는 channel-wise spatial reduction인 GAP에서 관찰됐다.

## Backward Trace

| Seed | First non-zero backward | Largest backward stage | Largest relative L2 |
|---:|---|---|---:|
| 42 | `logits` | `conv1` | `4.9169e-7` |
| 123 | `logits` | `conv1` | `4.8777e-7` |
| 2026 | `logits` | `pool1` | `4.9554e-7` |

Backward의 첫 비영 stage는 세 Seed 모두 `logits`였다. 최대 민감 위치는 Seed 42/123의 `conv1`, Seed 2026의 `pool1`로 완전히 고정되지 않았다.

## CommonAdam Update Trace

| Seed | Largest parameter-gradient group (rel. L2) | Largest update group (rel. L2) | Global gradient | Global update | Post-weight |
|---:|---|---|---:|---:|---:|
| 42 | `conv1/kernel` (`1.2973e-6`) | `conv2/kernel` (`1.9660e-5`) | `5.7572e-7` | `1.0164e-5` | `2.7100e-7` |
| 123 | `conv1/kernel` (`1.1444e-6`) | `conv3/kernel` (`9.4449e-6`) | `5.4993e-7` | `6.2322e-6` | `1.6753e-7` |
| 2026 | `conv1/kernel` (`1.3768e-6`) | `conv3/kernel` (`1.6726e-5`) | `6.1434e-7` | `9.3125e-6` | `2.4987e-7` |

동일 CommonAdam을 사용해도 non-zero gradient가 update 및 post-weight 차이로 이어졌다. 이 관찰은 특정 parameter group이나 optimizer를 causal root cause로 확정하지 않는다.

## BN vs BN-Free Comparison

| Seed | BN first / prefix | BN-free first / prefix | BN max forward | BN-free max forward | BN-free / BN |
|---:|---|---|---:|---:|---:|
| 42 | `bn1_batch_mean` / 3 | `gap` / 13 | `5.8152e-7` | `1.4833e-7` | `0.255×` |
| 123 | `bn1_batch_mean` / 3 | `gap` / 13 | `5.8085e-7` | `1.2208e-7` | `0.210×` |
| 2026 | `bn1_batch_mean` / 3 | `gap` / 13 | `5.7712e-7` | `2.2055e-7` | `0.382×` |

BN model에서는 Conv1과 BN1 input까지 3개 stage가 exact였고 BN1 batch mean이 첫 비영 stage였다. BN-free에서는 13개 stage가 exact였고 first entry가 GAP로 이동했다. Maximum-forward relative L2도 paired initial-step에서 BN 모델의 약 `21.0%–38.2%`였다. 이 비율은 initial diagnostic의 기술적 수치이며 장기 안정성이나 정확도 개선을 뜻하지 않는다.

## Answer to the V3 Research Question

BatchNorm을 제거하자 3개 Seed 모두 GPU-side first observed numerical difference가 첫 BN batch-mean reduction에서 GAP로 이동했다. Conv1–Conv4, 각 ReLU와 Pool은 모두 exact였다. 이는 tested GPU custom CNN에서 first BN entry가 BN presence에 의존한다는 가설을 **지지한다**.

그러나 cross-framework divergence 자체가 제거된 것은 아니다. GAP에서 시작한 작은 차이가 Dense/logits, backward, parameter gradient 및 CommonAdam update에 남았다. 따라서 V3 결과는 “BN이 framework divergence의 유일한 원인”이라는 해석을 지지하지 않으며, 더 넓은 가설을 **한정한다(qualify)**. 이번 조건에서는 BN이 있을 때 BN reduction이 더 이른 관찰 진입점을 제공했고, BN이 없을 때는 GAP reduction이 다음 관찰 진입점이 됐다.

## Interpretation Rules

- First observed entry point, largest forward divergence, backward sensitivity, optimizer-update sensitivity 및 root cause를 구분한다.
- `gap`은 관찰된 첫 비영 stage이지 고유한 causal root cause로 확정되지 않았다.
- BN-free의 더 작은 initial maximum-forward relative L2를 정확도·일반화·학습 안정성 향상으로 해석하지 않는다.
- Phase 2와 V3의 동일 Conv가 다른 graph context에서 exact였다는 사실만 보고하며 backend fusion/kernel 선택을 추측하지 않는다.
- Keras, PyTorch, Metal, MPS, BN 또는 GAP의 본질적 우열을 주장하지 않는다.

## Limitations

- 3개 Seed와 하나의 initial synchronized one-step만 검증했다.
- BN-free 장기 trajectory, validation/test generalization 및 checkpoint selection을 수행하지 않았다.
- 단일 dataset과 하나의 Apple GPU TensorFlow Metal/PyTorch MPS 환경, float32에 한정된다.
- BN 제거는 normalization operation뿐 아니라 gamma/beta와 running state도 함께 제거한다.
- Conv bias는 격리를 위해 의도적으로 비활성화했으므로 일반적인 BN-free 최적 architecture와 다를 수 있다.
- GAP reduction, 저수준 kernel/reduction order, graph fusion 및 compiler lowering을 직접 계측하지 않았다.
- Full architecture family나 다른 batch size/device/dtype으로 일반화하지 않는다.

## Source-of-Truth Artifacts

| 구분 | Artifact | 역할 |
|---|---|---|
| Preflight | `results/preflight/architecture_equivalence.json` | 17개 semantic stage shape/order gate |
| Preflight | `results/preflight/parameter_equivalence.json` | 421,864 parameter 및 canonical mapping gate |
| Preflight | `results/preflight/no_batchnorm_check.json` | graph/state의 BN 부재 gate |
| Preflight | `results/preflight/gpu_device_check.json` | 실제 Metal/MPS placement gate |
| Preflight | `results/preflight/fixed_batch_check.json` | persisted fixed-batch identity gate |
| Preflight | `results/preflight/shared_w0_check.json` | V3/Phase 2 shared Conv/Dense W0 gate |
| Preflight | `results/preflight/resync_precheck.json` | model/Adam/input/label sync gate |
| 계측 검증 | `results/preflight/trace_equivalence.json` | plain/traced 30개 exact 비교 |
| 반복성 | `results/preflight/repeatability.json` | fresh traced run 30개 exact 비교 |
| Forward | `results/forward/seed*_initial_forward.csv` | stage별 forward/loss metric |
| Backward | `results/backward/*activation_gradients.csv` | activation-gradient metric |
| Backward | `results/backward/*parameter_gradients.csv` | canonical parameter-gradient metric |
| Optimizer | `results/optimizer/*updates.csv` | gradient/m/v/update/post-weight metric |
| Summary | `results/summaries/case_summary.csv` | Seed별 headline 및 exact prefix |
| Summary | `results/summaries/bn_vs_bnfree_initial.csv` | Experiment 08 BN paired 비교 |
| Summary | `results/summaries/diagnostic_summary.json` | 최종 validity/status |
| Manifest | `results/manifests/v3_parameter_manifest.csv` | 24개 Seed-parameter mapping/hash |
| Manifest | `results/manifests/v3_architecture_manifest.csv` | semantic architecture shape |
| Manifest | `results/manifests/v3_case_manifest.csv` | case/state/batch/device identity |
| Manifest | `results/manifests/environment.json` | 환경/config/commit identity |

BN 자체가 없으므로 BN trace directory/file은 생성하지 않았다.

## Reproducing the Diagnostic

저장소 root에서 다음 순서로 실행한다.

```bash
source .venv-metal/bin/activate
python -u experiments/external_validation/V3_bn_free_cnn/src/bnfree_preflight.py
python -u experiments/external_validation/V3_bn_free_cnn/src/bnfree_trace.py
python -u experiments/external_validation/V3_bn_free_cnn/src/compare_bn_vs_bnfree.py
```

`RUN_FULL_TRAINING=False`를 유지해야 한다. Protected artifact writer는 기존 결과와 내용이 다르면 자동 덮어쓰기를 거부한다.

## Next Validation

V3의 GAP 관찰과 Phase 2/V2의 BN batch-mean 관찰은 reduction operator를 exact-tensor microbenchmark로 분리 검증할 후속 가설을 남긴다. 이는 reduction이 root cause이거나 BN과 GAP가 같은 내부 원인을 공유한다는 결론이 아니다. 구체적인 후보는 상위 External Validation 계획 문서에 기록했다.

V4 CIFAR-10 Independent Workload Validation은 **Completed / VALID**다. Stage A에서 3/3 Seeds의 first entry가 BN1 batch mean으로 관찰됐고 Stage B 6-run full training과 trajectory 분석도 완료됐다.
