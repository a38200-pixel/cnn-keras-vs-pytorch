# V2 - ResNet18 + BatchNorm 아키텍처 검증

## 상태

**Completed / VALID**

## 연구질문

> 실행환경, dataset, input, optimization semantics 및 synchronization 절차를 고정한 상태에서 custom CNN을 의미적으로 동등한 ResNet18로 교체하면 framework 간 최초 수치 차이의 위치와 전파 양상이 달라지는가?

V2는 Phase 2 GPU 환경을 유지하고 아키텍처만 변경하는 initial synchronized one-step 검증이다. ResNet18의 장기 trajectory, test generalization 또는 Framework 우열은 분석 범위가 아니다.

## 변경 조건과 고정 조건

| 구분 | V2 조건 |
|---|---|
| 변경 요인 | Phase 2 custom CNN → post-activation ResNet18 |
| 실행 장치 | TensorFlow/Keras Metal GPU, PyTorch MPS GPU |
| Dataset/input | 동일 Young AffectNet HQ train split, 고정 32개 batch, 128×128 RGB float32 |
| Objective | 동일 integer-label mean cross-entropy, raw logits |
| Optimizer/BN | 동일 CommonAdam, 동일 Common BatchNorm |
| 초기화 | 동일 policy와 Seeds 42/123/2026, Seed별 exact shared canonical ResNet18 W0 |
| Case | Initial shared state 3개만 수행 |
| 범위 | Synchronized one-step trace만 수행, Full Training 없음 |

아키텍처가 다르므로 custom CNN의 W0 tensor를 재사용하지 않는다. 정확한 통제는 동일 초기화 policy/Seed를 사용하면서 V2 내부의 양 framework가 Seed별 exact shared canonical ResNet18 initial state를 받는 것이다.

## Reproducibility Environment

공식 실행환경은 `results/manifests/environment.json`에 기록된 값으로 한정한다.

| 항목 | 기록값 |
|---|---|
| Experiment identity | `V2 - ResNet18 + BatchNorm Architecture Validation` |
| OS / architecture | `macOS-26.6.2-arm64-arm-64bit` / `arm64` |
| Python / NumPy | `3.10.3` / `2.0.2` |
| TensorFlow / tensorflow-metal | `2.18.1` / `1.2.0` |
| TensorFlow 실제 ResNet Conv device | `/job:localhost/replica:0/task:0/device:GPU:0` |
| PyTorch | `2.14.0` |
| PyTorch MPS | available, 실제 ResNet Conv device `mps:0` |
| MPS fallback | `PYTORCH_ENABLE_MPS_FALLBACK=0` |
| dtype | `float32` |
| Git commit | `f4e5fc51d3e4e330fe7ea37d851a73fea8eac7d1` |
| Configuration SHA-256 | `c9079c1642154cc29bb341ec12e1e337655f22f86ec3bd515fde5a4aecb4a1d2` |

Keras, torchvision 및 하드웨어 모델명은 이 V2 environment manifest에 별도 필드로 기록되지 않았으므로 사후 추정해 추가하지 않는다. `PRETRAINED=false`, `RUN_FULL_TRAINING=false`도 동일 manifest에 기록되어 있다.

## 직접 구현한 ResNet18

외부 Keras ResNet package와 `torchvision.models.resnet18()`을 서로 비교하지 않는다. Torchvision ResNet18의 architectural specification만 참고하고 양쪽 framework에 다음 구조를 직접 구현한다.

- Stem: explicit pad 3 → Conv 7×7, 64, stride 2, bias 없음 → Common BN → ReLU → explicit pad 1 → MaxPool 3×3, stride 2
- Residual stages: `(64, 128, 256, 512)` channels, 각 BasicBlock 2개
- Stage 2–4의 첫 block: stride 2와 Conv1×1+Common BN projection shortcut
- Post-activation BasicBlock: Conv3×3→BN→ReLU→Conv3×3→BN, shortcut과 add 후 ReLU
- Head: GAP → 8 logits
- Pretrained weight와 model 내부 softmax 없음

Keras stride convolution은 `ZeroPadding2D` 후 `padding="valid"`로 구현해 PyTorch의 explicit symmetric padding geometry와 정렬한다. Stem MaxPool도 ReLU 이후 양쪽에서 explicit zero padding을 사용한다.

## Common BN과 Common Adam

Common BN은 epsilon `1e-3`, population variance, running update rate `0.01`, gamma/beta `1/0`, running mean/variance `0/1`을 사용한다. NHWC/NCHW reduction axis만 layout에 맞게 다르다. 실제 forward에서 생성된 BN input, batch mean/variance, x̂, output을 저장하며 trace를 위해 BN을 다시 실행하지 않는다.

CommonAdam은 기존 learning rate `0.001`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-7`, weight decay/AMSGrad/clipping 없음 조건을 그대로 사용한다.

공식 one-step diagnostic은 양쪽 모두 **training mode**로 수행했다. Keras는 `training=True`, PyTorch는 `model.train()`을 사용하며, Common BN은 현재 batch의 population mean/variance로 정규화하고 running state를 한 번 갱신한다. 저장된 trace는 그 실제 forward에서 수집되며 계측을 위해 BN forward를 두 번째로 실행하지 않는다. Architecture-only shape preflight의 inference/eval 실행은 공식 one-step 수치 결과가 아니다.

## Canonical state와 parameter mapping

Canonical convolution layout은 HWIO, Dense layout은 IO다. Keras에는 그대로 적재하고 PyTorch에는 convolution HWIO→OIHW, Dense IO→OI로 변환한다. Stem, 8개 BasicBlock, 3개 projection shortcut, 20개 Common BN 및 classifier의 모든 trainable/non-trainable state를 semantic name으로 매핑한다.

단순 parameter count 일치는 synchronization의 충분조건이 아니다. V2 gate는 semantic parameter 목록과 shape/dtype을 먼저 맞춘 뒤, convolution kernel, classifier kernel, BN gamma/beta와 running mean/variance, projection shortcut까지 canonical layout에서 array 값과 hash가 exact인지 확인한다. Seed당 102개 state array(62개 trainable parameter group + 20개 BN의 running mean/variance)를 비교했고, 3개 Seed의 manifest 306개 행이 모두 exact였다. Trainable parameter는 framework별 `11,180,616`개, BN non-trainable state는 각각 `9,600`개다.

생성 artifact:

- `results/manifests/resnet18_parameter_manifest.csv`
- `results/manifests/resnet18_architecture_manifest.csv`
- `results/manifests/v2_case_manifest.csv`
- `results/manifests/environment.json`

## 사전 검증 Gate

Trace 전에 다음 조건을 모두 확인한다.

```text
RESNET18_ARCHITECTURE_EQUIVALENCE = VALID
RESNET18_PARAMETER_EQUIVALENCE = VALID
V2_GPU_DEVICE_CHECK = VALID
V2_FIXED_BATCH_CHECK = VALID
V2_RESYNC_PRECHECK = VALID
```

Parameter gate는 semantic list, framework별 trainable count, canonical W0, BN state 및 초기 CommonAdam m/v/step exact 여부를 포함한다. Architecture gate는 모든 semantic output shape와 Stage 2–4 downsample main/shortcut/add shape를 포함한다.

실제 결과는 다음과 같다.

- Keras/PyTorch trainable parameter: 각각 `11,180,616`개로 exact
- BN running mean/variance state: 각각 `9,600`개로 exact
- Semantic state array: Seed당 102개, 3개 Seed 총 306개 mapping 모두 exact
- Semantic trace stage: 173개 shape 모두 일치
- TensorFlow actual ResNet Conv device: `/device:GPU:0`
- PyTorch actual ResNet Conv device: `mps:0`
- Fixed batch full hash: `c3ef4b5c6cb842154ddb03fa6cdc175dd5ab4e69a1e02f2bbc5493c3e2a48e7c`, 기존 manifest와 exact

## GPU 반복 재현성과 계측 동등성

각 Seed에서 fresh state로 plain one-step과 layer-traced one-step을 비교한다. Logits, loss, global gradient, update, post-weight 및 BN running state가 exact여야 `V2_TRACE_EQUIVALENCE = VALID`이다.

Framework별 traced one-step은 fresh state에서 두 번 실행한다. 반복 재현성 기준은 실행 전에 다음과 같이 고정했다.

- `VALID`: 모든 비교 array exact
- `QUALIFIED`: 비영 차이가 있지만 모두 `atol=1e-12`, `rtol=1e-7` 이내이고 유한함
- `INVALID`: 하나 이상의 값이 허용범위를 벗어나거나 NaN/Inf 발생

실제 결과는 `V2_TRACE_EQUIVALENCE = VALID`, `V2_GPU_REPEATABILITY = VALID`이다. 각 검사는 3개 Seed × 2개 framework × 7개 global metric, 총 42개 비교에서 모두 exact였다.

## Semantic trace

Stem Conv/BN/ReLU/MaxPool, 각 residual block의 두 Conv/BN, projection shortcut, main/shortcut output, residual add, block output, GAP, logits 및 loss를 기록한다. 모든 BN은 input, batch mean, batch variance, x̂, affine output을 포함한다. 첫 비영 stage는 `max_abs_diff > 0`인 실제 첫 행으로 결정하며 Conv, BN 또는 residual Add 중 어느 위치도 미리 가정하지 않는다.

## Metric Definitions

모든 비교 array는 metric 계산 전에 `float64`로 변환해 평탄화한다. 원소별 차이를 `d = x - y`라고 할 때 지표는 다음과 같다.

- `max_abs_diff = max_i |d_i|`
- `mean_abs_diff = mean_i |d_i|`
- `relative_l2_error = ||x-y||₂ / max(||x||₂, ||y||₂, 1e-12)`
- `cosine_similarity = (x·y) / (||x||₂||y||₂)`; 두 norm이 모두 0이고 array가 exact이면 `1`, 그 밖의 zero-norm case는 `0`
- `sign_mismatch_fraction`: 절댓값이 `1e-12` 이하인 값을 0으로 매핑한 뒤 부호가 다른 원소의 비율

분모가 매우 작은 지표는 ratio가 커질 수 있으므로 absolute metric과 함께 해석한다. 특히 이 문서의 `7.03×–7.34×`는 Seed별 **initial one-step maximum-forward relative L2**의 ResNet18/Custom CNN 비율이지, 정확도·학습 안정성·모델 품질의 비율이 아니다.

### First non-zero 판정과 세 가지 위치

공식 최초 비영 stage는 forward CSV의 semantic order를 따라 `max_abs_diff > 0`을 처음 만족하는 행이다. 별도 tolerance를 적용해 작은 값을 0으로 만들지 않는다. 따라서 다음 개념을 구분한다.

- **Entry point:** 위 규칙으로 관찰된 최초 비영 stage
- **Largest observed difference:** 해당 비교 범위에서 `relative_l2_error`가 가장 큰 stage
- **Root cause:** 통제 실험으로 인과성이 확인된 원인

V2는 앞의 두 항목을 보고하지만 root cause를 확정하지 않는다. `stem.bn.batch_mean`의 첫 비영 값은 Seed별 max absolute difference `1.49e-8–2.98e-8`, relative L2 `3.85e-8–4.84e-8` 수준이었다. 이 미세한 관찰이 해당 연산을 장기 성능 차이의 원인으로 만들지는 않는다.

## Seed별 결과

| Seed | 최초 비영 forward stage | 최대 forward stage (relative L2) | 최대 residual Add | 최대 backward stage | 최대 parameter gradient | 최대 update |
|---:|---|---:|---:|---|---|---|
| 42 | stem.bn.batch_mean | stage4.block2.bn2.x_hat (`4.0903e-6`) | stage4.block2.add (`3.5101e-6`) | stage1.block1.add | stage1.block1.bn2.beta | stage2.block2.bn2.gamma |
| 123 | stem.bn.batch_mean | stage4.block2.bn2.x_hat (`4.0990e-6`) | stage4.block2.add (`3.5336e-6`) | stage1.block2.conv1 | stem.bn.beta | stage3.block1.bn2.gamma |
| 2026 | stem.bn.batch_mean | stage4.block2.bn2.x_hat (`4.2339e-6`) | stage4.block2.add (`3.6318e-6`) | stage4.block2.conv1 | stage1.block1.bn2.beta | stage1.block1.bn1.gamma |

3개 Seed 모두 input과 stem Conv output이 exact였고 최초 비영 값은 stem BN batch-mean reduction에서 나타났다. 따라서 tested GPU stack에서 custom CNN의 `BN1 batch mean` 관찰은 ResNet18의 대응 위치인 `stem.bn.batch_mean`에서도 유지됐다.

## Residual Add와 순전파

Residual Add는 최초 진입점이 아니었다. 그러나 세 Seed 모두 Stage 1 block 1의 add relative L2 약 `2.91e-7–2.99e-7`에서 시작해 Stage 4 block 2의 `3.51e-6–3.63e-6`까지 반복적으로 증가했다. 최대 forward 차이는 모두 마지막 block의 `stage4.block2.bn2.x_hat`에서 나타났다.

이 패턴은 더 깊은 residual computation graph를 따라 이미 존재하는 차이가 누적·전파되는 현상과 일치한다. 다만 residual Add 자체가 원인 또는 독립적인 amplification mechanism임을 증명하지는 않는다.

## 역전파와 CommonAdam update

최대 backward 위치는 Seed마다 달랐다.

- Seed 42: `stage1.block1.add`, relative L2 `3.3002e-3`
- Seed 123: `stage1.block2.conv1`, relative L2 `4.9584e-3`
- Seed 2026: `stage4.block2.conv1`, relative L2 `5.3402e-3`

최대 parameter-gradient group과 update group도 Seed별로 달랐다. 고정된 하나의 residual block이나 parameter group이 세 Seed 전체의 dominant sensitivity point가 되지는 않았다. Global gradient relative L2는 `2.47e-3–5.02e-3`, global update relative L2는 `3.38e-2–5.63e-2`, post-weight relative L2는 `1.19e-3–1.99e-3`였다.

## Custom CNN Initial GPU와 직접 비교

| Seed | Custom first stage | ResNet18 first stage | Custom max forward | ResNet18 max forward | ResNet18/Custom |
|---:|---|---|---:|---:|---:|
| 42 | bn1_batch_mean | stem.bn.batch_mean | `5.8152e-7` | `4.0903e-6` | `7.03×` |
| 123 | bn1_batch_mean | stem.bn.batch_mean | `5.8085e-7` | `4.0990e-6` | `7.06×` |
| 2026 | bn1_batch_mean | stem.bn.batch_mean | `5.7712e-7` | `4.2339e-6` | `7.34×` |

Architecture 변경 후에도 최초 진입점의 semantic operation은 stem의 BN batch-mean reduction으로 유지됐다. 반면 maximum-forward relative L2는 모든 Seed에서 약 7배 커졌고 최대 forward/backward/gradient/update 위치는 custom CNN과 달라졌다. 따라서 architecture는 tested initial one-step의 **최초 진입점보다 downstream propagation magnitude와 sensitivity pattern에 더 뚜렷한 영향**을 보였다.

큰 global update/post-weight 비율, 특히 Custom CNN Seed 2026 대비 비율은 custom 값의 분모가 매우 작다는 영향도 받으므로 architecture quality 또는 장기 trajectory 차이로 해석하지 않는다.

## V1과 V2의 관계

V1 CPU custom CNN에서는 9/9 case의 최초 비영 stage가 Conv1이었고, Phase 2 GPU custom CNN에서는 9/9 case가 BN1 batch mean이었다. V2 GPU ResNet18에서는 3/3 Seed가 `stem.bn.batch_mean`이었다. 따라서 exact entry point는 execution stack에 민감하지만, 테스트한 Metal/MPS GPU 조건의 BN batch-mean entry는 두 architecture에서 반복됐다. ResNet18의 더 큰 downstream maximum은 architecture-dependent propagation을 시사하되 architecture나 BN의 보편적 인과성을 입증하지 않는다.

## Interpretation Rules

- `exact`는 저장된 비교 array가 element-wise 동일하다는 뜻이며 모든 플랫폼·버전에서의 보편적 동등성을 뜻하지 않는다.
- 최초 비영 stage는 **관찰된 진입점**이지 원인 또는 성능 분기점이 아니다.
- **Propagation**은 앞에서 생긴 차이가 이후 stage에도 관찰되는 현상이고, **local amplification**은 특정 구간에서 지표가 더 커진 관찰이다. 둘 다 그 stage가 차이를 처음 만들었다는 뜻은 아니다.
- **Dominant relative-L2 location**은 비교 범위의 최대 지표 위치이지 최초 진입점이나 causal root cause와 같지 않을 수 있다.
- Seed별 위치 차이는 그대로 보고하며 하나의 layer를 dominant 원인으로 일반화하지 않는다.
- Relative L2, gradient, update 및 parameter 차이만으로 accuracy/generalization 우열을 추론하지 않는다.
- V1/V2의 비교는 execution-stack 및 architecture sensitivity를 기술하며 TensorFlow, PyTorch, GPU, CPU, BN 또는 residual connection의 본질적 우열을 주장하지 않는다.

## V2 연구질문에 대한 답

동일한 Metal/MPS GPU 실행환경에서 custom CNN을 semantic-equivalent ResNet18로 바꿔도 3개 Seed 모두 stem Conv까지 exact였고 최초 비영 차이는 stem BN batch-mean reduction에서 나타났다. 따라서 tested GPU stack에서의 BN reduction 진입점은 이번 architecture 변경 후에도 유지됐다.

반면 ResNet18에서는 차이가 residual stages를 따라 전달되면서 최대 forward magnitude가 custom CNN보다 `7.03×–7.34×` 커졌고, 최대 backward·parameter-gradient·update 위치도 Seed 및 architecture에 따라 달라졌다. Residual Add는 새로운 최초 진입점이 아니었지만 더 깊은 block으로 갈수록 차이가 증가하는 progression을 보였다.

따라서 V2는 architecture dependence를 **보완해서 한정한다(qualify)**. 최초 수치 진입점은 tested 두 architecture에서 유지됐지만, 이후 전파 크기와 민감 위치는 architecture-dependent했다. 이 결과는 BatchNorm이 고유 원인임을 증명하지 않으며 ResNet18의 장기 학습이나 일반화 동작도 설명하지 않는다.

## 한계

- 3개 Seed의 initial synchronized one-step만 검증하며 ResNet18 장기 학습 trajectory를 다루지 않는다.
- Validation/test generalization과 checkpoint selection을 수행하지 않았다.
- 하나의 ResNet18 계열과 하나의 dataset/고정 batch만 다뤘다.
- 아키텍처 변경에는 parameter count, depth, 연산 수 및 residual connectivity 변화가 함께 포함되어 각 요소를 따로 격리하지 못했다.
- 결과는 Apple GPU의 TensorFlow Metal/PyTorch MPS, `float32` 실행에 한정되며 두 framework가 같은 저수준 kernel을 사용한다는 뜻이 아니다.
- Reduction 순서, compiler, graph lowering 및 backend kernel 내부는 계측하지 않았다.
- `7.03×–7.34×`는 paired initial-step maximum-forward 지표의 기술적 비교일 뿐, ResNet18이 더 나쁘거나 덜 정확하거나 불안정하다는 증거가 아니다.
- 다른 architecture family, 장치, dtype, batch size, framework version으로 일반화하려면 추가 격리가 필요하다.

## Source-of-Truth Artifacts

문서의 수치와 validity 판정은 다음 저장 artifact를 source of truth로 사용한다.

| 구분 | Artifact | 역할 |
|---|---|---|
| Preflight | `results/preflight/architecture_equivalence.json` | 173개 semantic stage shape와 downsample branch gate |
| Preflight | `results/preflight/parameter_equivalence.json` | parameter count, canonical W0/state exact gate |
| Preflight | `results/preflight/gpu_device_check.json` | 실제 GPU/MPS placement와 fallback gate |
| Preflight | `results/preflight/resync_precheck.json` | fixed batch 및 3-Seed full-state synchronization |
| 계측 검증 | `results/preflight/trace_equivalence.json` | plain/traced one-step 42개 비교 |
| 반복성 | `results/preflight/repeatability.json` | fresh-state GPU 반복 실행 42개 비교 |
| 공식 요약 | `results/summaries/case_summary.csv` | Seed별 entry/largest/group headline |
| 공식 요약 | `results/summaries/diagnostic_summary.json` | 전체 validity gate와 완료 상태 |
| 직접 비교 | `results/summaries/resnet18_vs_custom_cnn_initial.csv` | Seed-paired Custom CNN/ResNet18 비교 |
| Manifest | `results/manifests/resnet18_parameter_manifest.csv` | 306개 semantic state mapping, shape, hash |
| Manifest | `results/manifests/resnet18_architecture_manifest.csv` | 173개 architecture stage와 shape |
| Manifest | `results/manifests/v2_case_manifest.csv` | case identity, batch/state/device synchronization |
| Manifest | `results/manifests/environment.json` | 환경, device, config hash, commit |
| Forward | `results/forward/seed*_initial_forward.csv` | Seed별 173개 stage와 loss를 포함한 174행 metric |
| Backward | `results/backward/*activation_gradients.csv` | activation-gradient metric |
| Backward | `results/backward/*parameter_gradients.csv` | canonical parameter-gradient metric |
| Optimizer | `results/optimizer/*updates.csv` | gradient/m/v/update/post-weight metric |
| BatchNorm | `results/bn/*bn_trace.csv` | 20개 BN의 내부값과 running-state metric |

## Reproducing the Diagnostic

저장소 root에서 `.venv-metal` 환경을 사용해 아래 순서로 실행한다. 이 명령은 문서화된 diagnostic 재현 절차이며 이번 문서 audit에서는 실행하지 않았다.

```bash
source .venv-metal/bin/activate
python -u experiments/external_validation/V2_resnet18_bn/src/resnet18_preflight.py
python -u experiments/external_validation/V2_resnet18_bn/src/resnet18_trace.py
python -u experiments/external_validation/V2_resnet18_bn/src/compare_custom_vs_resnet18.py
```

첫 명령은 architecture/parameter/device/fixed-batch gate를, 두 번째는 3-Seed synchronized one-step trace를, 세 번째는 기존 Custom CNN initial GPU 결과와의 paired 비교를 생성한다. `RUN_FULL_TRAINING=False`, `PRETRAINED=False`를 유지해야 하며 보호된 artifact writer는 기존 파일과 내용이 다르면 자동 덮어쓰기를 거부한다.

## 안전장치

`RUN_FULL_TRAINING = False`, `PRETRAINED = False`를 유지한다. Epoch 학습, validation/test 평가, checkpoint 선택 및 V3/V4 구현·실행은 허용하지 않는다.

## 최종 검증 상태

```text
RESNET18_ARCHITECTURE_EQUIVALENCE = VALID
RESNET18_PARAMETER_EQUIVALENCE = VALID
V2_GPU_DEVICE_CHECK = VALID
V2_FIXED_BATCH_CHECK = VALID
V2_RESYNC_PRECHECK = VALID
V2_TRACE_EQUIVALENCE = VALID
V2_GPU_REPEATABILITY = VALID
V2_SELECTED_CASES_COMPLETE = TRUE
NAN_INF_FOUND = FALSE
NEW_FULL_TRAINING_EXECUTED = FALSE
V2 STATUS = Completed / VALID
```

이후 V3 BN-free CNN과 V4 CIFAR-10 validation은 모두 완료됐다. V2 실행 당시 후속 실험을 실행하지 않았다는 provenance는 유지하며, 최종 상태는 상위 [External Validation README](../README.md)에서 관리한다.
