# Experiment 04 - Common Initialization & Controlled Training

## 1. Phase Transition

00–03은 framework-native 조건의 차이를 탐색한 **Phase 1 preliminary comparison**이다. 그 결과는 그대로 보존한다. 04부터는 여러 외부 조건을 동시에 고정하고 실제 연산·학습 trajectory가 어디서 처음 달라지는지 추적하는 **Phase 2 strict controlled comparison**이다. 04는 `Baseline + Initial Weight only`인 OFAT이 아니라 Phase 2의 controlled baseline이다.

## 2. Research Question

같은 W0, input, label, batch/augmentation schedule에서 출발할 때 Forward → Loss → Gradient → Optimizer Update → BatchNorm State → Multi-step Training 중 어디에서 numerical divergence가 처음 관찰되며, 그것이 Epoch 30 결과로 어떻게 누적되는가?

## 3. Why Common Initialization Is Required

`W0_Keras ≠ W0_PyTorch`면 이후 차이를 framework 연산과 출발 상태 중 어디에 귀속할지 분리하기 어렵다. 같은 seed를 각 Framework initializer에 넣는 것만으로는 같은 numeric weight가 되지 않는다. 따라서 NumPy에서 W0를 생성해 양쪽에 transpose/import하고, 다시 canonical layout으로 export해 exact audit한다.

## 4. Controlled Variables

| Condition | Phase 2 control |
|---|---|
| Dataset/class | 동일 physical 70/15/15 split, 8-class integer labels |
| Architecture/dtype | 동일 CNN, 422,824 trainable params, float32 |
| W0 | Seed별 canonical NumPy Glorot Uniform + exact import/export |
| Input | 공통 Pillow RGB·bilinear resize·float32 `/255`, NHWC canonical |
| Batch | Seed별 저장된 NumPy permutation, batch 32, drop_last=False |
| Augmentation | 공통 PIL flip→rotation 적용 후 동일 tensor 배포 |
| Loss | 양쪽 raw logits + integer-label mean cross-entropy |
| Adam | lr=.001, β=(.9,.999), epsilon=1e-7, weight decay=0, amsgrad=False |
| BN | eps=1e-3, Keras momentum=.99 ↔ PyTorch momentum=.01 |
| Schedule | 고정 30 epochs, constant LR, EarlyStopping·LR scheduler OFF |

## 5. Remaining Framework-Native Components

TensorFlow/PyTorch Conv·BN·pool·autograd·native Adam 구현과 TensorFlow Metal/PyTorch MPS backend는 유지한다. Adam **hyperparameters**는 맞췄지만 내부 optimizer implementation이 bitwise 같다고 가정하지 않는다. 이 residual numerical difference 자체가 분석 대상이다.

## 6. Architecture

RGB 128×128 → `[Conv3×3(no bias) → BN → ReLU → MaxPool] × 4` (32/64/128/256 channels) → GAP → Dense/Linear 128 → ReLU → 8 **raw logits**. Output Softmax, Dropout, gradient clipping은 없다. Keras total count에는 BN non-trainable state가 포함될 수 있으므로 양쪽 **trainable 422,824**개를 비교한다.

## 7. Canonical Initialization

`common/controlled_initialization.py`가 NumPy `SeedSequence([seed, WEIGHT namespace])`로 Seed 42/123/2026 각각 다른 W0를 만든다. Conv HWIO·Dense IO layout, Glorot limit `sqrt(6/(fan_in+fan_out))`, bias=0, BN γ=1/β=0/mean=0/variance=1이다. PyTorch에 Conv OIHW·Linear OI로 transpose해 적재한다.

파일: [`results/initialization/`](results/initialization/)의 `seed{seed}_initial_weights.npz`, `initial_weight_manifest.json`. Seed별 전체 W0 SHA-256은 서로 다르다.

| Seed | Canonical combined W0 hash |
|---:|---|
| 42 | `f542c6536d6d7d69ec944f22a95c4dddf47e6cdd075c4340adb613b1cedd5fff` |
| 123 | `ea9200b72ccd20bba7f9252569e9bb7a2e22a47ea81c71a6a4cdaedc54aaf877` |
| 2026 | `5432cd04e8aa88919357954ba37afc13ad500cc8f21b54679e55020c242953a4` |

## 8. Canonical Input Pipeline

`common/controlled_data.py`: `Image.open → RGB → 128×128 PIL BILINEAR → common augmentation → np.float32 / 255`. Canonical batch는 NHWC다. Keras에 그대로, PyTorch에는 NCHW transpose해서 전달한다. Test/validation에는 augmentation을 적용하지 않는다. Dataset을 다시 split·이동·복사하지 않는다.

## 9. Common Batch Order

Class index·상대 경로 정렬의 10,251-sample list를 [`results/control/canonical_train_samples.csv`](results/control/canonical_train_samples.csv)에 저장한다. Seed별 NumPy namespace ORDER stream에서 30개 permutation을 생성·검증한다. 파일은 `results/control/batch_order/seed{seed}_epoch_permutations.npz`다. 각 epoch 321 batches, 마지막 batch 11 samples이며 양쪽 manual loop가 동일 indices를 읽는다. Framework-native shuffle은 사용하지 않는다.

## 10. Common Augmentation

Seed·zero-based epoch·stable sample path에서 SHA-256으로 flip(p=.5), angle Uniform[-5°, +5°]를 결정한다. PIL에서 horizontal flip→bilinear rotation(fill 0)을 수행한 **동일 canonical augmented tensor**를 양쪽에 준다. Weight, batch-order, augmentation RNG stream은 독립적으로 설계했다. TF/torchvision native random augmentation은 training path에 없다.

## 11. Loss Alignment

Keras `SparseCategoricalCrossentropy(from_logits=True)`, PyTorch `CrossEntropyLoss(reduction="mean")`이며 labels는 integer다. First-step에서는 각 native loss 외에 양쪽 logits로 NumPy reference CE를 계산한다. Epoch loss는 batch mean의 단순 평균이 아니라 `Σ(batch_loss × batch_size)/10,251`로 계산한다.

## 12. Optimizer Configuration

양쪽 native Adam에 lr=.001, β1=.9, β2=.999, epsilon=1e-7, weight_decay=0, amsgrad=False를 명시한다. Keras clipnorm/clipvalue/global_clipnorm=None·use_ema=False, PyTorch foreach=False다. 설정은 맞추되 optimizer kernel 및 epsilon placement의 잔여 차이는 분석 대상으로 남긴다.

## 13. BatchNorm Configuration

양쪽 eps=1e-3, affine/center/scale ON, 초기 γ=1·β=0·running mean=0·variance=1이다. Keras momentum=.99와 PyTorch momentum=.01은 반대 정의의 대응 설정이다. Native BN kernel과 running variance update 차이는 제거하지 않고 first-step state trace로 기록한다.

## 14. Fixed Training Schedule

두 Framework는 explicit manual loop(`tf.GradientTape` / `loss.backward`)를 사용한다. Seed마다 30 epochs × 321 batches = **9,630 optimizer steps**를 수행한다. LR=.001은 고정이며 EarlyStopping·ReduceLROnPlateau·Dropout·weight decay·gradient clipping은 없다. Validation best checkpoint는 기록만 하고 trajectory에 복원하지 않는다.

## 15. Preflight Validation

[`results/preflight/preflight_summary.json`](results/preflight/preflight_summary.json)의 기존 `CONTROLLED_PRECHECK=VALID`는 그대로 보존했다: dataset 10,251/2,194/2,203, class mapping, architecture/trainable params, float32, W0, input/labels, order, augmented tensor, config, fixed schedule, TensorFlow Metal GPU, PyTorch MPS가 PASS다. 새 [`preflight_extended_summary.json`](results/preflight/preflight_extended_summary.json)은 확장 trace·checkpoint 검증 결과를 별도로 읽으며 `EXTENDED_DIAGNOSTIC_STATUS=VALID`, `CHECKPOINT_ROUNDTRIP=VALID`, `FULL_TRAINING_READY=TRUE`다. 통제조건/구조 오류는 FAIL이지만 연구 대상인 numerical difference 자체는 FAIL이 아니다.

## 16. Initial Weight Audit

`results/preflight/seed{seed}_initial_weight_audit.csv`에 canonical·Keras·PyTorch SHA-256, shape/dtype, MAE/MSE/max 차이를 저장했다. 세 Seed의 모든 layer에서 **max_abs_diff=0, exact match**다. Seed별 W0는 서로 다르다.

## 17. Input Tensor Audit

[`input_tensor_audit.json`](results/preflight/input_tensor_audit.json)의 세 Seed 모두 canonical NHWC와 Keras input, PyTorch NCHW→NHWC의 SHA-256이 같고 **max_abs_diff=0**이다. Labels도 exact match다.

## 18. First-Step Forward Trace

Full-training model과 분리한 diagnostic model에서 동일 첫 batch를 한 번만 처리했다. 세 Seed 모두 W0·input·labels와 Conv1 output이 exact였고, 첫 **비영 차이**는 BN1이었다. 이는 “통계적으로 뚜렷한 divergence”라는 뜻이 아니다.

| Seed | BN1 max abs diff | Conv1 max abs diff | First nonzero |
|---:|---:|---:|---|
| 42 | 9.54×10⁻⁷ | 0 | BN1 |
| 123 | 7.15×10⁻⁷ | 0 | BN1 |
| 2026 | 9.54×10⁻⁷ | 0 | BN1 |

Layer별 shape, mean/std, MAE, max, MSE/RMSE, cosine은 `results/first_step/seed{seed}_forward_trace.csv`에 있다.

## 19. Loss Trace

첫 step native mean CE는 Seed 42 **2.4292793**, Seed 123 **2.2443163**, Seed 2026 **2.0988493**으로 각각 양쪽의 표시된 float32 값이 같았다. NumPy reference CE는 각 logits 기준 별도 기록했고 두 reference 간 차이는 8.62×10⁻⁹, 6.59×10⁻⁸, 5.30×10⁻⁸이다. 원본: `results/first_step/seed{seed}_loss_trace.json`.

## 20. Gradient Trace

모든 Conv kernel, BN γ/β, Dense weight/bias를 canonical layout으로 비교했다. Seed 42 Conv1 gradient max abs diff는 **3.75×10⁻⁴**였다. 다른 parameter와 Seed의 원시 MAE/MSE/cosine은 `seed{seed}_gradient_trace.csv`에 보존했다. 작은 forward 차이가 backward에서 확대될 수 있으나 단일 step만으로 원인을 단정하지 않는다.

## 21. Optimizer Update Trace

W1−W0 delta 및 W1 차이를 기록했다. Seed 42 Conv1 delta max abs diff는 **1.98×10⁻³**였다. 양쪽 native Adam 구현 차이와 gradient 차이가 섞인 관찰이며, 이후 06에서 분리 분석한다. `seed{seed}_update_trace.csv` 참조.

## 22. BatchNorm State Trace

첫 step 후 4개 BN의 running mean/variance를 canonical로 비교했다. Seed 42 BN1 variance는 exact, BN1 mean max abs diff는 **2.33×10⁻¹⁰**이었다. 다른 층/Seed의 차이는 `seed{seed}_bn_state_trace.csv`에 기록했다.

## 23. 3-Seed Training

> **Completed / VALID.** Keras와 PyTorch 모두 Seed 42/123/2026에서 30 epochs, Seed당 9,630 optimizer updates를 완료했다. 두 Framework의 config hash는 `341720f5...750f`로 일치하며 checkpoint manifest의 42개 항목(2 Framework × 3 Seeds × 7 시점)이 모두 존재한다.

Epoch 30 train accuracy는 Keras/PyTorch가 Seed 42 `65.07/64.86%`, Seed 123 `63.82/63.63%`, Seed 2026 `64.52/64.44%`였다. 3-Seed 평균은 `64.47/64.31%`로 거의 같았다. 그러나 validation과 Test에서는 더 큰 차이가 나타났으며 특히 PyTorch Seed 2026의 후반부 변화가 컸다.

## 24. Evaluation

Primary는 같은 update count에서 비교하는 **fixed Epoch 30 final model**이고, Secondary는 Framework별 best validation-loss checkpoint다. Best checkpoint는 학습 trajectory에 복원하지 않았다. 두 결과를 섞어 해석하지 않는다.

### Fixed Epoch 30 — Primary

| Seed | Keras Acc. | PyTorch Acc. | Keras Macro F1 | PyTorch Macro F1 |
|---:|---:|---:|---:|---:|
| 42 | 46.71% | 47.71% | 43.89% | 45.17% |
| 123 | 42.31% | 41.17% | 35.15% | 33.72% |
| 2026 | 41.03% | 33.09% | 40.41% | 28.79% |
| **Mean ± sample std** | **43.35 ± 2.98%** | **40.66 ± 7.32%** | **39.82 ± 4.40%** | **35.89 ± 8.40%** |

Epoch 30의 signed mean gap(`PyTorch−Keras`)은 Accuracy `−2.69%p`, Macro F1 `−3.93%p`다. 그러나 이 평균은 Seed 2026의 late-stage degradation 영향을 크게 받으며, Seed 42에서는 PyTorch가 오히려 `+1.28%p` Macro F1이었다. 이를 Framework의 절대적인 성능 우열로 해석하지 않는다.

### Best Validation Loss — Secondary

| Seed | Best epoch K/P | Keras Acc. | PyTorch Acc. | Keras Macro F1 | PyTorch Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 42 | 26 / 23 | 50.39% | 47.80% | 47.64% | 44.71% |
| 123 | 29 / 21 | 48.12% | 48.39% | 46.94% | 46.79% |
| 2026 | 22 / 26 | 49.34% | 52.02% | 47.65% | 50.67% |
| **Mean ± sample std** | — | **49.28 ± 1.14%** | **49.40 ± 2.29%** | **47.41 ± 0.41%** | **47.39 ± 3.02%** |

Best checkpoint의 평균 Macro F1은 `47.41%`와 `47.39%`로 거의 같아 signed mean gap은 `−0.02%p`였다. 다만 Seed별 paired absolute gap 평균은 `2.03%p`이므로 “모든 Seed에서 동일”하다는 뜻은 아니다. Fixed Epoch 30과 Best Validation 결과의 차이는 동일 update count에서의 최종 성능과 Framework별 generalization timing이 서로 다른 질문임을 보여준다.

## 25. Limitations

- Metal/MPS, native Conv/BN/autograd/Adam은 bitwise 동일을 보장하지 않는다.
- 현재 first nonzero BN1 차이는 약 10⁻⁶ 수준의 관찰이며 실용적 유의성을 주장하지 않는다.
- 하나의 CNN·dataset·physical split 및 3 Seeds에 제한된다.
- Best checkpoint는 각 Framework에서 서로 다른 epoch일 수 있으므로 동일 update count 비교가 아니다.
- Parameter distance와 성능 차이의 관계는 단조롭지 않으며 3 Seeds만으로 상관 또는 인과를 주장할 수 없다.

## 26. Next Experiments

Experiment 04의 first-step trace가 Forward와 Loss를 이미 포괄하므로 **05 Forward & Loss Divergence의 standalone 실행은 생략**한다. 다음 독립 분석은 **06 Gradient & Optimizer Update Divergence**이며, Adam을 원인으로 확정하지 않고 backward 차이와 native optimizer의 추가 효과를 분리한다. 07 BN State, 08 Multi-Step/Epoch, 09 Layer Trajectory는 이후 분석 대상으로 유지한다.

Preflight 및 diagnostic 재실행(학습 없음):

```bash
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/first_step_trace.py
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/early_step_trace.py
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/checkpoint_roundtrip.py
.venv-metal/bin/python experiments/04_common_initialization_controlled_training/preflight_validate.py
```

기존 VALID artifact와 새 diagnostic snapshot은 overwrite-protected다. 이미 생성된 같은 파일을 다시 만들려고 하면 기존 값을 조용히 덮지 않고 중단될 수 있다. 이후 재실험 시에는 기존 artifact를 별도 보존하고 새 경로/버전을 지정해야 한다. 일반적인 학습 직전 확인은 `preflight_validate.py`만 다시 실행하면 된다.

완료 결과는 `results/keras_controlled_results.csv`, `pytorch_controlled_results.csv`, `controlled_comparison_summary.json`과 history/trajectory/checkpoint manifest에 보존되어 있다. 결과 보호를 위해 재학습은 기본 workflow가 아니다.

## 27. Extended First-Step Diagnostic

기존 max_abs_diff 하나로는 드문 outlier와 tensor 전반의 변화를 구별할 수 없다. 원래의 `seed*_forward/gradient/update_trace.csv`는 **수정하지 않고**, `*_distribution.csv`를 별도로 만들었다. 각 비교는 MAE·median·max·MSE·RMSE·L2·relative L2·cosine·부호 불일치율·비영 차이율·p50/90/95/99/99.9를 기록한다. `relative_L2 = ||A−B||₂ / max(||A||₂, ||B||₂, 1e−12)`이며, `|x|≤1e−12`는 sign 비교에서 zero다. 비영 비율은 float64로 변환한 값의 **exact inequality**이며 유의성 판정이 아니다. `seed*_exactness_status.json`은 EXACT/NONZERO만 구분하고 유의성은 판정하지 않는다. Gradient/Update 상위 10개 차이 위치는 `*_outliers.csv`에 있다.

| Seed | BN1 비영 요소 비율 | BN1 relative L2 | BN1 max diff | First non-zero |
|---:|---:|---:|---:|---|
| 42 | 69.07% | 8.99e−8 | 9.54e−7 | BN1 |
| 123 | 68.05% | 1.13e−7 | 7.15e−7 | BN1 |
| 2026 | 69.01% | 8.77e−8 | 9.54e−7 | BN1 |

BN1의 비영 요소는 넓게 분포하지만 상대 크기는 매우 작다. **The first non-zero numerical difference was observed at BN1, but its magnitude was approximately 1e-6 and is not by itself treated as evidence of practically meaningful divergence.**

아래 global 값은 422,824개 trainable element의 canonical tensor 비교다. Cosine은 전체 dot product와 norm으로, sign mismatch는 element 수로 가중해 계산했다.

| Seed | Gradient relative L2 | Gradient cosine | Gradient sign mismatch | Update relative L2 | Update cosine | Update sign mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 9.62e−4 | 0.99999954 | 0.0144% | 2.94e−2 | 0.99957280 | 0.0144% |
| 123 | 2.83e−5 | 1.00000000 | 0% | 2.48e−2 | 0.99970292 | 0% |
| 2026 | 3.10e−5 | 1.00000000 | 0.00024% | 2.20e−2 | 0.99976550 | 0.00024% |

Seed 42 Conv1 gradient의 median 차이는 `3.28e−5`, p99.9는 `3.49e−4`, max는 `3.75e−4`다. 따라서 max 하나만의 고립된 outlier라고 보기 어렵다. 반면 Conv1 update는 median `2.24e−7`, p99.9 `3.03e−4`, max `1.98e−3`으로 꼬리의 소수 요소 영향이 크다. 원시 분포와 위치는 [first_step 산출물](results/first_step/)을 참조한다.

## 28. Adam Internal State Diagnostic

설치된 Keras optimizer slot 이름·shape를 실제 확인해 `variable.path → *_momentum/*_velocity`에 매핑하고, PyTorch `named_parameters → exp_avg/exp_avg_sq`를 canonical Conv HWIO·Dense IO로 바꿨다. 첫 step counter는 모든 Seed에서 양쪽 모두 **1**이다. Seed 42 Conv1의 m relative L2는 `1.66e−3`, v relative L2는 `1.91e−3`이다. Seed 123은 각각 `9.13e−5`, `1.57e−4`, Seed 2026은 `1.29e−4`, `1.97e−4`다. 전체 parameter 결과는 `seed*_adam_state_trace.csv`, counter/config는 `seed*_optimizer_metadata.json`에 있다.

## 29. Reference Adam Diagnostic

[`common/reference_adam.py`](../../common/reference_adam.py)는 bias-corrected NumPy float64 수식 `mₜ=β₁mₜ₋₁+(1−β₁)g`, `vₜ=β₂vₜ₋₁+(1−β₂)g²`, `Δ=−lr·m̂/(√v̂+ε)`를 계산한다. 각 Framework gradient로 따로 계산한 reference delta와 양쪽 native delta를 비교한다. 설치된 구현 소스를 확인한 결과, PyTorch의 epsilon은 bias-corrected `√v̂` 뒤에 더해지고 Keras는 uncorrected `√v` 뒤에 더해진 값을 `α=lr·√(1−β₂ᵗ)/(1−β₁ᵗ)`와 함께 사용한다. 따라서 두 native Adam은 동일 hyperparameter여도 수식 배치가 완전히 같지 않다.

| Seed | Conv1 Keras native ↔ reference relative L2 | PyTorch native ↔ reference | Reference(K-grad) ↔ reference(P-grad) |
|---:|---:|---:|---:|
| 42 | 3.05e−3 | 2.15e−6 | 6.79e−2 |
| 123 | 5.74e−3 | 2.12e−6 | 3.60e−6 |
| 2026 | 2.36e−2 | 2.11e−6 | 6.20e−2 |

Seed 123에서는 reference 두 gradient의 update가 거의 같지만 native Keras↔reference 차이가 남아 optimizer 구현의 추가 기여 가능성이 보인다. Seed 42/2026에서는 gradient 값의 작은 상대 차이도 near-zero 요소의 Adam update에서 커질 수 있다. 이는 진단적 분해이며 단일 step의 인과 증명은 아니다. 전체 결과는 `seed*_reference_adam_trace.csv`다.

## 30. Early-Step Divergence Tracking

별도의 fresh diagnostic model을 같은 W0·첫 epoch batch order에서 step `0,1,2,5,10,20,50,100`까지 업데이트했다. Full Training model에 state를 전달하지 않았다. 각 시점의 weight/gradient/update/Adam m·v/BN state 및 현재 배치 loss·accuracy는 [early_steps](results/early_steps/)에 있다.

| Seed | step 0 | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0 | .000608 | .001405 | .004475 | .010038 | .020277 | .047275 | .081144 |
| 123 | 0 | .000511 | .000618 | .001519 | .004269 | .011339 | .031571 | .060836 |
| 2026 | 0 | .000454 | .000546 | .000977 | .002512 | .008323 | .026219 | .048222 |

표는 global trainable weight의 relative L2다. Step 100에서 Conv1/Conv4/FC128/Output relative L2는 Seed 42가 `.0413/.1579/.0423/.0305`, 123이 `.0346/.1195/.0338/.0252`, 2026이 `.0279/.0974/.0275/.0198`이었다. 세 Seed 모두 이 짧은 구간에서 거리가 증가했으며, 아래 full-training trajectory에서도 증가가 이어졌다. [진단 그림](results/figures/diagnostic/)은 첫-step 3종과 Seed별 early-step 곡선을 담는다.

## 31. Training Checkpoint Strategy

완료된 Full Training은 각 Framework에서 독립적으로 `initial(step 0)`, `after_first_step(step 1)`, `epoch_001(321)`, `epoch_005(1605)`, `epoch_010(3210)`, `epoch_020(6420)`, `epoch_030(9630)`을 저장했다. 저장은 export/copy만 수행해 추가 forward/backward/update가 없었다. 기존 Seed의 history·result·checkpoint는 조용히 덮어쓰지 않으며 자동 resume도 없다. `ALLOW_OVERWRITE=False`다.

## 32. Canonical Checkpoint Format

공식 분석 산출물은 Framework-neutral NPZ다. 기존 canonical key를 고정해 Conv `conv1/kernel`은 HWIO, Dense `fc128/kernel`과 `logits/kernel`은 IO, BN은 `bn1/gamma`, `bn1/beta`, `bn1/mean`, `bn1/variance` 형식을 사용한다. Adam state는 별도 NPZ의 `<parameter>/m`, `<parameter>/v`이다. Native resume 파일은 기본 생성하지 않는다(`SAVE_NATIVE_CHECKPOINT=False`); 저장된 optimizer NPZ는 분석용이며 현재 **자동/native resume 기능은 없다**.

## 33. Checkpoint Integrity / SHA-256

각 checkpoint의 metadata JSON에는 seed·epoch·global step·batch/augmentation version·config SHA-256, canonical model/optimizer state SHA-256, NPZ 파일 SHA-256과 key 목록을 기록한다. `results/checkpoints/checkpoint_manifest.csv`에는 42개 checkpoint의 전체 경로·hash가 있으며, `compare_controlled.py`가 모든 Seed/Framework의 7개 시점과 step/config hash를 확인해 trajectory CSV를 생성했다. Diagnostic checkpoint는 별도 `results/early_steps/checkpoints/seed*/step*/`에 있다. Step 0/1의 canonical model을 fresh Keras/PyTorch model로 load→re-export한 결과가 exact였고, 저장된 Adam m/v NPZ의 hash/값 round-trip도 PASS다. Native optimizer resume round-trip은 구현·검증하지 않았다. 대용량 diagnostic/full-training checkpoint 폴더는 로컬에는 보존하되 `.gitignore`로 Git 추적에서는 제외한다.

## 34. Resume Safety and Readiness

기존 VALID preflight·first-step artifact는 보존하고 확장 진단과 학습 결과를 별도 파일에 저장했다. NaN/Inf는 forward·loss·gradient·m/v·update·BN 추출 시 검사했으며 발견되지 않았다. `CONTROLLED_PRECHECK=VALID`, `EXTENDED_DIAGNOSTIC_STATUS=VALID`, `CHECKPOINT_ROUNDTRIP=VALID`였고, 그 gate를 통과한 3-Seed controlled training과 비교도 완료됐다. 결과 해석의 유효성은 이 통제조건과 고정 config hash를 전제로 한다.

## 35. Full-Training Parameter and BN Trajectory

Global trainable-weight relative L2는 모든 Seed에서 Epoch 1부터 30까지 저장 시점마다 증가했다.

| Seed | Epoch 1 | Epoch 5 | Epoch 10 | Epoch 20 | Epoch 30 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.195 | 0.566 | 0.793 | 1.013 | 1.116 |
| 123 | 0.189 | 0.577 | 0.807 | 1.023 | 1.126 |
| 2026 | 0.174 | 0.555 | 0.784 | 1.003 | 1.104 |

Epoch 30 Conv kernel의 **absolute L2 distance**는 세 Seed 모두 `Conv1 < Conv2 < Conv3 < Conv4`였다: Seed 42 `1.18/12.32/33.25/68.57`, Seed 123 `1.68/14.04/34.88/68.54`, Seed 2026 `1.55/11.80/32.31/68.25`. 이는 깊은 층일수록 parameter 수가 증가하는 영향도 포함한다. 크기를 정규화한 relative L2에서는 깊은 Conv가 Conv1보다 대체로 컸지만 엄격한 단조 순서는 아니었고, Epoch 30에는 Conv3가 Conv4보다 큰 경우가 반복됐다. 따라서 깊은 layer에서 더 큰 trajectory separation이 관찰됐다고는 할 수 있으나, 깊이 자체가 원인이라고 단정하지 않는다.

BN running state도 학습과 함께 분리됐다. 예를 들어 Epoch 30 BN4 running-mean relative L2는 Seed 42/123/2026에서 `1.264/1.206/1.009`였다. 이는 native BN state trajectory가 달라졌음을 보여주지만 BN을 성능 차이의 원인으로 확정하지 않으며 07에서 별도 격리가 필요하다.

Seed 2026 PyTorch는 Epoch 26에 validation loss `1.3150`, validation accuracy `50.91%`로 best checkpoint를 기록한 뒤 Epoch 30에는 validation loss `2.4045`, accuracy `33.45%`로 악화됐다. Test Macro F1도 best checkpoint `50.67%`에서 fixed Epoch 30 `28.79%`로 떨어졌다. 반면 train accuracy는 Epoch 30까지 `64.44%`로 증가해 Keras `64.52%`와 거의 같았다. 이는 학습 적합도 자체보다 late-stage generalization timing의 차이를 시사하는 사례다.

Epoch 30 global weight relative L2는 세 Seed 모두 비슷한 `1.10–1.13` 범위였지만 fixed-final Macro F1 paired gap은 Seed 42 `+1.28%p`, 123 `−1.44%p`, 2026 `−11.63%p`로 크게 달랐다(`PyTorch−Keras`). 따라서 parameter distance가 크다는 사실만으로 performance gap의 크기나 방향을 예측할 수 없다.

## Answer to the Research Question

동일 초기 가중치와 동일 입력에서는 Conv1까지 exact였고, 최초의 비영 수치 차이는 BN1에서 약 `1e−6` 수준으로 관찰됐다. 이 위치를 실질적인 성능 분기점으로 취급하지 않는다.

초기 gradient는 cosine similarity가 거의 1로 방향이 매우 유사했지만 non-zero difference가 존재했고, native Adam update에서는 gradient보다 더 큰 relative divergence가 관찰됐다. NumPy reference Adam과의 비교에서도 native optimizer 수식의 추가 차이가 관찰됐으나, 이 결과만으로 Adam을 원인이라고 확정할 수 없다. 이는 backward 차이와 optimizer 구현을 더 분리할 필요가 있는 plausible amplification point다.

반복 update를 거치며 Step 0→100과 Epoch 1→30에서 parameter trajectory는 지속적으로 분리됐다. Conv kernel의 absolute distance는 깊은 layer로 갈수록 증가했고 BN running state도 크게 분리됐다. 그러나 Epoch 30 train accuracy는 두 Framework가 거의 같았고, Best Validation checkpoint의 평균 Macro F1도 Keras `47.41%`, PyTorch `47.39%`로 거의 같았다.

따라서 Experiment 04에서 Framework 차이는 분류 성능의 일관된 절대 우열보다 **optimization trajectory와 generalization timing의 차이**에서 더 뚜렷하게 나타났다. 특히 Seed 2026의 late-stage degradation은 fixed Epoch 30 결과가 best-validation 결과와 크게 달라질 수 있음을 보여준다. Parameter distance와 performance gap도 단조 관계가 아니므로, 다음 06 실험에서 gradient와 optimizer update를 추가 격리해야 한다.
