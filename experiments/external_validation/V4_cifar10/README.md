# V4 - CIFAR-10 Independent Workload Validation

## Final Status

**Completed / VALID**

- Stage A synchronized initial one-step: Completed / VALID
- Stage B controlled full training: Completed / VALID
- Training runs: 3 Seeds × 2 frameworks = 6/6 complete
- Test evaluation: complete
- NaN/Inf: not found

V4는 dataset-only intervention이 아니다. Young AffectNet HQ에서 CIFAR-10으로 전환하며 image domain, dataset size, input resolution, spatial reduction geometry, class 수, classifier, steps/epoch, total updates와 augmentation policy가 함께 달라진다. 따라서 공식 명칭은 **Independent Workload Validation**이다.

## Research Questions

> Independent CIFAR-10 workload에서도 tested GPU execution stack의 initial numerical divergence pattern과 subsequent training trajectory separation이 관찰되는가?

- **Stage A:** Exact shared initialization과 fixed batch에서 first cross-framework numerical difference는 어디인가?
- **Stage B:** W0, split, order, batch, loss, CommonBN과 CommonAdam을 정렬해도 30-epoch full workload에서 trajectory separation이 나타나는가?

## Configuration

| Item | Value |
|---|---|
| Dataset | CIFAR-10 |
| Official data | 50,000 train / 10,000 test |
| Controlled split | 45,000 train / 5,000 validation / 10,000 test |
| Per class | 4,500 train / 500 validation |
| Seeds | 42, 123, 2026 |
| Frameworks | TensorFlow/Keras, PyTorch |
| Epochs / batch | 30 / 32, `drop_last=False` |
| Steps | 1,407/epoch, 42,210 total |
| Preprocessing | uint8 RGB → float32 / 255 |
| Augmentation | None |
| Optimizer | CommonAdam, lr=.001, β1=.9, β2=.999, ε=1e−7 |
| BatchNorm | CommonBN, ε=1e−3, population variance, update rate=.01 |
| dtype | float32 |
| Devices | TensorFlow `GPU:0`, PyTorch `mps:0` |
| MPS fallback | disabled |

Framework pair는 Seed별 exact canonical W0, split, epoch permutation, batch membership와 within-batch order를 공유한다. Conv1–4, BN affine/running state와 FC128 body W0는 Phase 2에서 exact 재사용했고 10-class classifier만 Seed별 deterministic Glorot-uniform canonical W0를 생성해 양쪽에 동일하게 load했다. Trainable parameters는 각각 `423,082`, BN running state는 960 values다.

## Validity Gates

Persisted artifacts의 최종 상태는 다음과 같다.

```text
V4_DATASET_INTEGRITY = VALID
V4_SPLIT_CHECK = VALID
V4_ARCHITECTURE_EQUIVALENCE = VALID
V4_PARAMETER_EQUIVALENCE = VALID
V4_GPU_DEVICE_CHECK = VALID
V4_FIXED_BATCH_CHECK = VALID
V4_SHARED_W0_CHECK = VALID
V4_PHASE2_SHARED_BODY_W0 = VALID
V4_RESYNC_PRECHECK = VALID
V4_TRACE_EQUIVALENCE = VALID
V4_GPU_REPEATABILITY = VALID
V4_TRAINING_CONFIG_EQUIVALENCE = VALID
V4_EPOCH_ORDER_CHECK = VALID
V4_STEP_CHECKPOINT_CHECK = VALID
V4_EPOCH_CHECKPOINT_CHECK = VALID
V4_TRAINING_RUNS_COMPLETE = TRUE
completed_runs = 6
V4_TEST_EVALUATION_COMPLETE = TRUE
NAN_INF_FOUND = FALSE
```

`new_full_training_executed=false`는 각 preflight command가 새 full training을 시작하지 않았다는 뜻이다. `training_executed_by_analysis=false`도 final analysis가 existing training artifact만 읽었다는 뜻이다. 이는 별도로 완료되어 histories/checkpoints/test metrics에 보존된 Stage B 6 runs와 충돌하지 않는다.

## Stage A Results

세 Seed 모두 exact shared initial state와 exact fixed batch에서 같은 pattern을 보였다.

```text
input exact
Conv1 exact
first non-zero forward = bn1.batch_mean
exact forward prefix length = 3
largest forward = bn4.x_hat
first backward = logits
largest backward = conv1
largest parameter gradient = conv1/kernel
largest update = conv4/kernel
```

| Seed | Largest forward rel L2 | Global gradient rel L2 | Global update rel L2 | Post-weight rel L2 |
|---:|---:|---:|---:|---:|
| 42 | 6.431674e−7 | 6.891989e−7 | 2.136102e−5 | 4.425088e−7 |
| 123 | 6.417550e−7 | 7.093393e−7 | 6.894080e−5 | 1.424425e−6 |
| 2026 | 6.402945e−7 | 7.443260e−7 | 1.428515e−4 | 2.949889e−6 |

Phase 2 custom CNN/Young AffectNet, V2 ResNet18/Young AffectNet과 V4 custom CNN/CIFAR-10의 GPU+BN 조건에서 first BatchNorm batch-mean entry가 반복됐다. CPU V1에서는 Conv1, BN-free V3에서는 GAP였으므로 이를 universal BN root cause로 일반화하지 않는다.

## Stage B Epoch-Matched Trajectory

Global weight relative L2:

| Epoch | Global step | Seed 42 | Seed 123 | Seed 2026 | Mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 1,407 | 0.4194 | 0.4103 | 0.4325 | 0.4207 |
| 5 | 7,035 | 0.8521 | 0.8441 | 0.8761 | 0.8575 |
| 10 | 14,070 | 1.0196 | 1.0160 | 1.0403 | 1.0253 |
| 20 | 28,140 | 1.1165 | 1.1118 | 1.1339 | 1.1207 |
| 30 | 42,210 | 1.1522 | 1.1475 | 1.1697 | **1.1565** |

Relative L2가 1보다 크다는 것은 normalized parameter-vector distance다. Performance가 100% 이상 다르다는 뜻이 아니다.

### E30 layer-wise distance

| Group | 3-Seed mean relative L2 |
|---|---:|
| Conv1 | 0.2880 |
| Conv2 | 0.7847 |
| Conv3 | 1.0903 |
| Conv4 | 1.2481 |
| FC128 | 0.6037 |
| Classifier | 0.3894 |

Convolution hierarchy에서는 `Conv1 < Conv2 < Conv3 < Conv4`가 반복됐다. Conv4나 depth를 root cause로 해석하지 않는다.

### E30 BN running mean distance

| BN state | 3-Seed mean relative L2 |
|---|---:|
| BN1 | 0.3550 |
| BN2 | 0.3282 |
| BN3 | 0.4215 |
| BN4 | 0.8358 |

CommonBN formula가 같아도 diverged weights가 다른 activations/batch statistics를 만들면 running state는 다시 갈라질 수 있다. Large BN-state divergence는 already-diverged training state의 downstream consequence일 수 있다.

## Step-Matched Comparison with Phase 2

동일 epoch 수는 동일 optimization budget이 아니다. Phase 2는 30 epochs에 9,630 steps, V4는 42,210 steps이므로 same-update-budget analysis를 함께 사용한다.

| Global step | V4 CIFAR-10 | Phase 2 Young AffectNet HQ |
|---:|---:|---:|
| 0 | 0.0000 | 0.0000 |
| 321 | 0.1677 | 0.1823 |
| 1,605 | 0.4521 | 0.5522 |
| 3,210 | 0.6345 | 0.7759 |
| 6,420 | 0.8337 | 0.9915 |
| 9,630 | 0.9416 | 1.0976 |

V4에서도 same update budget에서 substantial separation이 나타났지만 magnitude는 Phase 2보다 작았다. Exact magnitude는 workload-dependent다. Step matching은 workload를 causal-equivalent하게 만들지 않는다.

## Training and Validation Trajectories

- E30 train accuracy mean: Keras `98.296%`, PyTorch `98.359%`
- 30-epoch mean absolute train accuracy gap: `0.136%p`
- 30-epoch mean absolute validation accuracy gap: `2.687%p`
- Mean absolute train/validation loss gap: `0.00296/0.19352`

Large E30 parameter distance `1.1565`는 nearly identical training fit과 공존했다. Validation/generalization trajectory는 separated optimization state에 더 민감했지만 direct causality를 증명하지 않는다.

## Final Epoch 30 Test

| Seed | Keras loss | Keras Acc | Keras F1 | PyTorch loss | PyTorch Acc | PyTorch F1 | Acc P−K |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.18360 | .7766 | .77899 | 1.37918 | .7546 | .75593 | −.0220 |
| 123 | 1.26051 | .7723 | .76857 | 1.26611 | .7756 | .77398 | +.0033 |
| 2026 | 1.61352 | .7359 | .73997 | 1.44391 | .7571 | .75717 | +.0212 |

| Metric | Keras mean ± sample std | PyTorch mean ± sample std | Signed P−K | Mean absolute paired gap |
|---|---:|---:|---:|---:|
| Loss | 1.35254 ± .22926 | 1.36307 ± .08998 | +.01052 | .12360 |
| Accuracy | .76160 ± .02236 | .76243 ± .01147 | +.00083 | .01550 |
| Macro F1 | .76251 ± .02020 | .76236 ± .01008 | −.00015 | .01523 |

Seed별 방향이 교차했으며 framework-level final generalization superiority는 관찰되지 않았다.

## Best Validation-Loss Checkpoint

| Seed | Keras epoch / Acc / F1 | PyTorch epoch / Acc / F1 | Acc P−K | F1 P−K |
|---:|---|---|---:|---:|
| 42 | 5 / .7408 / .73724 | 5 / .7381 / .72785 | −.0027 | −.00938 |
| 123 | 4 / .7231 / .71857 | 6 / .7547 / .75313 | +.0316 | +.03455 |
| 2026 | 3 / .7270 / .72036 | 9 / .7832 / .78361 | +.0562 | +.06325 |

Aggregate accuracy는 Keras `.73030`, PyTorch `.75867`, signed P−K `.02837`이고 Macro F1 signed gap은 `.02948`이었다. 그러나 `n=3`, Seed42의 반대 방향과 서로 다른 selection epoch 때문에 framework superiority를 주장하지 않는다. Validation-loss optimum timing은 framework- and seed-dependent였다.

## Answer to the V4 Research Question

CIFAR-10 Stage A에서도 exact input과 Conv1 뒤 first numerical difference가 BN1 batch mean에서 관찰됐다. Stage B에서 global weight distance는 Step 9,630 `0.9416`, E30/Step 42,210 `1.1565`까지 증가했다. 따라서 tested independent workload에서도 initial entry와 long-term parameter trajectory separation이 모두 관찰됐다.

그러나 magnitude는 Phase 2보다 작았고 generalization 방향은 Seed/checkpoint에 따라 달랐다. V4는 pure dataset effect가 아니며 결과는 workload independence의 보편적 증명도 아니다. 적절한 결론은 **cross-framework trajectory separation이 독립 workload에서도 재관찰됐지만 exact magnitude와 generalization behavior는 workload- and seed-dependent였다**는 것이다.

## Artifact Index

### Validity and manifests

- [Diagnostic summary](results/summaries/diagnostic_summary.json)
- [Training validity](results/preflight/training_validity.json)
- [Dataset integrity](results/preflight/dataset_integrity.json)
- [Split manifest](results/manifests/cifar10_split_manifest.json)
- [Environment](results/manifests/environment.json)
- [Architecture manifest](results/manifests/v4_architecture_manifest.csv)
- [Parameter manifest](results/manifests/v4_parameter_manifest.csv)
- [Fixed batch manifest](results/manifests/fixed_batch_manifest.json)
- [Checkpoint manifest](results/snapshots/checkpoint_manifest.csv)

### Numerical results

- [Stage A summary](results/summaries/stage_a_case_summary.csv)
- [Epoch-matched trajectory](results/summaries/epoch_matched_trajectory_summary.csv)
- [Step-matched trajectory](results/summaries/step_matched_trajectory_summary.csv)
- [Training curves](results/summaries/training_curve_comparison.csv)
- [Final test summary](results/summaries/test_final_summary.csv)
- [Best-validation test summary](results/summaries/test_best_val_summary.csv)
- [CIFAR-10 vs Young AffectNet initial comparison](results/summaries/cifar10_vs_youngaffectnet_initial.csv)

## Closure

V4 Stage A와 Stage B는 완료됐다. 새 다운로드, 학습, checkpoint 생성 또는 artifact 재생성은 필요하지 않다. Source code는 reproducibility를 위해 보존하지만 training defaults는 비활성 상태로 유지한다. Future CUDA/NVIDIA replication, reduction microbenchmark와 additional architecture/Seed 연구는 현재 repository scope 밖이다.
