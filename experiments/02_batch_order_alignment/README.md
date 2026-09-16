# Experiment 02 - Batch Order Alignment

> **OFAT audit status: INVALID / 재학습 필요.** 기존 3-Seed 실행에서 Keras 3의 pre-fit `on_epoch_end()` 호출 때문에 Keras permutation이 PyTorch보다 한 칸 앞서 사용됐다. 아래 결과는 실제 CSV/history의 기술적 기록이며 Batch Order Alignment의 인과 효과나 가설 판정에 사용하지 않는다. 코드는 실제 epoch 시작 시 order를 선택하도록 수정했고 `RUN_TRAINING = False`로 되돌렸다.

## 1. Experiment Purpose

동일한 training sample order와 mini-batch composition이 Keras-PyTorch Framework Gap과 Seed Stability에 미치는 영향을 검증하려는 실험이다.

## 2. Why This Experiment?

같은 dataset과 model이라도 sample 순서, mini-batch 조합, gradient update 순서가 달라지면 optimization trajectory가 달라질 수 있다. 따라서 Framework 외부에서 만든 동일 permutation을 양쪽에 제공해 Batch Order의 단독 개입 효과를 측정하고자 했다.

## 3. Baseline

직접 비교 기준은 Experiment 00 Final CNN의 실제 CSV다.

| Metric | Keras | PyTorch | Signed Gap |
|---|---:|---:|---:|
| Test Accuracy | 47.19 ± 2.18% | 51.17 ± 0.11% | +3.98%p |
| Macro F1 | 45.84 ± 2.45% | 50.08 ± 0.23% | +4.23%p |
| Test Loss | 1.3669 | 1.2733 | -0.0936 |

Signed Gap은 `PyTorch - Keras`다.

## 4. Hypotheses

- **H02-0:** Training Batch Order 차이는 Framework Gap의 주요 원인이 아니며, 동일화 후에도 Baseline과 비슷한 Gap이 유지된다.
- **H02-1:** Framework별 shuffle/mini-batch sequence 차이가 Gap에 영향을 준다면 동일한 order와 grouping 사용 후 Accuracy 또는 Macro F1 Gap이 감소한다.
- 추가 관찰: Batch Order가 Seed Sensitivity에 영향을 준다면 3-Seed 표준편차도 Baseline에서 달라질 수 있다.

## 5. OFAT Design

의도한 설계는 **00 Baseline + Training Batch Order only**다. Experiment 01 Input Tensor Alignment를 누적하거나 공통 input loader를 재사용하지 않았다. Experiment 01과 02는 독립 branch다.

그러나 사후 audit에서 실제 Keras 학습 order가 의도한 schedule과 어긋난 것이 확인돼 기존 실행은 유효한 OFAT 결과가 아니다.

## 6. Changed Variable

의도한 변경 변수는 다음 Batch Order 요소뿐이다.

- Epoch별 training sample permutation
- Batch sample composition과 내부 순서
- 마지막 partial batch 처리

## 7. Unchanged Variables

코드 비교에서는 아래 조건이 00 Baseline과 동일했다.

- Keras: TensorFlow decode, bilinear resize, model 내부 `Rescaling(1/255)`
- PyTorch: PIL RGB, torchvision `Resize`, `ToTensor`
- Keras `RandomFlip`/`RandomRotation`, PyTorch `RandomHorizontalFlip`/`RandomRotation`
- Framework별 default weight initialization
- Keras Softmax + probability loss, PyTorch logits + CrossEntropy
- Adam: lr/beta/epsilon/weight decay/amsgrad
- Conv/BatchNorm/ReLU/MaxPool/GAP/Dense 구조와 BN 설정
- EarlyStopping 및 ReduceLROnPlateau 설정과 Framework-specific timing
- Validation/Test preprocessing, batch size 32, sample count

History recorder, progress display와 `clear_session()`은 기록/격리를 위한 변경이며 학습 목적 변수는 아니다.

## 8. Batch Order Alignment Implementation

Train 10,251개를 `class_index → dataset-relative path`로 정렬한 뒤 다음 파일에 저장한다.

```text
results/batch_order/canonical_train_samples.csv
```

각 Seed의 `np.random.default_rng(seed)`에서 30개 permutation을 순서대로 만들고 다음 NPZ를 양쪽 Framework가 함께 읽는다.

```text
results/batch_order/seed_42_epoch_orders.npz
results/batch_order/seed_123_epoch_orders.npz
results/batch_order/seed_2026_epoch_orders.npz
```

한 permutation을 32개씩 자르므로 epoch당 321 batches이며 마지막 batch는 11 samples다. PyTorch는 fixed sampler와 `shuffle=False, drop_last=False, num_workers=0`을 사용한다.

### 발견된 Keras lifecycle bug

기존 Keras `Sequence`는 `on_epoch_end()`에서 다음 permutation으로 이동했다. Keras 3 `fit()`은 첫 실제 epoch 전에 `epoch_iterator.reset()`을 호출하며, 이 reset도 `on_epoch_end()`를 호출한다.

그 결과 기존 실행은 다음과 같이 진행됐다.

- PyTorch epoch 1: schedule index 0
- Keras epoch 1: schedule index 1
- Keras 30 epoch 실행 시 schedule index 0은 사용되지 않고 index 29가 epoch 29/30에 반복됨

수정된 코드는 `on_epoch_end()`에서 advance하지 않고 실제 `on_epoch_begin()`에서 현재 schedule을 선택한다. sanity check도 pre-fit reset → epoch begin lifecycle을 재현한다.

## 9. Alignment Sanity Check and OFAT Audit

저장된 schedule 자체와 pre-fit object 상태에서는 intended epoch 1 hash가 일치했다. 그러나 Keras 3 lifecycle을 적용해 기존 실행의 실제 첫 epoch를 재구성하면 다음처럼 다르다.

| Seed | PyTorch epoch 1 (index 0) | 기존 Keras epoch 1 (index 1) | Match |
|---:|---|---|---|
| 42 | `9a886b206e6612468b009a85ea9a7dacab036a0403d5dffca64b3fc72f42ffc2` | `ce2ad93d2f2eff66add4cfeb6b83c4f025b7b71735f0d2f79ab437ec63911453` | `False` |
| 123 | `d0f508343f1049553be6f5799d05221b4666cf5b0ed6d8b8ec8a2ba30093abd1` | `8db997f177908952d8bffe0c54168c7c559ec4c07b32ef71e9d8fd35b2f414cc` | `False` |
| 2026 | `c11e142e8c2a9d2f7a50da5d0b1ea2a3888d093e043a724fe119a6d7a4643855` | `3b3f9d6f32d0b21dafcc5ccca6afac081281a9760a84870855788bfb2a58d10d` | `False` |

Canonical list, NPZ와 preview는 정상이고 각 permutation은 10,251개 index를 중복·누락 없이 포함한다. 그러나 기존 sanity는 `fit()`의 pre-fit reset을 재현하지 않아 실제 학습 mismatch를 발견하지 못했다.

**Audit 결론: CASE C — 실제 Batch Order exact match가 아니므로 기존 Experiment 02는 invalid다.** 다른 Baseline 조건의 의미 있는 변경은 발견되지 않았지만 핵심 독립변수 통제가 실패했으므로 기존 결과를 원인 분석에 사용하지 않는다.

## 10. Experimental Environment

- Apple M1 Pro, arm64
- Python 3.10.3
- TensorFlow/Keras 2.18.1, TensorFlow Metal GPU
- PyTorch 2.14.0, MPS
- Seeds: 42, 123, 2026

## 11. 3-Seed Results — Invalid Run Record

아래 값은 실제 CSV의 기록이다. `†`는 Batch Order mismatch가 있었던 invalid run임을 뜻한다.

| Seed | Keras Accuracy | PyTorch Accuracy | Accuracy Gap | Keras Macro F1 | PyTorch Macro F1 | Macro F1 Gap |
|---:|---:|---:|---:|---:|---:|---:|
| 42† | 44.98% | 50.02% | +5.04%p | 43.28% | 49.05% | +5.77%p |
| 123† | 48.03% | 51.66% | +3.63%p | 45.96% | 51.77% | +5.81%p |
| 2026† | 35.81% | 45.62% | +9.80%p | 30.35% | 43.32% | +12.97%p |

## 12. Mean ± Std — Descriptive Only

| Metric | Keras | PyTorch |
|---|---:|---:|
| Accuracy† | 42.94 ± 6.36% | 49.10 ± 3.12% |
| Macro F1† | 39.86 ± 8.35% | 48.05 ± 4.31% |
| Test Loss† | 1.4584 | 1.3140 |

## 13. Comparison with Baseline — Descriptive Only

| Metric | Baseline | Invalid run | Raw change |
|---|---:|---:|---:|
| Keras Accuracy | 47.19% | 42.94% | -4.25%p |
| Keras Macro F1 | 45.84% | 39.86% | -5.98%p |
| PyTorch Accuracy | 51.17% | 49.10% | -2.07%p |
| PyTorch Macro F1 | 50.08% | 48.05% | -2.03%p |
| Accuracy signed gap | +3.98%p | +6.16%p | +2.18%p |
| Macro F1 signed gap | +4.23%p | +8.18%p | +3.95%p |

두 Framework 모두 평균 성능이 낮았고 Keras의 하락 폭이 더 컸다. 다만 서로 다른 batch schedule을 사용한 실행이므로 이를 Batch Order 동일화의 효과로 해석할 수 없다.

## 14. Gap Effect — Not Eligible for OFAT Conclusion

원시 계산상 Accuracy absolute gap은 3.98→6.16%p, Macro F1 absolute gap은 4.23→8.18%p다. 각각 54.75%, 93.33% 확대에 해당한다. 이 값들은 `comparison_summary.json`의 산술 결과와 일치하지만, intervention fidelity가 실패했으므로 유효한 Gap Reduction/Expansion 추정치가 아니다.

## 15. Seed Stability — Descriptive Only

| Framework / Metric | Baseline std | Invalid run std | Raw change |
|---|---:|---:|---:|
| Keras Accuracy | 2.18%p | 6.36%p | +4.18%p |
| Keras Macro F1 | 2.45%p | 8.35%p | +5.90%p |
| PyTorch Accuracy | 0.11%p | 3.12%p | +3.01%p |
| PyTorch Macro F1 | 0.23%p | 4.31%p | +4.09%p |

원시 실행에서는 두 Framework 모두 Seed variation이 커졌지만, schedule mismatch 때문에 Batch Order 동일화의 효과로 귀속하지 않는다.

## 16. Seed 2026 Observation

Keras는 Accuracy 35.81%, Macro F1 30.35%, best validation-loss epoch 7, 14 epochs에서 종료됐다. PyTorch는 Accuracy 45.62%, Macro F1 43.32%, best validation-loss epoch 14, 21 epochs에서 종료됐다.

두 Framework 모두 42/123보다 낮았지만 같은 epoch order를 사용하지 않았으므로 공통 permutation이 양쪽에 동시에 불리했다고 판단할 수 없다. Keras train loss는 2.0984→1.4079로 계속 감소한 반면 validation loss는 epoch 7의 1.6215 이후 개선되지 않고 1.8232로 끝났다. PyTorch도 train loss는 2.0580→1.1180으로 감소했지만 validation-loss 최저점은 epoch 14의 1.4020이었다. 이는 train 정체보다는 validation 변동/일반화 정체 양상에 가깝다.

## 17. Training Dynamics

- Keras 42: best validation loss epoch 26; LR 변경 epoch 11/16/21/30; 30 epochs 완료.
- Keras 123: best validation loss epoch 27, best validation accuracy epoch 29; LR 변경 epoch 9/15/24; 30 epochs 완료.
- Keras 2026: best validation loss/accuracy epoch 7; LR 0.0005가 epoch 11부터 기록; 7 bad epochs 후 epoch 14 종료.
- PyTorch 42: best validation loss epoch 24, best validation accuracy epoch 30; LR 변경 epoch 19/29; 30 epochs 완료.
- PyTorch 123: best validation loss/accuracy epoch 29; LR 변경 epoch 17/28; 30 epochs 완료.
- PyTorch 2026: best validation loss epoch 14, best validation accuracy epoch 20; LR 0.0005가 epoch 19부터 기록; 7 bad epochs 후 epoch 21 종료.

모든 Seed에서 train loss는 전반적으로 하락했지만 validation curve는 큰 진동을 보였다. Seed 2026의 정체 시점은 Keras epoch 7과 PyTorch epoch 14로 같지 않다.

## 18. Figures

다음 여섯 파일이 실제로 존재하며 history CSV에서 생성됐다. 단, invalid run의 학습 곡선이다.

![Keras 3-Seed Loss](results/figures/keras_3seed_loss.png)

![Keras 3-Seed Accuracy](results/figures/keras_3seed_accuracy.png)

![PyTorch 3-Seed Loss](results/figures/pytorch_3seed_loss.png)

![PyTorch 3-Seed Accuracy](results/figures/pytorch_3seed_accuracy.png)

![Validation Loss Comparison](results/figures/validation_loss_3seed_comparison.png)

![Validation Accuracy Comparison](results/figures/validation_accuracy_3seed_comparison.png)

## 19. Hypothesis Evaluation

- **H02-1:** 판정 보류. 원시 gap은 감소하지 않았지만 실제 alignment가 실패했으므로 `Not Supported`라는 OFAT 증거로 채택하지 않는다.
- **H02-0:** 판정 보류. 성능과 분산이 크게 달라진 원시 결과가 있으나, mismatch 실행으로 Batch Order 영향이 없거나 있다는 결론 모두 낼 수 없다.

## 20. Interpretation

기존 실행에서는 평균 성능 하락, 더 큰 Framework Gap과 Seed variability가 관찰됐다. 그러나 핵심 조작인 동일 epoch order가 실제 `fit()`에서 유지되지 않았으므로 이 변화는 Batch Order Alignment의 결과가 아니다. Baseline Gap의 단독 원인 여부와 optimization sensitivity에 관한 해석은 수정된 코드로 3-Seed를 재실행한 뒤 내려야 한다.

## 21. Limitations

- 기존 실행은 Keras/PyTorch epoch order mismatch로 invalid다.
- 재실행 후에도 Seed 3개, 고정 split, 하나의 CNN/dataset이라는 한계가 있다.
- OFAT은 variable interaction을 직접 분리하지 못한다.
- Framework별 augmentation RNG, initialization, input processing, optimizer implementation과 callback timing은 다르다.
- TensorFlow Metal과 PyTorch MPS backend 차이가 유지된다.
- 저장된 schedule은 의도 조건을 증명하지만 실제 소비 순서의 별도 trace가 없으면 training fidelity를 완전히 증명하지 못한다.

## 22. Conclusion

기존 3-Seed 파일의 수치와 history는 정상적으로 읽히고 산술 요약도 정확하다. 하지만 Keras lifecycle bug로 실제 Batch Order가 Framework 간 일치하지 않아 Experiment 02의 OFAT 결과로는 무효다. 코드 수정과 lifecycle-aware sanity check는 완료했으며, 가설 및 Baseline 대비 효과 평가는 재학습 전까지 보류한다.
