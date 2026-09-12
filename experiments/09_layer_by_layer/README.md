# Experiment 09 - Layer-by-Layer Numerical Analysis

## 목적

동일 input과 동일 numeric initial weight에서 두 framework의 output이 최초로 벌어지는 연산을 추적한다. 이는 00–08 OFAT 성능 실험과 별개의 numerical diagnostic condition이다.

## Baseline

동일 Final RGB CNN의 Conv1부터 Logits까지 비교한다.

## Hypothesis

내부 연산 차이가 있다면 특정 layer 이후 MAE/MSE가 누적되고 cosine similarity가 낮아질 것이다.

## Changed Variable

진단 가능성을 위해 Input Tensor와 Initial Weight를 의도적으로 동시에 맞춘다. one-step에서는 loss와 Adam parameter도 맞춘다.

## Fixed / Unchanged Conditions

architecture, batch, BN 대응 설정 및 평가 layer를 고정한다.

## Seeds

42, 123, 2026

## Implementation

`compare_initial.py`는 초기 모델, `compare_one_step.py`는 동일 batch 1회 update 직후, `compare_trained.py`는 04의 양쪽 checkpoint가 있을 때만 비교한다. `compare_3seed.py`가 layer별 MAE, max absolute error, MSE, cosine similarity, 양쪽 mean/std의 Seed mean±std를 만든다.

## Results

> 아직 학습하지 않음.

## Comparison with Baseline

성능 gap reduction이 아니라 activation divergence 위치를 진단한다.

## Interpretation

결과가 생성된 후 최초 divergence와 이후 누적 양상을 기록한다.

## Limitations

동일 weight라도 GPU kernel, floating-point reduction order 및 pooling tie 처리로 bitwise 차이가 날 수 있다. trained 비교는 04 checkpoint가 모두 있어야 한다.
