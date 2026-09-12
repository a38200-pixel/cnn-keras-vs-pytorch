# Experiment 06 - Adam Alignment

## 목적

Adam의 lr, betas, epsilon, weight decay, AMSGrad를 명시적으로 같게 해 optimizer 영향을 확인한다.

## Baseline

새 공통 모델이 아니라 두 기존 노트북의 최종 RGB CNN 구현을 각 framework 방식 그대로 재현한다.

## Hypothesis

Adam 조건이 원인이라면 baseline 대비 absolute Macro F1 gap이 감소할 것이다.

## Changed Variable

Adam parameter만 정렬하며 epsilon은 1e-7을 사용한다.

## Fixed / Unchanged Conditions

입력 128×128 RGB, batch 32, 최대 30 epochs, Adam 0.001, CNN 구조와 callback 설정을 유지한다.

## Seeds

42, 123, 2026

## Implementation

양쪽 모두 lr=.001, betas=(.9,.999), eps=1e-7, weight_decay=0, amsgrad=False이다.

## Results

> 아직 학습하지 않음.

## Comparison with Baseline

compare.py가 baseline absolute gap과 현재 gap의 change, reduction, reduction rate를 계산한다.

## Interpretation

결과 생성 후 Seed별 gap 방향과 변동성을 근거로 작성한다.

## Limitations

parameter가 같아도 두 Adam 내부 구현은 bitwise identical하지 않을 수 있다.
