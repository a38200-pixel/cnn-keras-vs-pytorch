# Experiment 07 - BatchNorm Alignment

## 목적

BN의 epsilon과 moving-statistics 갱신률 대응을 명시적으로 검증한다.

## Baseline

새 공통 모델이 아니라 두 기존 노트북의 최종 RGB CNN 구현을 각 framework 방식 그대로 재현한다.

## Hypothesis

BN 동작이 원인이라면 baseline 대비 absolute Macro F1 gap이 감소할 것이다.

## Changed Variable

BatchNorm 설정만 확인한다. 단, 기존 PyTorch notebook이 이미 Keras에 대응시켜 둔 설정이다.

## Fixed / Unchanged Conditions

입력 128×128 RGB, batch 32, 최대 30 epochs, Adam 0.001, CNN 구조와 callback 설정을 유지한다.

## Seeds

42, 123, 2026

## Implementation

Keras epsilon=.001/momentum=.99와 PyTorch eps=.001/momentum=.01을 사용한다. gamma=1, beta=0, mean=0, variance=1도 확인한다.

## Results

> 아직 학습하지 않음.

## Comparison with Baseline

compare.py가 baseline absolute gap과 현재 gap의 change, reduction, reduction rate를 계산한다.

## Interpretation

결과 생성 후 Seed별 gap 방향과 변동성을 근거로 작성한다.

## Limitations

Baseline부터 이미 정렬된 음성 대조 성격이므로 gap 변화가 없어도 BN 무관성을 증명하지는 않는다.
