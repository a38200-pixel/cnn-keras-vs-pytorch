# Experiment 08 - EarlyStopping / LR Scheduler Alignment

## 목적

학습 종료와 LR 감소 epoch timing을 맞춰 callback/scheduler 영향을 확인한다.

## Baseline

새 공통 모델이 아니라 두 기존 노트북의 최종 RGB CNN 구현을 각 framework 방식 그대로 재현한다.

## Hypothesis

callback timing이 원인이라면 baseline 대비 absolute Macro F1 gap이 감소할 것이다.

## Changed Variable

EarlyStopping / ReduceLROnPlateau 동작만 정렬한다.

## Fixed / Unchanged Conditions

입력 128×128 RGB, batch 32, 최대 30 epochs, Adam 0.001, CNN 구조와 callback 설정을 유지한다.

## Seeds

42, 123, 2026

## Implementation

monitor=val_loss, min mode, min_delta=1e-4, factor=.5, min_lr=1e-6, best restore를 유지하고 PyTorch scheduler의 off-by-one patience 정의를 보정한다.

## Results

> 아직 학습하지 않음.

## Comparison with Baseline

compare.py가 baseline absolute gap과 현재 gap의 change, reduction, reduction rate를 계산한다.

## Interpretation

결과 생성 후 Seed별 gap 방향과 변동성을 근거로 작성한다.

## Limitations

고정 split 하나와 3개 Seed만으로 전체 변동 분포를 완전히 추정할 수 없다.
