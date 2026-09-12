# Experiment 02 - Batch Order Alignment

## 목적

epoch별 sample permutation과 batch grouping을 같게 만들어 mini-batch 순서의 영향을 확인한다.

## Baseline

새 공통 모델이 아니라 두 기존 노트북의 최종 RGB CNN 구현을 각 framework 방식 그대로 재현한다.

## Hypothesis

sampling order가 원인이라면 baseline 대비 absolute Macro F1 gap이 감소할 것이다.

## Changed Variable

Batch Order만 SHA-256 기반 공통 순열로 정렬한다.

## Fixed / Unchanged Conditions

입력 128×128 RGB, batch 32, 최대 30 epochs, Adam 0.001, CNN 구조와 callback 설정을 유지한다.

## Seeds

42, 123, 2026

## Implementation

두 loader가 Seed/epoch마다 같은 index 순서를 사용한다. Input Tensor 등은 baseline 방식이다.

## Results

> 아직 학습하지 않음.

## Comparison with Baseline

compare.py가 baseline absolute gap과 현재 gap의 change, reduction, reduction rate를 계산한다.

## Interpretation

결과 생성 후 Seed별 gap 방향과 변동성을 근거로 작성한다.

## Limitations

고정 split 하나와 3개 Seed만으로 전체 변동 분포를 완전히 추정할 수 없다.
