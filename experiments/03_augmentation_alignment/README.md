# Experiment 03 - Augmentation Alignment

## 목적

flip 여부와 ±5° 각도를 sample/epoch별로 같게 만들어 증강 구현 영향을 확인한다.

## Baseline

새 공통 모델이 아니라 두 기존 노트북의 최종 RGB CNN 구현을 각 framework 방식 그대로 재현한다.

## Hypothesis

증강 차이가 원인이라면 baseline 대비 absolute Macro F1 gap이 감소할 것이다.

## Changed Variable

Augmentation decision/operator만 공통 PIL bilinear, constant fill로 정렬한다.

## Fixed / Unchanged Conditions

입력 128×128 RGB, batch 32, 최대 30 epochs, Adam 0.001, CNN 구조와 callback 설정을 유지한다.

## Seeds

42, 123, 2026

## Implementation

공통 hash에서 flip과 rotation을 얻는다. 디코딩/반올림 차이가 끼지 않도록 증강 경계에서 공통 PIL 연산을 사용한다.

## Results

> 아직 학습하지 않음.

## Comparison with Baseline

compare.py가 baseline absolute gap과 현재 gap의 change, reduction, reduction rate를 계산한다.

## Interpretation

결과 생성 후 Seed별 gap 방향과 변동성을 근거로 작성한다.

## Limitations

PIL 공통 연산은 양쪽 notebook의 native augmentation 자체와 같지 않으며, GPU augmentation 특성도 측정하지 않는다.
