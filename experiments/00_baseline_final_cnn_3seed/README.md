# Experiment 00 - Baseline Final CNN 3-Seed

## 목적

기존 단일 Seed에서 관찰된 framework gap이 Seed 42, 123, 2026에서도 같은 방향으로 반복되는지 확인한다.

## Baseline

새 공통 모델이 아니라 두 기존 노트북의 최종 RGB CNN 구현을 각 framework 방식 그대로 재현한다.

## Hypothesis

같은 방향의 gap이 세 Seed에서 반복되면 단일 초기화의 우연만으로 설명하기 어려운 재현 가능한 차이일 수 있다.

## Changed Variable

없음.

## Fixed / Unchanged Conditions

입력 128×128 RGB, batch 32, 최대 30 epochs, Adam 0.001, CNN 구조와 callback 설정을 유지한다.

## Seeds

42, 123, 2026

## Implementation

Keras와 PyTorch의 입력 처리, 증강, 초기화, output/loss, Adam 기본 epsilon 및 학습 loop 차이를 보존한다. 현재 데이터는 클래스별 고정 70/15/15 물리 폴더로 나뉘며 모든 실험에서 같은 표본을 사용한다.

## Results

> 아직 학습하지 않음.

현재 상태: **Baseline 3-Seed GPU 학습 준비 완료** (`.venv-metal`의 TensorFlow Metal/MPS sanity check 통과, 학습 대기).

## Comparison with Baseline

이 실험 자체가 비교 기준이다.

## Interpretation

결과 생성 후 Seed별 gap 방향과 변동성을 근거로 작성한다.

## Limitations

고정 split 하나와 3개 Seed만으로 전체 변동 분포를 완전히 추정할 수 없다.
