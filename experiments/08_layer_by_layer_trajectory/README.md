# Experiment 08 - Layer-by-Layer Training Trajectory Analysis

Phase 2 - Strict Controlled Framework Comparison의 후속 분석 placeholder다.

Experiment 07에서 anchor/Seed sensitivity가 관찰된 checkpoint를 중심으로 Conv1 → BN1 → ReLU → Pool → Conv2 … → FC → Logits → Loss → Gradient → Update 순서에서 state-controlled one-step difference가 확대되는 위치를 추적한다.

우선 분석할 representative case는 다음과 같다.

1. Initial / shared anchor — synchronized 초기 baseline
2. Epoch 1 — representative low-divergence control
3. Epoch 30 / Seed 123 / K-anchor — late-stage local spike
4. Epoch 30 / Seed 2026 / K-anchor — late-stage local spike
5. 동일 Epoch 30 Seed의 P-anchor — anchor control

Forward activation, layer-wise gradient와 CommonAdam update를 단계별로 비교해 최초 non-zero 위치와 amplification stage를 구분한다.

현재 상태는 **Next after Experiment 07**이다. 이전 OFAT 설계의 미실행 코드는 [archive/pre_redesign](archive/pre_redesign/)에 보존했으며 이 Phase 2 결과로 사용하지 않는다.
