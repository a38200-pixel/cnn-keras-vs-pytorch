# Experiment Status

> **Overall Research Experiments: COMPLETED**
>
> Experimental phase: **Closed**
>
> 추가 실험은 이 repository의 현재 scope에 계획되어 있지 않다.

| Phase | ID | Experiment | Status | Training scope | 공식 역할 |
|---|---|---|---|---|---|
| Phase 1 | 00 | Baseline Final CNN | Preliminary / Completed | 3 Seeds × 2 frameworks | Native baseline |
| Phase 1 | 01 | Input Tensor Alignment | Preliminary / Completed | 3 Seeds × 2 frameworks | Input preprocessing pilot |
| Phase 1 | 02 | Batch Order Alignment | Preliminary / Completed; Attempt 2 VALID | 3 Seeds × 2 frameworks | Order-control pilot |
| Phase 1 | 03 | Augmentation Alignment | Preliminary / Completed / VALID | 3 Seeds × 2 frameworks | Augmentation-control pilot |
| Phase 2 | 04 | Common Initialization & Controlled Training | Completed / VALID | 3 Seeds × 2 frameworks × 30 epochs | Strict controlled baseline |
| Phase 2 | 05 | Common Adam Optimizer Control | Completed / VALID | 3 Seeds × 2 frameworks × 30 epochs | Native Adam contribution isolation |
| Phase 2 | 06 | Common BatchNorm Control | Completed / VALID | 3 Seeds × 2 frameworks × 30 epochs | Native BN semantics isolation |
| Phase 2 | 07 | Multi-Step Divergence & State Re-Synchronization | Completed / VALID | Existing checkpoints, one-step probes | Accumulated/new divergence separation |
| Phase 2 | 08 | Layer-by-Layer Training Trajectory Analysis | Completed / VALID | 9 synchronized one-step cases | First-entry/propagation localization |
| External | V1 | CPU-only Execution Validation | Completed / VALID | 9 synchronized CPU cases | Execution-stack validation |
| External | V2 | ResNet18 + BatchNorm Architecture Validation | Completed / VALID | 3 synchronized initial cases | Architecture validation |
| External | V3 | BatchNorm-Free Custom CNN Validation | Completed / VALID | 3 synchronized initial cases | BN-presence validation |
| External | V4-A | CIFAR-10 Initial One-Step Diagnostic | Completed / VALID | 3 synchronized initial cases | Independent-workload first entry |
| External | V4-B | CIFAR-10 Controlled Full Training | Completed / VALID | 6 runs × 30 epochs | Independent-workload long-term trajectory |

## Final phase status

| Research block | Status |
|---|---|
| Phase 1 — Preliminary Native Framework Comparison | Completed; pilot evidence only |
| Phase 2 — Strict Controlled Experiments | Completed / VALID |
| External Validation V1–V4 | Completed |
| Overall project | **Research experiments completed** |

Phase 1은 exact shared W0가 아니므로 causal framework comparison 근거로 사용하지 않는다. Phase 2와 External Validation의 validity 판정은 각 experiment의 persisted CSV/JSON/manifest를 따른다.

`new_full_training_executed=false` 또는 `training_executed_by_analysis=false`는 해당 preflight/analysis command가 새 학습을 시작하지 않았다는 뜻이다. 이미 완료되어 별도 training artifact로 보존된 V4 Stage B 6개 run의 존재와 충돌하지 않는다.
