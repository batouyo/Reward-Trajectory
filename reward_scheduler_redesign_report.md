# Reward scheduler redesign experiment report

Baseline: `fe1890c5da693737ddc50153a7e5c496131cfdf1`

## Implementation

- CLIP endpoint progress is used only for nondecreasing semantic order.
- Semantic coverage, coarse-gap, and legacy threshold fine-jump remain diagnostic-compatible but have default/effective weight zero.
- Adjacent order is an active-only linear hinge over `q_i > q_(i+1)`.
- Coarse pairwise order is the mean linear hinge over violated anchor pairs only.
- DreamSim first-order loss is `max(d_i) / max(sum(d_i), eps)`; second-order triangle deficit is retained.
- Dynamic phases are order-only recovery, transition, and reversible trajectory refinement.

## Unit tests

`PYTHONDONTWRITEBYTECODE=1 PYTHOOPATH=src pytest -q tests/pipelines/rewardflow`

Result: **270 passed, 16 skipped**.

## Real runs

Settings: 128x128, 12 denoise steps, 2 controlled steps, 5 trajectory nodes, 12 outer iterations, seed `20260914`, learning rate `0.05`, BF16, dynamic scheduler comparison.

### reversed_D

Initial q: `[0.0000, 0.9310, 0.8962, 0.6676, 1.0000]`

Final q: `[0.0000, 0.9269, 0.8710, 0.6855, 1.0000]`

Active adjacent violations per iteration: `2` for all 12 iterations. Active pairwise violations: `3` for all 12 iterations. Scheduler phase: `semantic_recovery` for all 12 iterations; refinement was not reached. DreamSim first-order changed `0.3497 -> 0.3560`; second-order changed `9.4196 -> 5.0047`. No NaN/OOM.

### collapsed_D

Initial q: `[0.0000, 0.8764, 0.8766, 0.8709, 1.0000]`

Final q: `[0.0000, 0.8449, 0.8695, 0.8947, 1.0000]`

Scheduler phases by iteration: recovery (1-5), transition (6-7), recovery (8-9), transition (10-11), refinement (12). DreamSim adjacent distances changed from `[0.24513, 0.000995, 0.000902, 0.020466]` to `[0.241292, 0.000752, 0.000531, 0.023092]`; first-order changed `0.9164 -> 0.9082`; second-order changed `0.9930 -> 0.0687`. No NaN/OOM.

Scalar-D residual ratios were nonzero after optimization: reversed final branches `[0.6053, 0.6134, 0.2856]` and `[0.8190, 0.8098, 0.4629]`; collapsed final branches `[0.3998, 0.3465, 0.4203]` and `[0.6039, 0.5486, 0.6224]` across the two controlled steps.

Artifacts: `experiments/reward_scheduler_redesign_20260918/reversed_D_run/` and `experiments/reward_scheduler_redesign_20260918/collapsed_D_run/`.
