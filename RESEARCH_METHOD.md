# Coupled Strength Trajectory Research Path

This document describes our research infrastructure, not a RewardFlow paper component. Legacy RewardFlow
(`reward_guidance=True`), the paper-faithful reproduction (`paper_config.enabled=True`), and this trajectory path
(`trajectory_config.enabled=True`) are mutually exclusive.

## Phase 1 definition

For one source/instruction, intermediate strengths are not independent generation tasks. They are coupled stochastic
branches with logical shape `[B, K, tokens, channels]`, flattened for model execution in B-major, strength-minor order:

```text
sample0-strength0, sample0-strength1, ..., sample1-strength0, ...
```

Every branch for one base sample has:

- exactly the same initial latent, sampled once at shape `[B, ...]` and copied across K;
- exactly the same source and text conditioning, prepared once at base batch size and copied across K;
- exactly the same stochastic increment at every step, sampled once at shape `[B, ...]` and copied across K;
- a possible branch-specific drift only through `StrengthRewardFn`.

Therefore, if strength reward drift is zero, every strength branch for one base sample must collapse to the same
trajectory. Strength never scales velocity, noise, CFG, prompt text, or latent state directly.

The shared SDE coefficient reuses `paper_gamma_schedule` and the Euler update utility as engineering infrastructure.
That reuse is not a new RewardFlow reproduction claim. All gamma values remain explicit caller inputs.

## Reward contract

`StrengthRewardFn` receives differentiable images `[B*K, 3, H, W]`, targets `[B*K]`, and a
`StrengthRewardContext`. It must return exactly `[B*K]`, one differentiable scalar per branch. Backpropagation uses
`branch_rewards.sum()`, so increasing K does not divide each branch gradient by K.

`source_endpoint` and `target_endpoint` are reserved context fields only. This phase defines no endpoint distance or
progress metric. A future cross-branch objective must use a separate `TrajectoryRewardFn(images=[B,K,...], ...)`
instead of changing the per-branch contract.

## Output and audit trace

Flat output index is `base_sample_index * K + strength_index`. `StrengthTrajectoryPipelineOutput.grouped_images`
provides a `[B][K]` view. `pipeline.last_strength_trajectory_trace` records per-step branch rewards and norms plus
`max_shared_noise_difference`, which must be exactly zero.

## Current boundary

This phase intentionally does not implement a real strength reward, target endpoint generation, endpoint-relative
progress, trajectory smoothness or monotonicity, Kontinuous Kontext LPIPS uniformity, VeloEdit, temporal/spatial
gating, partially correlated noise, training, branch chunking, or model-weight changes.

K branches increase transformer and reward-decoder batch memory by roughly K. Correctness currently takes precedence
over chunking because chunk order must not alter the coupled random process.
