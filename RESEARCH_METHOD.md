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

User-level `StrengthRewardContext` stores endpoints only in base layout `[B, C, H, W]`; callers must never expand
them to B*K. A prompt string is shared, while a prompt list has exactly B entries. For each full batch or chunk,
`StrengthRewardGuidance` uses explicit flat/base/strength indices to create `StrengthRewardBatchContext`: endpoints
are selected only for the current N branches and prompt lists are aligned in the same B-major/K-minor order.

`StrengthRewardFn` receives differentiable images `[N, 3, H, W]`, float32 targets `[N]`, and the aligned batch
context. It must return exactly `[N]`, one differentiable scalar per branch. Reward i may depend only on image i and
its aligned fields. Batch normalization across images, cross-branch softmax, neighbor comparisons, and trajectory
objectives violate this branch-local contract. Backpropagation uses `branch_rewards.sum()`, so increasing K or changing
chunk size does not divide each branch gradient.

`source_endpoint` and `target_endpoint` are reserved context fields only. This phase defines no endpoint distance or
progress metric. A future cross-branch objective must use a separate `TrajectoryRewardFn(images=[B,K,...], ...)`
instead of changing the per-branch contract.

## Output and audit trace

Flat output index is `base_sample_index * K + strength_index`. `StrengthTrajectoryPipelineOutput.grouped_images`
provides a `[B][K]` view. `pipeline.last_strength_trajectory_trace` records per-step branch rewards and norms plus
`max_shared_noise_difference`, which must be exactly zero.

The trace also records whether reward was active, whether branches had materialized, the effective Transformer batch,
configured chunk size, and number of chunks. `reward_start_step=0` and `reward_every_n_steps=1` preserve the original
every-step behavior. Other schedules are research controls and are not mathematically equivalent to every-step reward.

## Exact-safe compute optimizations

1. Before the first branch-specific reward, only B shared trajectories run. The current state and all conditioning
   materialize to B*K exactly once immediately before that reward step. If reward never activates, expansion happens
   only for final output.
2. A non-reward step uses no input autograd graph, clean prediction, VAE reward decode, reward forward, or backward.
3. After branches diverge, Transformer compute remains branch-specific. K different latent inputs are not
   mathematically redundant and are never approximated or shared.
4. Branch chunking reduces peak memory, not theoretical FLOPs. Base SDE noise is sampled once outside chunks and
   indexed by base ownership, so chunk size cannot change RNG consumption or stochastic increments.
5. A reward may cache fixed endpoint, prompt, or other base-level features once through `prepare_context`; the cache is
   cleared after the trajectory call. This infrastructure defines no actual endpoint feature.

Callbacks require a stable visible batch, so their presence disables lazy materialization and uses eager B*K from the
start.

## Current boundary

This phase intentionally does not implement a real strength reward, target endpoint generation, endpoint-relative
progress, trajectory smoothness or monotonicity, Kontinuous Kontext LPIPS uniformity, VeloEdit, temporal/spatial
gating, partially correlated noise, training, or model-weight changes.

After divergence, K branches still require K Transformer evaluations. Chunking bounds peak batch memory but does not
reduce total branch compute, and reward-active chunks still retain one denoiser/decoder graph per chunk.
