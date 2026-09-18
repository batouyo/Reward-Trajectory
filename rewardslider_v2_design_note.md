# RewardSlider V2 implementation audit and staged design

## Reuse boundary

The V2 path will reuse the current Reward-Trajectory implementations for
Kontext conditioning and frozen-model preparation (`FluxKontextTerminalControlPipeline`),
the official paper Euler update, source-clean latent alignment, sigma schedules,
DreamSim diagnostics, velocity relevance priors, and frozen-parameter audits.
The existing `rewards.py` `RewardGuidance` EMA/variance/clipping/softmax logic is
the reference for V2 deficit coordination; V2 will not duplicate that class or
apply its latent-update method.

The current V1 coupled path, semantic scheduler, CLIP progress diagnostics,
DreamSim first-order objective, and `native + arbitrary control` parameterization
remain available for backward compatibility. They are not the formal V2
trajectory-order objective.

## External implementation audit

VeloEdit's `compute_reference_velocity` uses
`(z_t - z_0) / (sigma + eps)`. The local VeloEdit checkout has no visible
top-level LICENSE file, so V2 reimplements this small mathematical operation
instead of copying source, with attribution in the module docstring.

RewardFlow's local `RewardGuidance` maintains detached EMA mean/variance,
normalizes and clips reward values, then applies a temperature-scaled softmax.
V2 will adapt this to explicitly signed deficit/loss values while preserving
gradient flow through the current loss values.

Kontinuous Kontext's `kl-filter-simple` computes adjacent LPIPS distances,
normalizes them by their sum, compares them with a uniform distribution using
`sum(p_i * log(p_i / u_i))`, and accepts samples at or below a configured KL
threshold. V2 will provide a tensor-native differentiable LPIPS adapter and use
this criterion for trajectory calibration; DreamSim remains diagnostic only.

## V2 stages

1. Velocity-strength scaffold: current-state keep velocity, native edit velocity
   interpolation, and the four-step intervention boundary.
2. Strictly ordered learnable alpha using interval logits and softmax gaps.
3. Tensor-native LPIPS trajectory KL objective.
4. Alpha-only differentiable terminal unroll and native parity tests.
5. Branch-specific FP32 V_goal and residual/parallel/spatial regularizers.
6. Preservation and pluggable differentiable quality reward.
7. RewardFlow-style deficit coordination and gradient-routed three-phase scheduler.
8. Plateau-gated adaptive node insertion/pruning.
9. Independent V2 pipeline, runner, JSONL audit logging, and real FLUX smoke test.

Each stage is tested and committed separately. No V1 pipeline is replaced.
