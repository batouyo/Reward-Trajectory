# Coupled Strength Trajectory Research Paths

The repository now keeps three roles separate:

1. `FluxRewardFlowPipeline` retains the RewardFlow reproduction (legacy and paper-faithful paths).
2. Its older coupled trajectory path remains an infrastructure regression environment.
3. `FluxKontextStrengthTrajectoryPipeline` is the backbone for subsequent continuous image-editing experiments.

The third path directly subclasses the repository's official `FluxKontextPipeline`. With trajectory mode disabled it
delegates to that official implementation. With trajectory mode enabled it preserves Kontext-native source-image
conditioning, CLIP pooled embeddings, T5 token embeddings, true CFG, guidance embeddings, and IP-Adapter inputs.
This is research infrastructure, not a RewardFlow paper component.

## Phase 1 definition

For one source/instruction, intermediate strengths are not independent generation tasks. They are coupled stochastic
branches with logical shape `[B, K, tokens, channels]`, flattened for model execution in B-major, strength-minor order:

```text
sample0-strength0, sample0-strength1, ..., sample1-strength0, ...
```

Every branch for one base sample, including the Kontext implementation, has:

- exactly the same initial latent, sampled once at shape `[B, ...]` and copied across K;
- exactly the same source and text conditioning, prepared once at base batch size and copied across K;
- exactly the same stochastic increment at every step, sampled once at shape `[B, ...]` and copied across K;
- a possible branch-specific drift only through `StrengthRewardFn`.

For Kontext, fixed source tokens are prepared once by the inherited VAE path and shared across strengths. Position
IDs remain the official shared, non-batched tensors; only genuinely batch-first conditioning is expanded in
B-major/K-minor order. The Transformer receives sampling tokens followed by fixed source tokens and only its sampling
prefix is treated as velocity.

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

1. Before the first branch-specific reward, only B shared trajectories run. The current state and all batch-first conditioning
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

This phase intentionally does not implement a real strength reward, source/target endpoint generation,
endpoint-relative progress, trajectory smoothness or monotonicity, Kontinuous Kontext LPIPS uniformity, VeloEdit,
temporal/spatial gating, partially correlated noise, training, or model-weight changes.

After divergence, K branches still require K Transformer evaluations. Chunking bounds peak batch memory but does not
reduce total branch compute, and reward-active chunks still retain one denoiser/decoder graph per chunk.

## Terminal early-velocity control experiment

The terminal-control path is a separate mechanism-validation experiment. It does not change the branch-local reward
contract above. For one fixed initial latent, it adds FP32 master controls to the model velocity only at a configured
prefix of steps:

```text
v_effective[t] = v_native(z[t]) + delta_velocity[t]  for t < control_steps
v_effective[t] = v_native(z[t])                      otherwise
```

The normal deterministic Euler update consumes `v_effective`; a control is never passed as `reward_drift`, whose sign
has different semantics. A differentiable final VAE decode produces an endpoint-relative diagnostic loss. Backward
then traverses every later native Kontext evaluation to the early controls. Model parameters stay frozen, but latent
Jacobians remain enabled. Non-reentrant per-step checkpointing recomputes future Transformer forwards to bound peak
activation memory. No trajectory state or native velocity is detached inside the loss graph; detached velocity copies
exist only for norm and cosine logging.

The blue-direction and pixel-interpolation endpoint targets are controllability probes, not semantic interpolation
claims. This path adds no SDE, model training, local per-step reward, or production strength-reward implementation.

### Spatiotemporally masked shared-direction diagnostic

The extended controller can restrict the early prefix to a fixed, native-trajectory-defined token support. At each
controlled step, the analytic source-restoring reference velocity is

```text
v_ref[t] = (z_native[t] - z_source) / sigma[t]
```

because the FlowMatch clean prediction `z_native[t] - sigma[t] * v_ref[t]` is exactly `z_source`. The per-token RMS
discrepancy between `v_native[t]` and `v_ref[t]` ranks the velocity-defined edit region. A top-k mask is computed once
from the zero-control native trajectory, detached, and shared by every strength and optimization iteration. Source
and sampling tokens are aligned by validating their official Kontext position IDs; unequal grids use the official
unpack layout, spatial interpolation, and official repacking. Direct control is exactly zero outside the mask, but
subsequent global Transformer dynamics can still alter the final background.

The `shared-linear` diagnostic learns one FP32 direction per controlled step and jointly accumulates terminal losses
from all strengths:

```text
effective_control[t, s] = (1 - s) * mask[t] * shared_direction[t]
```

There are no branch-specific residual directions. Regularization is applied to these effective controls averaged
over strengths and steps, rather than to the unscaled directions. This is an experimental controller-capacity test:
the pixel-interpolated endpoint remains an oracle target and is not a semantic edit-strength reward or a claim that
continuous semantic editing has been solved.

### Reward-calibrated shared magnitude diagnostic

The `shared-calibrated` mode tests whether the hand-written linear magnitude, rather than the learned shared
direction, limits oracle accuracy. It keeps the same four early directions and velocity-top-k masks. Stage A learns
those directions with the unchanged `shared-linear` controller. Stage B freezes every direction and optimizes only
four scalar interval logits:

```text
interval_drops = softmax(raw_interval_logits)
amplitude(0) = 1
amplitude(s_i) = 1 - cumulative_sum(interval_drops) at s_i
amplitude(1) = 0
effective_control[t, s] = amplitude(s) * mask[t] * shared_direction[t]
```

Positive normalized drops structurally guarantee monotonic amplitudes in `[0, 1]` and exact endpoints. Initial
drops equal the strength intervals, so Stage B begins at `amplitude(s) = 1 - s`. Its optimizer contains only the
calibration logits; frozen directions and frozen FLUX modules must have no gradients. The terminal pixel-oracle
loss is averaged across strengths and the existing regularizer is evaluated on the actual amplitude-scaled masked
controls. This is reward-calibrated scalar magnitude over a reward-learned shared early control direction. It is not
velocity interpolation and does not establish semantic strength calibration.

## Endpoint-relative semantic progress experiment

This independent research path reuses the frozen, differentiable Qwen2.5-VL teacher-forced answer scorer but does
not change RewardFlow paper mode. For each semantic primitive, it defines answer contrast and endpoint-normalized
progress as:

```text
contrast(I) = log P(target_answer | I, question) - log P(source_answer | I, question)
progress(I) = (contrast(I) - contrast(Source)) / (contrast(NativeFull) - contrast(Source))
```

Fixed Source and NativeFull anchors are prepared under `torch.no_grad()`. Candidate scoring stays grad-enabled and
never converts the candidate to PIL/NumPy. Optimization uses raw, unclamped progress; clamping is diagnostic only.
The implementation accepts multiple weighted primitives, while the first real controller experiment is explicitly
restricted to one primitive and one shared direction family.

**Research assumption:** source-vs-target answer contrast is a continuous semantic-progress coordinate. This is not
a RewardFlow paper statement. Endpoint preference, ordered pixel probes, gradient directionality, and existing
on-manifold controller outputs are separate fail-fast gates before any end-to-end optimization. Pixel oracle and
blue-direction scores remain evaluation-only and never participate in semantic backward.

The recorded `binary-contrast-v1` experiment passed Source/NativeFull discrimination but did not define a continuous
coordinate: its seven pixel-blend probes had Spearman `0.6786` with five inversions, and its existing real-controller
outputs were not strictly ordered. Endpoint discrimination is therefore not evidence of continuous progress.

## Ordinal semantic progress v2 experiment

Ordinal v2 keeps binary v1 unchanged and asks one five-way multiple-choice question per visual view. The five audited
stage descriptions correspond internally to nodes `[0, 0.25, 0.5, 0.75, 1]`; these numbers are never shown to Qwen.
One frozen Qwen forward returns next-token log probabilities for the single-token labels A-E. A softmax restricted to
those labels produces a distribution, and its expectation over the five nodes is calibrated between separately cached
Source and NativeFull expectations. Candidate expectations and calibrated progress remain differentiable and
unclamped in the loss. Fixed endpoint preparation is under `torch.no_grad()`.

**Research hypothesis:** probabilities over visually grounded ordinal descriptions may supply a continuous semantic
coordinate. The stages are text descriptions, not intermediate ground-truth images; pixel interpolation is used only
as a probe and never as training or optimization supervision. A single-question form and a preregistered equal-weight
three-question ensemble are compared without deleting failed questions or tuning temperature (`1.0`).

The v2.1 ball-color bake-off rejected both variants before controller optimization. Single-question/ensemble pixel
probe Spearman values were `0.7143`/`0.6429`, with four/six inversions. Both scored the existing `.2/.5/.8` Kontext
outputs as increasing and then decreasing. The single question chose stage D rather than E at NativeFull; only one of
the ensemble's three questions passed both endpoint top-stage checks. Both retained finite nonzero image gradients,
but the preregistered target-`.8` direction check failed because the middle probe was already scored above the target.
Consequently ordinal v2.1 did not solve binary saturation and the controller was not run.

## Relative Endpoint Semantic Reward v3

V3 is a separate research hypothesis. For each primitive, frozen Qwen sees three ordered images at once: Source,
current Candidate, and NativeFull. A second forward reverses the two reference positions. After remapping both
answers to the same direction, the production score is

```text
m1 = log P(Full choice | Source, Candidate, Full) - log P(Source choice | Source, Candidate, Full)
m2 = log P(Full choice | Full, Candidate, Source) - log P(Source choice | Full, Candidate, Source)
margin = 0.5 * (m1 + m2)
```

Source and NativeFull margins are cached under `torch.no_grad()`. A requested scalar is mapped linearly into this
instance-specific margin interval, and optimization minimizes the squared normalized margin residual without
clamping. This calibration is not pixel, latent, or velocity interpolation, and the margin is not a claim about
human-perceived semantic strength.

The old RewardFlow Figure-10 parser remains unchanged. V3 has an independent strict Source/NativeFull parser schema,
cache, CLI, and an explicit TianyuAI OpenAI-compatible Chat Completions adapter. Source and NativeFull are sent as
ordered image data URLs and the request requires strict JSON Schema output. Online parsing is one offline preparation
call, never part of differentiable inference. The service is third-party, not OpenAI. Its base URL and provider name
are included in cache identity and provenance, while `TIANYUAI_API_KEY` is read only from the environment and never
stored. Human-audited specs remain supported with explicit provenance.

The existing binary-v1 and ordinal-v2 ball experiments both failed continuous ordering. V3 therefore must pass
multi-image processor fidelity, both endpoint preferences in both orders, finite endpoint range, strict ordering on
held-out model-generated `.2/.5/.8` outputs, and bidirectional real-candidate gradients before the unchanged terminal
controller can run. Pixel blends are a secondary `NOT MODEL-GENERATED` diagnostic only.

The formal v3 ball audit also failed before controller optimization. Multi-image processor fidelity was high
(`pixel_cosine=0.999982`; grids and shapes exact), endpoint teacher-forced color answers were correct, and the real
oracle `.5` Candidate had finite nonzero gradients in both directions. However, the comparative choice itself was
position-sensitive: Source produced order-remapped margins `-0.125` and `+0.125`, while NativeFull produced `0` and
`0`. Symmetrization therefore collapsed both endpoint anchors to zero. Held-out model-generated `.2/.5/.8` raw
margins were `0`, `0.0625`, and `0.0625`, failing strict order at the last pair. The unchanged controller was not run.
Endpoint-answer validity and gradient existence are therefore still insufficient evidence of a usable continuous
semantic coordinate.

## Endpoint comparator bake-off v4

V4 freezes the controller and parser and isolates the comparator. It audits the unchanged three-image A/B formula
with true BF16 and true FP32 Qwen checkpoints, then compares it with two preregistered alternatives: independently
scored symmetric Candidate/Source and Candidate/Full full-sentence affinities, and a focus-conditioned Qwen hidden-
state distance baseline. The pairwise method teacher-forces exactly the same affirmative sentence for both endpoints;
the feature baseline uses the L2-normalized final-layer hidden state at the final prompt token and a cosine-distance
ratio. Neither score is called a human semantic percentage.

The fixed ball statement, pair prompt, feature layer/token/normalization, dtype, and score formulas are defined before
reading the held-out `.2/.5/.8` model outputs. A comparator must have a finite positive endpoint range, strict held-out
ordering without ties, both gradient directions correct for at least two of RMS steps `1e-5`, `3e-5`, and `1e-4`, and
acceptable order robustness. Step `3e-4` is diagnostic only. Spearman uses average ranks for ties and never overrides
a strict-order failure. Pixel blends remain explicitly secondary, non-model-generated diagnostics.

**Research hypothesis:** endpoint-relative structure may be recoverable by removing BF16 quantization, removing direct
three-image A/B endpoint competition, or reading a continuous internal multimodal representation. These are separate
causal hypotheses; no result is assumed in advance and the terminal controller remains unchanged and unrun in v4.

The formal H20 run rejected both language-head comparators. True FP32 removed BF16's exact `0`/`0.0625`/`0.125`
score steps, but the three-image A/B oracle scores were still non-monotonic (`.01420`, `-.00628`, `.02860`) and its two
image orders disagreed. The symmetric full-sentence pairwise coordinate ordered the oracle probes (`.88721`, `.90667`,
`.96461`), but its raw Source-to-Full affinity range was negative (`-.16898`), both unsymmetrized orders failed, and
only one of three toward-Source small steps improved. Thus it failed the endpoint, order, and gradient gates.

The FP32 focus-conditioned feature distance was the only method to pass every preregistered gate. Its Source/Full
coordinate was approximately `-0.000006`/`1.000006`; oracle `.2/.5/.8` scores were `.88297`, `.91486`, and `.95465`
with no ties, and all three small steps improved in both gradient directions. This is evidence that this Qwen contains
usable continuous endpoint structure internally while the tested language likelihood heads do not read it out
reliably. It is still a single-edit diagnostic, not a general strength metric. The controller remains frozen and was
not run; v4 provides no evidence that controller changes are needed.
