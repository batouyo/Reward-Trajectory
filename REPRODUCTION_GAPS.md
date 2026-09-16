# RewardFlow Paper-Faithful Reproduction Ledger

The strength trajectory research path is intentionally separate from RewardFlow paper reproduction.

This ledger separates statements recoverable from the CVPR 2026 paper and supplement, behavior inherited from the
official repository, explicit engineering assumptions in the opt-in implementation, and intentionally missing work.
The legacy `reward_guidance=True` path remains separate.

## PAPER-SPECIFIED

- Flow-matching produces a predicted clean latent, which is decoded before differentiable reward evaluation.
- Reward gradients propagate through both decoder and denoiser Jacobians; PyTorch autograd implements this chain.
- The clean-latent tether has energy `0.5 * ||clean_pred - z0||^2` and drift `-lambda_KL * grad_z(energy)`.
- Image editing uses `lambda_KL = 1.5`; text-to-image generation disables the tether with `lambda_KL = 0`.
- The reverse update combines backbone, reward, and KL drifts and adds Gaussian noise with standard deviation
  `sqrt(2 * gamma_k * eta_k)`.
- The noise schedule has functional form
  `gamma_min + (gamma_max - gamma_min) * (t_k / t_bar) ** rho`, with positive parameters and stronger early noise.
- The VQA reward uses frozen Qwen2.5-VL 3B, teacher-forced target-answer token logits, negative cross-entropy, and a
  margin term. The supplement describes an approximate maximum answer length of 70 tokens.
- Figure 10 specifies 5-12 semantic primitives of at most six words and exactly one visually answerable Q&A pair;
  parsing is performed once and cached.
- The adaptive-policy formula, feedback signal, object direction, and reward-aware step-size formula are stated in
  the paper, but their author configurations are not fully disclosed.

## REPO-SPECIFIED

- `FlowMatchEulerDiscreteScheduler` uses deterministic Euler update
  `sample + (sigma_next - sigma) * model_output` and its stochastic branch defines
  `x0 = sample - sigma * model_output`.
- The official RewardFlow pipeline packs latents, applies VAE BatchNorm statistics, concatenates reference-image
  tokens for conditioning, and uses optional CFG.
- Legacy `_apply_reward_guidance` evaluates the current latent, normalizes its gradient, and then calls
  `scheduler.step`; this behavior is retained when `paper_config` is absent or disabled.
- The official pipeline eagerly initializes SigLIP and its placeholder global-CLIP `RegionCLIPReward`. Paper mode's
  built-in reward selection deliberately uses only that existing SigLIP instance.

## OUR-ASSUMPTION

- **UNKNOWN-HYPERPARAM:** exact experimental `lambda_R` is unavailable; paper reward guidance requires an explicit
  `lambda_reward`.
- **UNKNOWN-HYPERPARAM:** `gamma_min`, `gamma_max`, and `gamma_rho` are unavailable; SDE noise requires all three.
- **UNKNOWN-HYPERPARAM:** VQA `margin` and `lambda_margin` are unavailable; Qwen VQA construction requires both.
- The scheduler difference `eta = sigma_k - sigma_(k+1)` is used as the Euler-Maruyama step. The paper instead has an
  adaptive algorithm-time step, so this mapping prioritizes exact equivalence with the repository's Euler scheduler.
- Consequently the reverse-flow drift is `-model_output`; with all added drifts/noise zero this exactly reproduces
  `sample + (sigma_next - sigma) * model_output`.
- The paper defines its gamma schedule in diffusion time. This implementation maps `t_k / t_bar` to
  `sigma_k / sigma_start` because sigma is this scheduler's exposed flow-time coordinate.
- KL energy sums all non-batch latent dimensions and averages the batch. This exactly matches the stated gradient for
  batch size one; the paper does not specify multi-sample loss reduction.
- The paper defines one source `z0`. Multi-reference KL therefore requires explicit `source_image_index`; it never
  silently chooses a reference.
- `ResearchStaticRewardGuidance` is a signed-weight research utility. It is not the paper's non-negative,
  simplex-valued prompt-aware adaptive softmax policy; `StaticRewardGuidance` remains only as a compatibility alias.
- The semantic cache uses schema v2 and binds its SHA-256 key to an image-content fingerprint, normalized instruction,
  and parser-prompt version. The paper says parses are cached but does not specify the fingerprint or file format;
  schema v1 instruction-only entries are deliberately rejected rather than reused.
- Qwen image preprocessing reproduces checkpoint resize bounds, normalization, channel order, temporal duplication,
  patch order, and grid construction with torch operations. Differentiable bicubic interpolation is not bit-identical
  to the reference PIL/NumPy bicubic path. Official-vs-differentiable fidelity assertions use explicitly labeled
  engineering sanity thresholds derived from observed local-model diagnostics, not paper values.
- Paper mode freezes transformer, VAE, text-encoder, and reward-model parameters once before sampling while retaining
  input autograd. Static reward routing may use differentiable device copies across GPUs; neither behavior changes the
  paper's reward-gradient objective.
- The Qwen chat prefix comes from the checkpoint's official chat template. Only raw target-answer tokens, excluding
  the assistant terminator, are scored because the paper does not define the exact serialized answer boundary.
- Qwen VQA currently accepts one image per reward call. The paper reports batch size one for editing and does not
  define Q&A association for batched or multi-image reward calls.
- The VQA Eq. 4 typography is ambiguous. `qwen_vqa_token_reward` implements the textual description: target token log
  probability minus a non-negative target-vs-best-other margin penalty, so higher is always better.
- The endpoint-relative semantic-progress path is a new research hypothesis: target-vs-source answer contrast is
  assumed to provide a continuous coordinate between Source and NativeFull. It is not part of RewardFlow.
- Its independent cache binds both endpoint fingerprints, the instruction, and a parser version. The paper Figure-10
  cache is intentionally unchanged and incompatible with this new schema.
- Ordinal semantic progress v2 is another independent research hypothesis, not a RewardFlow component. It maps a
  Qwen A-E next-token distribution over five visually described stages to fixed nodes and then calibrates the raw
  expectation by Source/NativeFull anchors. The five nodes, multiple-choice wording, temperature `1.0`, endpoint
  validation, equal question weights, and fixed label permutation audit are preregistered engineering choices.
- Ordinal v2 uses a separate schema/cache version bound to Source fingerprint, NativeFull fingerprint, instruction,
  and parser version. Its stages are textual descriptions, not intermediate ground-truth images. Multi-question and
  multi-primitive weighted aggregation are implemented as research interfaces, while the formal ball experiment uses
  one primitive.
- Relative Endpoint Semantic Reward v3 is independent research code. It symmetrizes two three-image Qwen comparison
  forwards and calibrates the resulting Full-vs-Source preference margin using only Source and NativeFull anchors.
  Treating that calibrated margin interval as a controllable interior coordinate is an explicitly gated hypothesis,
  not a RewardFlow paper claim and not human semantic ground truth.
- V3 multi-image preprocessing concatenates the existing per-image torch-native patches in image-placeholder order.
  The official processor comparison and its `pixel_cosine >= 0.999` engineering gate are diagnostics, not paper
  thresholds.
- The V3 parser uses its own strict schema/cache and a TianyuAI OpenAI-compatible Chat Completions request with JSON
  Schema output. TianyuAI is a third-party provider, not OpenAI. Provider, base URL, model, prompt version, endpoint
  fingerprints, and instruction are cache-bound; credentials are not stored. The prompt, schema, default
  `gpt-5.6-luna` model name, and human-audit fallback are research choices and never replace the paper Figure-10 parser.
- Endpoint comparator v4 is independent research code, not a RewardFlow paper component. Its fixed affirmative
  pairwise statement, symmetric four-forward affinity, final-layer/final-prompt-token Qwen representation, L2
  normalization, cosine-distance ratio, strict gates, and gradient-step sizes are preregistered engineering choices.
- The brief does not numerically define when pairwise and feature gaps are "close". The bake-off records the explicit
  selection assumption that pairwise is close when its minimum adjacent oracle gap is at least 90% of the feature
  baseline's gap; this affects recommendation only, never comparator scores or gates.
- Semantic embedding geometry v6 is an independent measurement bake-off, not a RewardFlow paper component. Its
  Source-to-NativeFull chord projection, off-axis ratio, formal/generalization gates, exact CLIP/SigLIP checkpoints,
  and the five-case suite are preregistered engineering choices.
- The slow official CLIP and SigLIP processors quantize through an 8-bit PIL image. The torch-native differentiable
  preprocessing adapter reproduces that forward quantization with a straight-through estimator, including the
  separable bicubic intermediate, so the forward embedding can meet the `.999` fidelity gate while candidate-image
  gradients remain available. The straight-through backward is an explicit research assumption, not an official
  processor derivative.
- V6 treats Qwen endpoint answers as an endpoint-validity audit only. Its hidden representation enters the same
  geometric bake-off as a candidate encoder, but neither Qwen answer likelihood nor a language-model judgment is used
  as the continuous coordinate.
- V6 model-generated pixel-oracle probes are fixed data preparation only. They are not produced by the evaluated
  feature method, are not semantic ground truth, and cannot establish human-perceptual percentage calibration.
- Text-conditioned semantic geometry v7 is independent research code, not a RewardFlow paper component. Its matched
  semantic-text parser, CLIP/SigLIP text direction, endpoint-relative margin normalization, formal gates, and
  five-case selection rule are preregistered engineering choices.
- V7 `orthogonal_ratio` divides the text-axis projection residual norm by total candidate-to-Source image-feature
  change. This is an explicit diagnostic assumption; it is not part of the primary progress coordinate or a loss.
- V7 depends on a third-party TianyuAI `gpt-5.6-luna` parse. Provider availability, model snapshot stability, and
  semantic-text quality are external variables. Cache identity binds provider, model, base URL, parser version,
  instruction, and both endpoint fingerprints; credentials are never cached.

## NOT-YET-IMPLEMENTED

- Full Prompt-Aware Adaptive Policy and author intent-prior templates for add/remove/style.
- Author values for `beta`, `kappa_fb`, `kappa_sch`, exact `h_i(t)`, `eta_min`, `eta_max`, `gamma_eta`, and `r0`.
- Exact intent classifier implementation and exact GPT-5 snapshot/deterministic parser outputs.
- Exact RegionCLIP region proposals, pooling, and reward implementation; the repository class remains a placeholder.
- Exact text-guided SAM2 object reward, mask-confidence mixture, leakage penalty, and add/remove direction.
- Perception Encoder reward and a differentiable HPSv2 integration.
- Repeatable live TianyuAI parser validation remains environment-gated by `TIANYUAI_API_KEY` and the explicit
  `RUN_TIANYUAI_SEMANTIC_PARSER_INTEGRATION=1` slow-test flag. A credentialed `gpt-5.6-luna` parse on 2026-09-15
  succeeded with `cache_hit=false` and strict structured output; provider-side determinism and availability remain
  external dependencies. There is deliberately no fallback to unstructured text or regex parsing.
- Batched Qwen VQA association remains undefined. V3 supports an ordered list of batch-one images for one comparison;
  it does not claim general multi-image/multi-sample Q&A semantics.
- The paper's source-latent noising initialization `z^(0) = alpha_tbar * z0 + sigma_tbar * noise`; the official
  pipeline instead samples target latents and supplies source latents as concatenated conditioning tokens.
- Pure utility implementations of the unconfigured adaptive-policy formulas; formulas are documented, but author
  configuration is not recoverable and they are not connected to the pipeline.
- VeloEdit, FLUX-Kontext, continuous editing, training, and model-weight changes are intentionally out of scope.
- A real end-to-end RewardFlow-model smoke test on the current H20 host is blocked because no official RewardFlow
  model directory is installed there. Component tests and real SigLIP/Qwen gradient smoke tests are available.
- The formal ordinal-v2.1 ball-color reward failed before controller optimization: two configurations failed endpoint,
  pixel-probe monotonicity, existing Kontext-output ordering, and the preregistered target-`.8` direction gate. Thus
  discrete ordinal choices are not yet a reliable continuous coordinate for this edit, and reward-controller coupling
  remains untested in this round.
- The formal relative-endpoint-v3 ball audit also failed before controller optimization. Symmetrized Source and
  NativeFull margins both collapsed to zero because the two image orders disagreed or tied. Held-out model-generated
  `.2/.5/.8` margins were `0`, `0.0625`, `0.0625`, so strict ordering failed. Processor fidelity and both image-gradient
  directions passed, demonstrating that differentiability alone does not make the VLM preference a reliable
  continuous coordinate.
- The formal endpoint-comparator-v4 bake-off found that true FP32 removes BF16 score quantization but does not repair
  three-image A/B ordering or position/label sensitivity. Symmetric pairwise full-sentence affinity orders the three
  oracle probes only after calibration by a reversed Source-to-Full raw range and fails order and toward-Source
  gradient gates. It must not be connected to the controller.
- The FP32 focus-conditioned Qwen feature-distance baseline passed this one ball-edit bake-off, including strict
  oracle ordering and both small-step gradient gates. Generalization across objects, attributes, prompts, seeds, and
  endpoint pairs is not established. Connecting it to the frozen terminal controller is intentionally deferred to a
  separately versioned experiment.
- Feature-controller v5 freezes the v4 reward and existing shared terminal controller. Its 50% improvement, `.10`
  per-node error, 5/6 blind-order agreement, majority relation/violation rules, six fixed permutations, and top-25%
  endpoint-difference mask are explicit engineering evaluation choices, not paper thresholds.
- The brief does not numerically define a dense-curve "sudden jump". V5 reports an adjacent feature-coordinate change
  above `.2` between requested `.1` nodes as a diagnostic jump. It is not a hard controller-drive or visual gate.
- TianyuAI `gpt-5.6-luna` is an independent offline VLM judge, not human perceptual ground truth. Even unanimous blind
  ordering cannot establish exact perceived 20/50/80 calibration or cross-edit generalization.
- In formal v5 Part A, the frozen shared controller failed to reach the feature targets: Best `.2/.5/.8` coordinates
  remained `0.94277/0.95465/0.96098`, mean error improved by only 9.44%, and the dense curve contained ten descending
  pairs. This does not identify whether the limiting factor is long-horizon optimization, reward geometry off the
  native manifold, or the shared `(1-s)D` parameterization; v5 deliberately does not change any of them.
- The v6 formal ball case is not sufficient for encoder selection. SigLIP passed that case, while CLIP failed gap/span
  gates and Qwen hidden failed a low-probe and toward-Source-gradient gate; the preregistered five-case generalization
  suite must finish before any winner is accepted.
- The initially selected v6 texture case failed the Qwen Source/Full endpoint audit and was rejected before any suite
  CLIP/SigLIP geometry was computed. The next fixed real-data case, a scene reimagination, replaced it; this rejection
  and replacement are recorded in the suite manifest rather than hidden.
- V6 deliberately does not integrate, tune, or rerun the terminal controller. Even a winning encoder would establish
  only better endpoint-axis geometry on the tested probes, not controller success or exact perceived edit strength.
- The completed v6 five-case suite selected no winner. CLIP failed the formal ball gap/span gates despite 4/5 strict
  ordering and 5/5 bidirectional-gradient passes. SigLIP passed the formal ball but reached only 3/5 strict ordering,
  below the 80% gate. Qwen hidden reached 4/5 strict ordering but only 2/5 bidirectional-gradient passes and remained
  compressed near Full. These failures block controller integration.
- V6 does not determine whether a text-conditioned image direction will fix the remaining cross-case ambiguity. It is
  the next recommended isolated test because failures include global environment edits; localization alone would not
  explain or solve all observed inversions. Crop/region localization remains untested and may still be needed for
  small local attributes after the text-direction question is isolated.
- V7 does not test prompt ensembles, automatic prompt search, crops, segmentation, localization, or region encoders.
  It also cannot establish human-perceptual percentage calibration from the fixed model-generated diagnostic probes.
  Controller integration remains blocked unless one encoder passes every preregistered v7 gate.
- The completed v7 suite selected no encoder. CLIP dropped from 4/5 to 3/5 strict ordering and SigLIP stayed at 3/5;
  both retained 5/5 bidirectional gradients but had median spans below `.10` and stronger Full-side compression than
  v6. The scene case became non-monotonic for both, and the environment case remained non-monotonic. Thus matched
  semantic text alone does not resolve global progress ambiguity on this suite, and no controller claim follows.
