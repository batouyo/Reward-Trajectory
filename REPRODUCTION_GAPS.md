# RewardFlow Paper-Faithful Reproduction Ledger

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
- Static named reward weights are a transparent first-stage substitute, not the paper's prompt-aware adaptive policy.
- The semantic cache uses a versioned JSON file keyed by SHA-256 of the edit instruction; the paper only says parses
  are cached.
- Qwen image preprocessing reproduces checkpoint resize bounds, normalization, channel order, temporal duplication,
  patch order, and grid construction with torch operations. Differentiable bicubic interpolation is not bit-identical
  to the reference PIL/NumPy bicubic path.
- The Qwen chat prefix comes from the checkpoint's official chat template. Only raw target-answer tokens, excluding
  the assistant terminator, are scored because the paper does not define the exact serialized answer boundary.
- Qwen VQA currently accepts one image per reward call. The paper reports batch size one for editing and does not
  define Q&A association for batched or multi-image reward calls.
- The VQA Eq. 4 typography is ambiguous. `qwen_vqa_token_reward` implements the textual description: target token log
  probability minus a non-negative target-vs-best-other margin penalty, so higher is always better.

## NOT-YET-IMPLEMENTED

- Full Prompt-Aware Adaptive Policy and author intent-prior templates for add/remove/style.
- Author values for `beta`, `kappa_fb`, `kappa_sch`, exact `h_i(t)`, `eta_min`, `eta_max`, `gamma_eta`, and `r0`.
- Exact intent classifier implementation and exact GPT-5 snapshot/deterministic parser outputs.
- Exact RegionCLIP region proposals, pooling, and reward implementation; the repository class remains a placeholder.
- Exact text-guided SAM2 object reward, mask-confidence mixture, leakage penalty, and add/remove direction.
- Perception Encoder reward and a differentiable HPSv2 integration.
- Provider/API integration for semantic parsing; this phase supplies only prompt, schema validation, and cache.
- Multi-Q&A aggregation and batched/multi-image Qwen VQA.
- The paper's source-latent noising initialization `z^(0) = alpha_tbar * z0 + sigma_tbar * noise`; the official
  pipeline instead samples target latents and supplies source latents as concatenated conditioning tokens.
- Pure utility implementations of the unconfigured adaptive-policy formulas; formulas are documented, but author
  configuration is not recoverable and they are not connected to the pipeline.
- VeloEdit, FLUX-Kontext, continuous editing, training, and model-weight changes are intentionally out of scope.
- A real end-to-end RewardFlow-model smoke test on the current H20 host is blocked because no official RewardFlow
  model directory is installed there. Component tests and real SigLIP/Qwen gradient smoke tests are available.
