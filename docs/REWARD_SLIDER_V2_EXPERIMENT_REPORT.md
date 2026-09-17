# RewardSlider V2: real FLUX-Kontext experiment report

Date: 2026-09-18

## Scope

This report covers the independent high-dimensional control implementation and
its real `FLUX.1-Kontext-dev` experiments on H20 GPU 2. Controls remain FP32
`[K, tokens, channels]` tensors; `D` is only an initialization/prior signal.
No endpoint-relative reward was added in this work.

## Code included

- Hierarchical semantic/DreamSim trajectory objectives and scale-invariant
  spatial energy prior.
- Full frozen-backbone two-pass image VJP path.
- Real one-pass/two-pass gradient comparison and forward replay audit.
- Explicit rejection of sliced VJP optimization: in BF16, changing replay
  batch size changes FLUX numerical derivatives materially.
- Deterministic collapsed-trajectory symmetry breaking, capped at
  `1e-4 * RMS(D)` (an explicit `ASSUMPTION` in code).

Relevant commits: `a15e60d`, `4a206c4`, `4db9446`, `71c898f`, `500bdf4`,
`7e86471`, `6dbf1c5`.

## Two-pass VJP audit, 128x128, K=3

| Replay batch | Result |
| --- | --- |
| K=3 (same batch shape) | Both control-gradient cosines were 0.999914 and 0.999859 versus one-pass; relative L2 differences were 0.01937 and 0.01973. Total loss matched exactly. |
| K=1 (sliced) | Not usable for optimization: gradient cosines were -0.9145 and -0.8206; relative L2 differences were 7.9368 and 6.7974. |

The K=1 failure is not hidden. Its replayed image mean-absolute errors were
only 0.00268--0.00358, but BF16 derivative sensitivity amplified these small
batch-size-dependent forward differences. Therefore `two_pass_vjp` with actual
optimization requires the replay batch to equal K; a smaller batch is only
permitted for zero-iteration diagnosis.

## Exact full-branch VJP feasibility

| Run | Resolution / nodes | Result |
| --- | --- | --- |
| `dense7_exact` | 128x128, N=7, K=5 | One full VJP iteration completed; no missing control gradients; frozen-module audit passed; 49.71 GB peak allocated, 21.03 s. |
| `dense10_exact` | 128x128, N=10, K=8 | One full VJP iteration completed; no missing control gradients; frozen-module audit passed; 57.42 GB peak allocated, 32.76 s. |
| `256_exact` | 256x256, N=5, K=3 | One full VJP iteration completed; no missing control gradients; frozen-module audit passed; 49.11 GB peak allocated, 20.93 s. |
| `512_exact` | 512x512, N=5, K=3 | Optimization intentionally blocked by the existing B=1 versus B=3 parity gate; no OOM occurred. |

The 512x512 block is numerical, not a control-path mismatch: branch identity
and the legacy materialization comparison were exact, while B=3 versus B=1
already differed at the first BF16 transformer forward (mean absolute error
0.01557), yielding final-latent mean absolute error 0.01203 against the 0.01
reference gate. The threshold was not relaxed.

## Stress initializations

- Reversed-D initialization: exact full-branch run completed all 8 iterations.
- Exactly collapsed-D initialization: first attempt produced non-finite
  gradients, so it is retained as a failed artifact.
- Collapsed-D retry: with deterministic capped symmetry-breaking noise,
  completed all 8 iterations without the earlier non-finite-gradient failure.

These runs establish numerical execution behavior only. They do not establish
perceptual strength calibration or endpoint-relative reward quality.

## Reproducibility artifacts committed with this report

For the exact runs, the repository includes configuration, parity localization,
optimization trace, and summary JSON. Image grids and all heavyweight cached
artifacts remain outside Git.

## Next decision

Do not run 512x512 optimization by weakening the B=1/B=3 threshold. The
correct follow-up is a two-tier validation: preserve this cross-batch BF16
diagnostic, while adding strict same-batch zero-control parity as the gate for
the K-branch control path. Only after that gate is implemented and passes
should 512x512 VJP be retried.
