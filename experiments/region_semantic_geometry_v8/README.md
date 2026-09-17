# Region semantic geometry v8 — formal ball diagnostic

This run reused the five immutable v6 `ball_color_seed_20260914` PNGs, the same FP32 CLIP/SigLIP checkpoints and preprocessing, and `CachedEndpointEmbeddingGeometry`. No probe, controller, reward, or segmentation model was generated or optimized. FLUX-Kontext native versus zero-control latent parity was exact (max/mean error `0`).

The fixed-region conversion is a **diagnostic assumption**, not a paper-defined component: independently min-max normalize the first four existing velocity-discrepancy score maps, average, use the existing stable top-25% token selector, nearest-neighbor upscale, then pad the nonzero-pixel bounding box by 10% per side.

## Outcome

The 16×16 score heatmap has its strongest area on the foreground ball, so the signal does cover the edit target. But selected tokens also occur at the top and right. The raw nonzero bbox is `[16,0,256,256]`; after prescribed padding, the fixed crop is `[0,0,256,256]`—the **entire image**. The overview, heatmap, mask, overlay, bbox, and all five saved crops make this failure visible. The crop is not a region-only view, so this run cannot test whether background dilution caused v6's feature compression.

| Encoder | View | p(0.2), p(0.5), p(0.8) | Adjacent gaps | Span | Strict order | Mean probe off-axis |
| --- | --- | --- | --- | ---: | --- | ---: |
| CLIP | whole | `.84947, .88265, .89466` | `.03317, .01201` | `.04518` | yes | `.31056` |
| CLIP | crop | same, exactly | same, exactly | `.04518` | yes | `.31056` |
| SigLIP | whole | `.75996, .82779, .92330` | `.06783, .09551` | `.16333` | yes | `.35405` |
| SigLIP | crop | same, exactly | same, exactly | `.16333` | yes | `.35405` |

The whole-image fields exactly reproduce the stored v6 formal-ball CSV (`max absolute field difference = 0`). Equal crop scores are tautological because the crop is the full image; they are **not evidence against the dilution hypothesis**. Neither encoder has a larger intermediate span or more stable ordering under this ineffective crop.

The 0.5-probe progress gradient exists and is finite/nonzero: full-image RMS `0.01888` (CLIP) and `0.02098` (SigLIP). Because the bbox is the full image, the reported inside-crop fraction of `1` and outside value of `0` provide no spatial-localization evidence. `gradient_audit.json` marks that check uninformative rather than a pass.

**Answers:** A. The hot area includes the ball, but the selected region is not sufficiently localized to crop it. B/C. CLIP and SigLIP whole/crop gaps and spans are identical, so improvement cannot be evaluated. D. Both were strictly ordered before and remain so only because the inputs are identical. E. Both gradients are finite and nonzero; crop-specific localization is untested.

Next decision, outside this run: choose and preregister a rule that isolates a connected high-score component or otherwise prevents remote selected tokens from expanding the bbox, then rerun this same geometry comparison. Do not relabel this full-image result as a successful region experiment or interpret progress as a semantic percentage.
