# FLUX.1-Kontext text suppression v2: one-case diagnostic

This is a single-pair, generator-native text-space diagnostic on the formal black-to-blue ball case. It is neither a paper-faithful contrastive Difference-of-Means reproduction nor a semantic-strength calibration. All scans use the same fixed model, source, prompt, seed `20260914`, 256² resolution, 12 denoising steps, guidance `2.5`, BF16, and B=1 sequential unrolls from a single captured state **within each run**. The capture is repeated between runs with the same settings; exact native/zero-steering parity is checked each time. No reward, velocity controller, mask, elastic band, SDE, or trained parameter is used.

| Method | Direction raw / normalized norm | Coarse largest adjacent image MAD | Fine largest adjacent image MAD* | Visually concentrated transition | Distinct intermediate ball colors† | Background / structure drift† |
| --- | --- | ---: | ---: | --- | --- | --- |
| A: declarative T5 | `4.51837 / 1.00000` | `0.04711`, `[-2.75,-2.50]` | `0.02684`, `[-2.65,-2.60]` | black at `-2.65`, dark blue at `-2.60`, brighter blue at `-2.55` (width ≈ `0.10` alpha) | about one obvious dark-blue middle state | Some background and ball appearance drift; no complete collapse |
| B: matched-instruction T5 | `5.56168 / 1.00000` | `0.05482`, `r=[0.4,0.5]` | `0.03084`, `r=[0.42,0.44]` | blue at `r=.42`, dark blue around `.44`, black by `.46` (width ≈ `.04` r, or `.22` T5 alpha) | about one obvious dark-blue middle state | Background/ball appearance drift persists |
| C control: matched T5-only, same capture as joint | `5.56168 / 1.00000` | `0.05482` | `0.03084` | same as B; B and C-control report metrics agree exactly | about one | persists |
| C: matched T5 + CLIP pooled | T5 `5.56168 / 1.00000`; CLIP `16.31425 / 1.00000` | `0.03614`, `r=[0.3,0.4]` | `0.03061`, `r=[0.40,0.42]` | blue at `r=.38`, dark blue near `.40`, black by `.42` (width ≈ `.04` r) | about one obvious dark-blue middle state | persists; no clear structural gain |

* Fine maximum excludes the artificial jump from the `r=0` Native Full anchor to the start of a local scan. Pixel MAD is a diagnostic distance, not semantic strength. † Assistant visual inspection of the saved grids; not an independent human evaluation or objective semantic metric.

The zero-steering path is exact in every run: captured native versus native re-unroll, native versus T5-only zero, and (for C) native versus joint zero all have latent and decoded-image maximum and mean absolute differences of `0`.

Source/Full pixel-distance trends are not globally monotonic. A moves away from Source before the narrow switch and then approaches it; B and C behave similarly around their shifted thresholds. Distance to Full rises around the switch and then plateaus or reverses at stronger suppression. These trends must not be interpreted as image-editing percentages.

## Answers to the three questions

1. **Dense A:** A narrow transition exists, but it remains threshold-like at practical sampling scales. The largest jump persists inside a single `.05` alpha interval, with only about one visible dark-blue intermediate state. Label: `THRESHOLD_LIKE_TRANSITION`, not a validated smooth slider.
2. **Matched B:** Matching declarative and imperative contexts does not materially improve smoothness. It shifts/broadens the narrow region slightly in alpha units, but retains a large jump and background/appearance drift. The hypothesis that context mismatch was the main cause is not supported by this case.
3. **Joint C:** At coarse spacing, joint steering appears to reduce the maximum jump; equal-resolution local scans show almost the same maximum as T5-only (`.03061` versus `.03084`) and about the same visible transition width. Joint CLIP mainly shifts the switch earlier. The hypothesis that fixed pooled CLIP conditioning was the main cause is not supported by this case.

**Conservative conclusion:** Negative text steering is responsive, but remains switch-like on this one FLUX-Kontext example; these data do not justify using it alone as a primary continuous-strength controller. A next study should test whether the same threshold behavior recurs across several instructions and seeds, without changing the method during this diagnostic.

Artifacts: `kontext_text_suppression_dense_v2/` (coarse + `fine/`), `kontext_text_suppression_matched_v2/` (coarse + `fine/`), and `kontext_text_suppression_joint_v2/` (T5-only/joint paired grids and `fine/`). Each directory contains machine-readable `report.json` and original PNG outputs.
