# Masked Reward-Guided Backward Trajectory MVP

This research MVP tests the hypothesis that a reliable native full-edit image can anchor a backward search toward the source. It applies output-level reward gradients only to the first four high-plasticity velocity steps, using the fourth-step clean-prediction proxy already implemented by ollout_native(..., early_stop_steps=4).

The reward combines source attraction, an instance-relative semantic survival hinge, and keep-region preservation. SigLIP and DINOv2 weights are frozen while gradients remain enabled from candidate pixels to the residual. DreamSim is a non-differentiable measurement and acceptance metric only.

The trajectory gate separately checks semantic monotonicity and floor, visible DreamSim change, sourceward perceptual movement, second-order continuity, keep-region drift, and the total residual trust region. It evaluates the full configured line-search set and accepts the valid candidate with the greatest sourceward movement, breaking ties by the smaller increment.

VeloEdit contributes only a frozen edit-region mask from the native trajectory. Its velocity edit direction is diagnostic only and never sets the correction direction. The reward gradient sets that direction; a single global RMS scaling preserves timestep-relative gradient weights.

Accepted states are recorded as NativeFull, followed by Accepted_01, Accepted_02, and so on. Iteration numbers are not slider strengths. DreamSim does not define slider strength. A future slider can be constructed by cumulative perceptual arc-length reparameterization; this MVP does not implement that interpolation.

This is a research MVP, not the final paper algorithm. All thresholds and line-search ratios are engineering hyperparameters that require validation in subsequent experiments.
