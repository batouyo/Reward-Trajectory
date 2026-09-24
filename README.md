# RewardFlow VeloEdit Calibration

Clean standalone project for the validated activation-range experiment.

The project has four independent layers:

- `rollout/`: deterministic VeloEdit-compatible FLUX-Kontext sampling.
- `calibration/`: source-image-referenced DreamSim valid-range detection.
- `metrics/`: DreamSim and LPIPS trajectory measurements.
- `scripts/`: reproducible experiment entry points.

The coarse probe uses one batch with four branches:

```text
alpha = [0.25, 0.50, 0.75, 1.00]
```

For each generated image, DreamSim is computed against the resized source
image. `alpha_start` is the first alpha whose distance is greater than the
configured threshold. Around the first inactive/active adjacent pair, the
reusable branch-refinement module inserts midpoint branches recursively twice
by default. The active endpoint after those two evaluations is the reported
`alpha_start`.

The generic midpoint operation is exposed as
`rewardflow_calibration.calibration.refine_activation_bracket`; it accepts an
`alpha -> distance` evaluator and is independent of VeloEdit and DreamSim.

After `alpha_start` is detected, the elastic-band stage initializes control
points by mapping `beta=[0, .25, .5, .75, 1]` to `[alpha_start, 1]`.  It then
uses the independent `elastic_band_search` module to repeatedly expand large
DreamSim gaps with midpoint branches and move interior points toward more
uniform perceptual spacing.  The endpoints remain fixed.  The final control
points and search history are stored under `elastic_band` in the result JSON.

## Environment

Use the existing `group-edit` environment. The project does not vendor the
large `diffusers` source tree or copy experiment outputs.

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$PWD/src
```

## Sample experiment

```bash
python scripts/run_activation_detection.py \
  --source /home/hyp/Code/VeloEdit/testdata/7.jpg \
  --prompt "Make him old." \
  --seed 42 \
  --steps 15 \
  --model /data15/hyp/weight/FLUX.1-Kontext-dev \
  --output-dir outputs/image_7
```

The validated sample produced `alpha_start=0.5000` with DreamSim distances
approximately `0.00047, 0.01609, 0.16506, 0.19352` at the four coarse alphas.

## Fixed-alpha `V_goal` correction

The optional `optimization/` package performs per-image, per-alpha test-time
optimization of a zero-initialized velocity residual. Alpha and all FLUX
parameters are frozen. `V_goal` is added after VeloEdit's existing velocity
intervention during the first `--goal-steps` transitions; the terminal
objective is backpropagated through the frozen remaining rollout. Activation
checkpointing is used on transformer and VAE forwards.

The edit term keeps the differentiable SigLIP score above the uncorrected
high-alpha score minus a tolerance. Preservation combines a differentiable
DreamSim distance to the source with dense DINOv2 patch-feature drift; the
DreamSim model is frozen, but its tensor preprocessing keeps gradients flowing
to the generated image. InsightFace identity similarity is reported as an
evaluation metric only: its current ONNX/NumPy path is not differentiable. The
current calibration checkout has no differentiable VQA implementation, so
this experiment uses the installed SigLIP reward.

Run the fixed-alpha comparison on the previously used face-aging sample:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=$PWD/src \
  /home/hyp/.conda/envs/group-edit/bin/python scripts/run_goal_residual_optimization.py \
  --source /home/hyp/Code/VeloEdit/testdata/7.jpg \
  --prompt "make him old" --target-prompt "an old man" \
  --alpha 0.843 --seed 42 --steps 15 --goal-steps 4 \
  --model /data15/hyp/weight/FLUX.1-Kontext-dev \
  --output-dir outputs/image_7_goal_residual
```

The script saves `baseline_vgoal_zero.png`, `optimized_vgoal.png`, the
optimized sample-specific `v_goal.pt`, and `goal_residual_result.json` with
edit score, DreamSim drift, DINOv2 drift, optional InsightFace similarity,
and per-iteration losses. Iterations are streamed to
`optimization_progress.json`. `--max-area` controls rollout resolution for
memory/runtime-constrained smoke tests (default: 1,048,576 pixels), and
`--structure-weight` controls the DINOv2 term inside preservation. Lower
DreamSim/DINOv2 drift with retained SigLIP/face scores is the target.
