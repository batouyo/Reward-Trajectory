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
