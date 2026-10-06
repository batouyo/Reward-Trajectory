import torch

from rewardflow_calibration.rollout.latent_accumulation import euler_latent_update


def test_default_latent_update_matches_legacy_cast_back_behavior():
    z = torch.tensor([1.0], dtype=torch.bfloat16)
    velocity = torch.tensor([0.001], dtype=torch.bfloat16)
    expected = (z.float() + torch.tensor(1.0) * velocity.float()).to(z.dtype)
    result = euler_latent_update(z, velocity, 1.0, accumulate_fp32=False)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_fp32_latent_accumulation_keeps_small_updates_between_steps():
    z_legacy = torch.tensor([1.0], dtype=torch.bfloat16)
    z_fp32 = z_legacy.clone()
    velocity = torch.tensor([0.001], dtype=torch.bfloat16)
    for _ in range(2):
        z_legacy = euler_latent_update(z_legacy, velocity, 1.0, accumulate_fp32=False)
        z_fp32 = euler_latent_update(z_fp32, velocity, 1.0, accumulate_fp32=True)
    assert z_fp32.dtype == torch.float32
    assert z_legacy.item() == 1.0
    assert z_fp32.item() > 1.001
