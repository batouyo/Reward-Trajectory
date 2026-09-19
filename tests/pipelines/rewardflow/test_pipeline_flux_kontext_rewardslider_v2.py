import torch

from diffusers.pipelines.rewardflow.pipeline_flux_kontext_rewardslider_v2 import (
    predict_kontext_velocity_branchwise,
)


def test_branchwise_prediction_calls_model_with_b_one_and_preserves_gradients():
    latent = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]], requires_grad=True)
    batch_sizes = []

    def predict(batch, **kwargs):
        del kwargs
        batch_sizes.append(batch.shape[0])
        # Deliberately include batch-size-dependent behavior: the production
        # path must avoid this B=K numerical path by using B=1 calls.
        return batch + float(batch.shape[0])

    actual = predict_kontext_velocity_branchwise(predict, latent, scale=torch.ones(3, 1, 1))

    torch.testing.assert_close(actual, latent + 1.0)
    assert batch_sizes == [1, 1, 1]
    actual.sum().backward()
    torch.testing.assert_close(latent.grad, torch.ones_like(latent))


def test_branchwise_prediction_slices_only_matching_batch_kwargs():
    latent = torch.zeros(2, 1, 1)
    seen = []

    def predict(batch, **kwargs):
        seen.append((batch.shape, kwargs["matching"].shape, kwargs["shared"].shape))
        return batch

    actual = predict_kontext_velocity_branchwise(
        predict, latent, matching=torch.ones(2, 3), shared=torch.ones(1, 3)
    )

    torch.testing.assert_close(actual, latent)
    assert seen == [((1, 1, 1), (1, 3), (1, 3)), ((1, 1, 1), (1, 3), (1, 3))]
