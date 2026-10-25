"""One-batch CPU forward/backward smoke tests."""

from __future__ import annotations

import pytest
import torch

from src.losses import compute_total_loss
from src.models import TriFeatureUNet, UNet


@pytest.mark.parametrize("model_class", [UNet, TriFeatureUNet])
def test_one_cpu_optimizer_step_has_finite_loss_and_gradients(model_class) -> None:
    torch.manual_seed(7)
    model = model_class(base_channels=4).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 32, 32)
    masks = torch.zeros(2, 1, 32, 32)
    masks[0, :, 7:25, 8:24] = 1
    masks[1, :, 5:20, 11:27] = 1

    outputs = model(images, return_aux=isinstance(model, TriFeatureUNet))
    losses = compute_total_loss(outputs, masks, images)
    total = losses["total_loss"]
    assert torch.isfinite(total)

    before = model.backbone.segmentation_head.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    parameters_with_grad = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    assert parameters_with_grad
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters_with_grad)

    if isinstance(model, TriFeatureUNet):
        for auxiliary in (model.boundary_auxiliary, model.texture_auxiliary):
            gradients = [parameter.grad for parameter in auxiliary.parameters()]
            assert gradients and all(gradient is not None for gradient in gradients)
            assert all(torch.isfinite(gradient).all() for gradient in gradients)
            assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0

    optimizer.step()
    after = model.backbone.segmentation_head.weight.detach()
    assert not torch.equal(before, after)
