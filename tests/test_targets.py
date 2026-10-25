"""Tests for auxiliary targets and numerically safe loss primitives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import src.losses as losses
from src.losses import (
    balanced_bce_with_logits,
    ambiguity_target,
    boundary_target,
    build_multiscale_targets,
    compute_total_loss,
    compute_lite_sdf_distill_loss,
    compute_performance_context_loss,
    hierarchical_output_distillation_loss,
    make_multiscale_targets,
    masked_balanced_bce_with_logits,
    masked_soft_dice_loss,
    segmentation_loss,
    soft_dice_loss,
    truncated_signed_distance_target,
)


def _square_mask(size: int = 16) -> torch.Tensor:
    mask = torch.zeros(1, 1, size, size)
    mask[:, :, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
    return mask


def test_square_boundary_and_texture_ignore_band() -> None:
    mask = _square_mask()
    image = torch.zeros(1, 3, 16, 16)
    target = make_multiscale_targets(mask, image, [(16, 16)])[0]
    boundary = target["boundary"][0, 0]
    valid = target["texture_valid"][0, 0]

    assert boundary[3, 6] > 0
    assert boundary[4, 6] > 0
    assert boundary[1, 6] == 0
    assert boundary[6, 6] == 0
    assert boundary[12, 6] > 0
    assert torch.equal(valid, 1 - boundary)
    assert valid[3, 6] < 1 and valid[4, 6] < 1
    assert valid[0, 0] == 1 and valid[6, 6] == 1
    assert torch.count_nonzero(target["texture"]) == 0


def test_multiscale_targets_match_each_feature_shape() -> None:
    mask = _square_mask(32)
    image = torch.zeros(1, 3, 32, 32)
    sizes = [(32, 32), (16, 16), (8, 8)]
    per_scale = make_multiscale_targets(mask, image, sizes)
    grouped = build_multiscale_targets(mask, image, sizes)
    for index, size in enumerate(sizes):
        for key in ("mask", "boundary", "texture", "texture_valid"):
            assert per_scale[index][key].shape == (1, 1, *size)
        assert torch.equal(
            grouped["boundary_targets"][index], per_scale[index]["boundary"]
        )
        assert torch.all((0 <= per_scale[index]["boundary"]) & (per_scale[index]["boundary"] <= 1))
    assert torch.any(
        (per_scale[0]["boundary"] > 0) & (per_scale[0]["boundary"] < 1)
    )


def test_sobel_magnitude_matches_a_hand_checked_vertical_step() -> None:
    step = torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]]).expand(1, 1, 4, 4)
    actual = losses.sobel_magnitude(step)
    assert actual.shape == step.shape
    assert torch.all(actual[:, :, :, 1:3] > 0)
    assert torch.count_nonzero(actual[:, :, :, 0]) == 0
    assert torch.count_nonzero(actual[:, :, :, 3]) == 0


def test_edge_loss_has_finite_gradients_in_flat_regions() -> None:
    logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
    mask = torch.zeros_like(logits)
    mask[:, :, 8:24, 8:24] = 1
    value = losses.segmentation_edge_loss(logits, mask)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_balanced_bce_uses_detached_negative_positive_ratio() -> None:
    logits = torch.zeros(1, 1, 1, 4, requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    loss = balanced_bce_with_logits(logits, target)
    # One positive weighted by three plus three ordinary negatives, averaged
    # over four pixels: (3 + 3) * log(2) / 4.
    assert torch.allclose(loss, torch.tensor(1.5 * math.log(2)), atol=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_no_positive_balanced_bce_falls_back_to_ordinary_bce() -> None:
    logits = torch.tensor([[[[-2.0, 0.5], [1.0, -0.25]]]])
    target = torch.zeros_like(logits)
    actual = balanced_bce_with_logits(logits, target)
    expected = F.binary_cross_entropy_with_logits(logits, target)
    assert torch.equal(actual, expected)


def test_masked_losses_ignore_invalid_values() -> None:
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    valid = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    logits_a = torch.zeros_like(target)
    logits_b = logits_a.clone()
    logits_b[:, :, 1] = torch.tensor([100.0, -100.0])
    assert torch.allclose(
        masked_balanced_bce_with_logits(logits_a, target, valid),
        masked_balanced_bce_with_logits(logits_b, target, valid),
    )
    assert torch.allclose(
        masked_soft_dice_loss(logits_a, target, valid),
        masked_soft_dice_loss(logits_b, target, valid),
    )


def test_empty_valid_region_has_differentiable_exact_zero_losses() -> None:
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, logits.shape).float()
    valid = torch.zeros_like(target)
    loss = masked_balanced_bce_with_logits(
        logits, target, valid
    ) + masked_soft_dice_loss(logits, target, valid)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_all_zero_mask_targets_and_losses_are_finite() -> None:
    mask = torch.zeros(2, 1, 32, 32)
    image = torch.zeros(2, 3, 32, 32)
    for target in make_multiscale_targets(
        mask, image, [(32, 32), (16, 16), (8, 8)]
    ):
        assert torch.count_nonzero(target["boundary"]) == 0
        boundary_logits = torch.randn_like(target["boundary"])
        texture_logits = torch.randn_like(target["texture"])
        losses = (
            balanced_bce_with_logits(boundary_logits, target["boundary"]),
            masked_balanced_bce_with_logits(
                texture_logits, target["texture"], target["texture_valid"]
            ),
            masked_soft_dice_loss(
                texture_logits, target["texture"], target["texture_valid"]
            ),
        )
        assert all(torch.isfinite(loss) for loss in losses)


def test_fp16_full_resolution_reductions_are_promoted_to_fp32() -> None:
    """A 256x256 mask has too many pixels for an FP16 count reduction."""

    logits = torch.zeros(
        1, 1, 256, 256, dtype=torch.float16, requires_grad=True
    )
    target = torch.ones_like(logits)
    valid = torch.ones_like(logits)
    losses = (
        soft_dice_loss(logits, target),
        segmentation_loss(logits, target),
        masked_balanced_bce_with_logits(logits, target, valid),
        masked_soft_dice_loss(logits, target, valid),
    )
    assert all(loss.dtype == torch.float32 for loss in losses)
    assert all(torch.isfinite(loss) for loss in losses)
    assert losses[2].item() > 0
    baseline = compute_total_loss({"seg_logits": logits}, target)
    assert torch.isfinite(baseline["boundary_loss"])
    assert torch.isfinite(baseline["texture_loss"])
    sum(losses).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_soft_sobel_boundary_is_localized_and_scale_matched() -> None:
    """A missing or mis-scaled fixed Sobel supervisor must fail this test."""

    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 10:22] = 1

    target = losses.soft_boundary_target(mask, (16, 16))

    assert target.shape == (1, 1, 16, 16)
    assert target.dtype == mask.dtype
    assert 0 <= float(target.min()) <= float(target.max()) <= 1
    assert target[0, 0, 6, 4] > 0
    assert target[0, 0, 0, 0] == 0
    assert target[0, 0, 8, 8] == 0


def test_texture_energy_is_zero_for_a_spatially_constant_image() -> None:
    """A texture supervisor must not invent texture from constant colour."""

    images = torch.zeros(2, 3, 16, 16)
    masks = torch.ones(2, 1, 16, 16)

    texture, valid = losses.texture_energy_target(images, masks, (16, 16))

    assert texture.shape == valid.shape == (2, 1, 16, 16)
    assert torch.count_nonzero(texture) == 0
    assert torch.all(valid[:, :, 2:-2, 2:-2] == 1)


def test_texture_energy_responds_to_lesion_stripes_and_masks_background() -> None:
    """Removing fixed local variance/Laplacian texture energy must fail."""

    images = torch.zeros(1, 3, 16, 16)
    masks = torch.zeros(1, 1, 16, 16)
    masks[:, :, 3:13, 3:13] = 1
    images[:, :, 3:13, 3:13:2] = 2

    texture, valid = losses.texture_energy_target(images, masks, (16, 16))

    assert float(texture[:, :, 5:11, 5:11].mean()) > 0.05
    assert torch.count_nonzero(texture * (1 - masks)) == 0
    assert valid[0, 0, 3, 8] < 1
    assert valid[0, 0, 8, 8] == 1


def test_multiscale_targets_expose_boundary_texture_and_validity() -> None:
    """Regressing to the old Region-mask target must fail this contract."""

    mask = _square_mask(32)
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 8:24, 8:24:2] = 1

    targets = losses.make_multiscale_targets(mask, image, [(32, 32), (16, 16)])

    assert len(targets) == 2
    assert set(targets[0]) == {"mask", "boundary", "texture", "texture_valid"}
    assert targets[1]["boundary"].shape == (1, 1, 16, 16)
    assert targets[1]["texture"].shape == (1, 1, 16, 16)


def test_auxiliary_contribution_is_capped_relative_to_segmentation() -> None:
    """An unbounded auxiliary objective must not dominate segmentation again."""

    mask = _square_mask(16)
    image = torch.zeros(1, 3, 16, 16)
    image[:, :, 4:12, 4:12:2] = 2
    outputs = {
        "seg_logits": torch.zeros_like(mask, requires_grad=True),
        "boundary_logits": [
            torch.zeros(1, 1, size, size, requires_grad=True)
            for size in (16, 8, 4)
        ],
        "texture_logits": [
            torch.zeros(1, 1, size, size, requires_grad=True)
            for size in (16, 8, 4)
        ],
    }

    result = losses.compute_total_loss(
        outputs,
        mask,
        image,
        lambda_boundary=10.0,
        lambda_texture=10.0,
        max_aux_to_seg_ratio=0.5,
    )

    seg = result["seg_loss"]
    weighted_aux = result["weighted_aux_loss"]
    assert isinstance(seg, torch.Tensor) and isinstance(weighted_aux, torch.Tensor)
    assert float(weighted_aux.detach()) <= 0.5 * float(seg.detach()) + 1e-6
    assert torch.allclose(result["total_loss"], seg + weighted_aux)
    assert float(result["aux_weight_scale"].detach()) < 1
    result["total_loss"].backward()
    assert torch.isfinite(outputs["seg_logits"].grad).all()


def test_signed_distance_target_has_expected_sign_and_finite_lite_loss() -> None:
    mask = _square_mask(32)
    target, foreground = truncated_signed_distance_target(mask, (16, 16), radius=4)
    assert target.shape == foreground.shape == (1, 1, 16, 16)
    assert target[0, 0, 8, 8] > 0
    assert target[0, 0, 0, 0] < 0
    assert abs(float(target[0, 0, 4, 8])) < 1e-6

    images = torch.randn(1, 3, 32, 32)
    from src.models import LiteSDFDistillUNet

    model = LiteSDFDistillUNet(base_channels=20)
    prediction = model(images, return_aux=True)
    teacher = torch.randn_like(prediction["seg_logits"])
    result = compute_lite_sdf_distill_loss(
        prediction,
        mask,
        images,
        teacher_logits=teacher,
        auxiliary_outer_scale=0.25,
    )
    assert torch.isfinite(result["total_loss"])
    assert float(result["weighted_aux_loss"].detach()) <= (
        0.25 * 0.35 * float(result["seg_loss"].detach()) + 1e-6
    )
    result["total_loss"].backward()


def test_performance_context_loss_supervises_decoder_scales_safely() -> None:
    from src.models import PerformanceContextUNet

    images = torch.randn(1, 3, 32, 32)
    mask = _square_mask(32)
    model = PerformanceContextUNet(base_channels=8)
    prediction = model(images, return_aux=True)
    result = compute_performance_context_loss(
        prediction,
        mask,
        images,
        auxiliary_outer_scale=0.5,
    )

    assert len(result["deep_supervision_scale_losses"]) == 4
    assert torch.isfinite(result["total_loss"])
    assert float(result["weighted_deep_supervision_loss"].detach()) <= (
        0.5 * 0.25 * float(result["seg_loss"].detach()) + 1e-6
    )
    result["total_loss"].backward()
    gradients = [parameter.grad for parameter in model.deep_supervision.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_hierarchical_distillation_transfers_all_segmentation_scales() -> None:
    from src.models import CompressedContextUNet, PerformanceContextUNet

    images = torch.randn(1, 3, 32, 32)
    student = CompressedContextUNet(base_channels=8)
    teacher = PerformanceContextUNet(base_channels=12).eval()
    student_outputs = student(images, return_aux=True)
    with torch.no_grad():
        teacher_outputs = teacher(images, return_aux=True)
    distill, scale_losses = hierarchical_output_distillation_loss(
        student_outputs, teacher_outputs
    )
    assert len(scale_losses) == 5
    assert torch.isfinite(distill) and float(distill.detach()) > 0
    distill.backward()
    assert student.backbone.segmentation_head.weight.grad is not None
    assert all(head.weight.grad is not None for head in student.deep_supervision)


def test_boundary_weighted_hierarchical_distillation_is_finite() -> None:
    from src.models import PicoLiteSelectiveFullPathDistillUNet, PerformanceContextUNet

    images = torch.randn(1, 3, 32, 32)
    mask = _square_mask(32)
    student = PicoLiteSelectiveFullPathDistillUNet()
    teacher = PerformanceContextUNet(base_channels=8).eval()
    student_outputs = student(images, return_aux=True)
    with torch.no_grad():
        teacher_outputs = teacher(images, return_aux=True)
    distill, scale_losses = hierarchical_output_distillation_loss(
        student_outputs,
        teacher_outputs,
        mask=mask,
        boundary_boost=1.5,
    )
    assert len(scale_losses) == 5
    assert torch.isfinite(distill) and float(distill.detach()) > 0
    distill.backward()
    assert student.backbone.segmentation_head.weight.grad is not None


def test_ambiguity_category_combines_boundary_uncertainty_and_teacher_error() -> None:
    from src.models import AmbiguityContextUNet, PerformanceContextUNet

    mask = _square_mask(32)
    reference = torch.full_like(mask, -8.0)
    target = ambiguity_target(mask, reference, (16, 16))
    assert target.shape == (1, 1, 16, 16)
    assert target[0, 0, 8, 8] == 1  # confident teacher false negative
    assert target[0, 0, 0, 0] < 0.1  # confident correct background

    images = torch.randn(1, 3, 32, 32)
    student = AmbiguityContextUNet(base_channels=8)
    teacher = PerformanceContextUNet(base_channels=12).eval()
    outputs = student(images, return_aux=True)
    with torch.no_grad():
        teacher_outputs = teacher(images, return_aux=True)
    result = compute_performance_context_loss(
        outputs,
        mask,
        images,
        teacher_outputs=teacher_outputs,
        lambda_distill=0.2,
    )
    assert float(result["ambiguity_loss"].detach()) > 0
    result["total_loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in student.ambiguity_auxiliary.parameters()
    )
