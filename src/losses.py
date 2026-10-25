"""分割损失及多尺度边界/区域目标。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
import torch.nn.functional as F
from torch import Tensor


SpatialSize = tuple[int, int] | torch.Size | Tensor

DATASET_MEAN = 0.5
DATASET_STD = 0.5


def _matched(logits: Tensor, value: Tensor, name: str) -> Tensor:
    if logits.shape != value.shape:
        raise ValueError(
            f"logits and {name} must have identical shapes, got "
            f"{tuple(logits.shape)} and {tuple(value.shape)}"
        )
    return value.to(device=logits.device, dtype=logits.dtype)


def _for_loss(tensor: Tensor) -> Tensor:
    """用 FP32 执行低精度归约，避免大图计数溢出。"""

    return tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor


def _loss_inputs(logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    return _for_loss(logits), _for_loss(_matched(logits, target, "target"))


def _dice_loss(
    logits: Tensor,
    target: Tensor,
    epsilon: float,
    valid_mask: Tensor | None = None,
) -> Tensor:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    if valid_mask is None:
        intersection = (probabilities * target).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)
        return (1 - (2 * intersection + epsilon) / (denominator + epsilon)).mean()

    valid_count = valid_mask.sum(dim=dims)
    intersection = (probabilities * target * valid_mask).sum(dim=dims)
    denominator = ((probabilities + target) * valid_mask).sum(dim=dims)
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    has_valid = (valid_count > 0).to(dtype=logits.dtype)
    return ((1 - dice) * has_valid).sum() / has_valid.sum().clamp_min(1)


def _balanced_bce(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
    min_pos_weight: float,
    max_pos_weight: float,
) -> Tensor:
    if min_pos_weight <= 0 or max_pos_weight < min_pos_weight:
        raise ValueError("invalid positive-weight bounds")
    with torch.no_grad():
        if valid_mask is None:
            positives = target.sum()
            negatives = target.numel() - positives
        else:
            positives = (target * valid_mask).sum()
            negatives = ((1 - target) * valid_mask).sum()
        pos_weight = (negatives / positives.clamp_min(1)).clamp(
            min=min_pos_weight, max=max_pos_weight
        )

    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="mean" if valid_mask is None else "none",
    )
    if valid_mask is None:
        return loss
    return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1)


def soft_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    """逐样本 Soft Dice 的批次均值。"""

    logits, target = _loss_inputs(logits, target)
    return _dice_loss(logits, target, epsilon)


def balanced_bce_with_logits(
    logits: Tensor,
    target: Tensor,
    *,
    min_pos_weight: float = 1.0,
    max_pos_weight: float = 20.0,
) -> Tensor:
    """使用批次级 ``clamp(负样本/正样本, 1, 20)`` 权重的 BCE。"""

    logits, target = _loss_inputs(logits, target)
    return _balanced_bce(logits, target, None, min_pos_weight, max_pos_weight)


def masked_balanced_bce_with_logits(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    min_pos_weight: float = 1.0,
    max_pos_weight: float = 20.0,
) -> Tensor:
    """仅统计有效像素的类别平衡 BCE；空有效区返回可反传零。"""

    valid_mask = _for_loss(_matched(logits, valid_mask, "valid_mask").clamp(0, 1))
    logits, target = _loss_inputs(logits, target)
    return _balanced_bce(
        logits, target, valid_mask, min_pos_weight, max_pos_weight
    )


def masked_soft_dice_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """忽略无效像素，并排除空有效区样本的 Soft Dice。"""

    valid_mask = _for_loss(_matched(logits, valid_mask, "valid_mask").clamp(0, 1))
    logits, target = _loss_inputs(logits, target)
    return _dice_loss(logits, target, epsilon, valid_mask)


def _spatial_size(size: SpatialSize) -> tuple[int, int]:
    if isinstance(size, Tensor):
        if size.ndim < 2:
            raise ValueError("a reference tensor must have at least two dimensions")
        height, width = size.shape[-2:]
    else:
        if len(size) != 2:
            raise ValueError(f"spatial size must have two values, got {tuple(size)}")
        height, width = size
    result = int(height), int(width)
    if min(result) <= 0:
        raise ValueError(f"spatial dimensions must be positive, got {result}")
    return result


def _binary_mask(mask: Tensor, name: str) -> Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"{name} must be shaped [B, 1, H, W], got {tuple(mask.shape)}")
    dtype = mask.dtype if mask.is_floating_point() else torch.float32
    return (mask > 0.5).to(dtype=dtype)


def resize_binary_mask(mask: Tensor, size: SpatialSize) -> Tensor:
    """用最近邻插值把二值掩码缩放到指定空间尺寸。"""

    return F.interpolate(_binary_mask(mask, "mask"), size=_spatial_size(size), mode="nearest")


def sobel_magnitude(tensor: Tensor) -> Tensor:
    """Return a normalized fixed-Sobel magnitude for a single-channel map."""

    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(
            f"tensor must be shaped [B, 1, H, W], got {tuple(tensor.shape)}"
        )
    value = _for_loss(tensor)
    kernel_x = value.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).view(1, 1, 3, 3) / 4.0
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
    gradient_x = F.conv2d(padded, kernel_x)
    gradient_y = F.conv2d(padded, kernel_y)
    magnitude_squared = gradient_x.square() + gradient_y.square()
    # ``sqrt`` has an infinite derivative at exactly zero.  Fixed targets do
    # not expose that issue, but the same operator is also used on predicted
    # probabilities by the parameter-free edge loss.  Clamp before sqrt and
    # explicitly restore exact zeros so flat regions have a finite zero gradient.
    epsilon = 16 * torch.finfo(magnitude_squared.dtype).eps
    magnitude = torch.sqrt(magnitude_squared.clamp_min(epsilon))
    magnitude = torch.where(
        magnitude_squared > epsilon, magnitude, torch.zeros_like(magnitude)
    )
    return magnitude.clamp(0, 1)


def soft_boundary_target(mask: Tensor, size: SpatialSize) -> Tensor:
    """Generate a scale-matched soft boundary from a ground-truth mask."""

    scaled = F.interpolate(
        _binary_mask(mask, "mask"),
        size=_spatial_size(size),
        mode="bilinear",
        align_corners=False,
    )
    return F.max_pool2d(sobel_magnitude(scaled), 3, stride=1, padding=1).clamp(0, 1)


def distance_boundary_target(
    mask: Tensor,
    size: SpatialSize,
    *,
    radius: int = 3,
    decay: float = 1.25,
) -> Tensor:
    """生成截断的软距离边界场。

    先提取轮廓，再逐像素膨胀并按指数衰减。与单像素边界相比，该目标能在
    预测尚未贴近真实轮廓时继续提供非零梯度，而且不依赖额外科学计算库。
    """

    if radius < 0 or decay <= 0:
        raise ValueError("radius must be non-negative and decay must be positive")
    edge = soft_boundary_target(mask, size)
    target = edge
    for distance in range(1, radius + 1):
        spread = F.max_pool2d(edge, 2 * distance + 1, stride=1, padding=distance)
        target = torch.maximum(target, math.exp(-distance / decay) * spread)
    return target.clamp(0, 1)


def ambiguity_target(
    mask: Tensor,
    reference_logits: Tensor,
    size: SpatialSize,
    *,
    error_radius: int = 2,
) -> Tensor:
    """Soft ambiguous-class target from boundary, uncertainty, and errors.

    ``reference_logits`` normally comes from the fixed wider teacher.  Using a
    fixed reference prevents the ambiguity category from chasing the student's
    rapidly changing early predictions.
    """

    if error_radius < 0:
        raise ValueError("error_radius must be non-negative")
    spatial_size = _spatial_size(size)
    scaled_mask = resize_binary_mask(mask, spatial_size)
    reference = F.interpolate(
        _for_loss(reference_logits.detach()),
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    )
    probability = reference.sigmoid().clamp(1e-6, 1 - 1e-6)
    entropy = -(
        probability * probability.log()
        + (1 - probability) * (1 - probability).log()
    ) / math.log(2.0)
    # Only the high-entropy half is considered genuinely ambiguous; this avoids
    # labelling large confident interiors as a third class.
    uncertainty = ((entropy - 0.5) / 0.5).clamp(0, 1)
    errors = ((probability >= 0.5) != (scaled_mask >= 0.5)).float()
    if error_radius:
        errors = F.max_pool2d(
            errors,
            2 * error_radius + 1,
            stride=1,
            padding=error_radius,
        )
    boundary = distance_boundary_target(mask, spatial_size, radius=2)
    return torch.maximum(boundary, torch.maximum(errors, uncertainty)).detach()


def _dataset_luminance(images: Tensor) -> Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"images must be shaped [B, 3, H, W], got {tuple(images.shape)}"
        )
    value = _for_loss(images)
    # ISICDataset maps RGB from [0, 1] to [-1, 1], not ImageNet z-scores.
    rgb = value * DATASET_STD + DATASET_MEAN
    weights = value.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def texture_energy_target(
    images: Tensor,
    mask: Tensor,
    size: SpatialSize,
) -> tuple[Tensor, Tensor]:
    """Build a lesion texture-energy target with fixed classical operators."""

    if images.shape[0] != mask.shape[0] or images.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            "images and mask must share batch and spatial dimensions, got "
            f"{tuple(images.shape)} and {tuple(mask.shape)}"
        )
    spatial_size = _spatial_size(size)
    luminance = F.interpolate(
        _dataset_luminance(images),
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    )
    pad_mode = "reflect" if min(luminance.shape[-2:]) > 2 else "replicate"
    padded = F.pad(luminance, (2, 2, 2, 2), mode=pad_mode)
    local_mean = F.avg_pool2d(padded, 5, stride=1)
    local_square_mean = F.avg_pool2d(padded.square(), 5, stride=1)
    local_variance = (local_square_mean - local_mean.square()).clamp_min(0)
    local_variance = torch.where(
        local_variance > 16 * torch.finfo(local_variance.dtype).eps,
        local_variance,
        torch.zeros_like(local_variance),
    )
    local_std = local_variance.sqrt()

    laplacian_kernel = luminance.new_tensor(
        ((0.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 0.0))
    ).view(1, 1, 3, 3) / 4.0
    laplacian = F.conv2d(
        F.pad(luminance, (1, 1, 1, 1), mode="replicate"),
        laplacian_kernel,
    ).abs()
    energy = local_std + 0.5 * laplacian
    flattened = energy.flatten(1)
    scale = flattened.mean(dim=1) + 2.0 * flattened.std(dim=1, unbiased=False)
    normalized = (energy / scale.view(-1, 1, 1, 1).clamp_min(1e-6)).clamp(0, 1)

    scaled_mask = resize_binary_mask(mask, spatial_size)
    boundary = soft_boundary_target(mask, spatial_size)
    return normalized * scaled_mask, (1 - boundary).clamp(0, 1)


def wavelet_texture_target(
    images: Tensor,
    mask: Tensor,
    size: SpatialSize,
    *,
    boundary_radius: int = 3,
) -> tuple[Tensor, Tensor]:
    """用两个无参数高通残差构造连续多频带纹理能量目标。

    ``|Y-Avg3(Y)|`` 描述细纹理，``|Avg3(Y)-Avg5(Y)|`` 描述较粗纹理。
    目标只在病灶内部生效，并排除软边界带，减少边缘与纹理监督的冲突。
    """

    spatial_size = _spatial_size(size)
    luminance = F.interpolate(
        _dataset_luminance(images),
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    )
    low3 = F.avg_pool2d(F.pad(luminance, (1, 1, 1, 1), mode="reflect"), 3, 1)
    low5 = F.avg_pool2d(F.pad(luminance, (2, 2, 2, 2), mode="reflect"), 5, 1)
    energy = (luminance - low3).abs() + 0.5 * (low3 - low5).abs()
    flattened = energy.flatten(1)
    scale = flattened.mean(1) + 2.0 * flattened.std(1, unbiased=False)
    energy = (energy / scale.view(-1, 1, 1, 1).clamp_min(1e-6)).clamp(0, 1)
    scaled_mask = resize_binary_mask(mask, spatial_size)
    boundary = distance_boundary_target(mask, spatial_size, radius=boundary_radius)
    valid = scaled_mask * (1 - boundary).clamp(0, 1)
    return energy * scaled_mask, valid


def boundary_target(mask_at_scale: Tensor) -> Tensor:
    """用 ``3x3`` 形态学梯度生成当前尺度的边界带。"""

    mask = _binary_mask(mask_at_scale, "mask_at_scale")
    dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
    eroded = 1 - F.max_pool2d(1 - mask, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


def truncated_signed_distance_target(
    mask: Tensor,
    size: SpatialSize,
    *,
    radius: int = 6,
) -> tuple[Tensor, Tensor]:
    """Approximate a signed distance field using differentiable morphology.

    Values are positive inside the lesion, negative outside, and near zero at
    its contour.  Distances saturate at ``radius`` feature pixels.  The binary
    mask is returned as well so the regression loss can balance foreground and
    background instead of being dominated by the latter.
    """

    if radius <= 0:
        raise ValueError("signed-distance radius must be positive")
    scaled_mask = resize_binary_mask(mask, size)

    def erode(value: Tensor) -> Tensor:
        return 1 - F.max_pool2d(1 - value, 3, stride=1, padding=1)

    inside, outside = scaled_mask, 1 - scaled_mask
    inside_distance = torch.zeros_like(scaled_mask)
    outside_distance = torch.zeros_like(scaled_mask)
    for _ in range(radius):
        inside = erode(inside)
        outside = erode(outside)
        inside_distance = inside_distance + inside
        outside_distance = outside_distance + outside
    signed = (inside_distance - outside_distance) / float(radius)
    return signed.clamp(-1, 1), scaled_mask


def make_multiscale_targets(
    mask: Tensor,
    images: Tensor,
    spatial_sizes: Sequence[SpatialSize],
) -> list[dict[str, Tensor]]:
    """逐尺度生成掩码、边界、区域目标和区域有效区。"""

    targets = []
    for size in spatial_sizes:
        scaled_mask = resize_binary_mask(mask, size)
        boundary = soft_boundary_target(mask, size)
        texture, texture_valid = texture_energy_target(images, mask, size)
        targets.append(
            {
                "mask": scaled_mask,
                "boundary": boundary,
                "texture": texture,
                "texture_valid": texture_valid,
            }
        )
    return targets


def make_gcsas_targets(
    mask: Tensor,
    images: Tensor,
    spatial_sizes: Sequence[SpatialSize],
) -> list[dict[str, Tensor]]:
    """为 GCSAS 候选生成距离边界与多频带纹理目标。"""

    targets = []
    for size in spatial_sizes:
        spatial_size = _spatial_size(size)
        # Preserve an approximately constant boundary width in input-image pixels.
        # A fixed radius of three at E3 previously represented ~12 input pixels and
        # removed too much of the already-small valid lesion interior.
        relative_scale = max(spatial_size) / max(mask.shape[-2:])
        radius = max(1, round(3 * relative_scale))
        scaled_mask = resize_binary_mask(mask, size)
        boundary = distance_boundary_target(mask, size, radius=radius)
        texture, texture_valid = wavelet_texture_target(
            images, mask, size, boundary_radius=radius
        )
        targets.append(
            {
                "mask": scaled_mask,
                "boundary": boundary,
                "texture": texture,
                "texture_valid": texture_valid,
            }
        )
    return targets


def build_multiscale_targets(
    mask: Tensor,
    images: Tensor,
    spatial_sizes: Sequence[SpatialSize],
) -> dict[str, list[Tensor]]:
    """按字段组织多尺度目标。"""

    targets = make_multiscale_targets(mask, images, spatial_sizes)
    return {
        "mask_targets": [item["mask"] for item in targets],
        "boundary_targets": [item["boundary"] for item in targets],
        "texture_targets": [item["texture"] for item in targets],
        "texture_valid_masks": [item["texture_valid"] for item in targets],
    }


def segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    """全分辨率 ``BCEWithLogits + SoftDice``。"""

    logits, target = _loss_inputs(logits, target)
    return F.binary_cross_entropy_with_logits(logits, target) + _dice_loss(
        logits, target, 1e-6
    )


def deep_segmentation_loss(
    logits: Sequence[Tensor],
    target: Tensor,
    *,
    weights: Sequence[float] | None = None,
) -> tuple[Tensor, list[Tensor]]:
    """Apply the main segmentation objective to coarse-to-fine decoder outputs."""

    values = list(logits)
    if not values:
        raise ValueError("deep supervision requires at least one logit tensor")
    if weights is None:
        coefficients = target.new_ones(len(values))
    else:
        if len(weights) != len(values) or any(weight < 0 for weight in weights):
            raise ValueError(
                "deep-supervision weights must match logits and be non-negative"
            )
        coefficients = target.new_tensor(weights)
    if float(coefficients.sum()) <= 0:
        raise ValueError("at least one deep-supervision weight must be positive")
    coefficients = coefficients / coefficients.sum()
    losses = [
        segmentation_loss(logit, resize_binary_mask(target, logit))
        for logit in values
    ]
    return (torch.stack(losses) * coefficients).sum(), losses


def boundary_scale_loss(logits: Tensor, target: Tensor) -> Tensor:
    """单尺度边界损失。"""

    logits, target = _loss_inputs(logits, target)
    return _balanced_bce(logits, target, None, 1.0, 20.0) + _dice_loss(
        logits, target, 1e-6
    )


def texture_scale_loss(logits: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """忽略边界带的单尺度区域损失。"""

    valid_mask = _for_loss(_matched(logits, valid_mask, "valid_mask").clamp(0, 1))
    logits, target = _loss_inputs(logits, target)
    probability = torch.sigmoid(logits)
    regression = F.smooth_l1_loss(probability, target, reduction="none")
    regression = (regression * valid_mask).sum() / valid_mask.sum().clamp_min(1)
    return _balanced_bce(logits, target, valid_mask, 1.0, 20.0) + regression


def shape_scale_loss(logits: Tensor, target: Tensor) -> Tensor:
    """用密集区域监督形状/内部概念通道。"""

    return segmentation_loss(logits, target)


def signed_distance_scale_loss(
    logits: Tensor,
    target: Tensor,
    foreground: Tensor,
) -> Tensor:
    """Balanced Smooth-L1 regression of a normalized signed distance field."""

    logits, target = _loss_inputs(logits, target)
    foreground = _for_loss(_matched(logits, foreground, "foreground").clamp(0, 1))
    error = F.smooth_l1_loss(torch.tanh(logits), target, reduction="none")
    background = 1 - foreground
    inside = (error * foreground).sum() / foreground.sum().clamp_min(1)
    outside = (error * background).sum() / background.sum().clamp_min(1)
    return 0.5 * (inside + outside)


def segmentation_edge_loss(logits: Tensor, mask: Tensor) -> Tensor:
    """Match the final probability gradient to the ground-truth contour."""

    logits, mask = _loss_inputs(logits, mask)
    predicted = F.max_pool2d(
        sobel_magnitude(torch.sigmoid(logits)), 3, stride=1, padding=1
    )
    target = soft_boundary_target(mask, logits)
    return F.smooth_l1_loss(predicted, target)


def binary_kl_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 2.0,
    pixel_weights: Tensor | None = None,
) -> Tensor:
    """Temperature-scaled Bernoulli KL from a fixed teacher to the student."""

    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    student, teacher = _loss_inputs(student_logits, teacher_logits.detach())
    student = student / temperature
    teacher = teacher / temperature
    teacher_probability = torch.sigmoid(teacher)
    divergence = teacher_probability * (
        F.logsigmoid(teacher) - F.logsigmoid(student)
    ) + (1 - teacher_probability) * (
        F.logsigmoid(-teacher) - F.logsigmoid(-student)
    )
    if pixel_weights is None:
        reduced = divergence.mean()
    else:
        weights = _for_loss(_matched(divergence, pixel_weights, "pixel_weights"))
        if torch.any(weights < 0):
            raise ValueError("distillation pixel weights must be non-negative")
        reduced = (divergence * weights).sum() / weights.sum().clamp_min(1e-12)
    return reduced * temperature**2


def hierarchical_output_distillation_loss(
    student_outputs: Mapping[str, object],
    teacher_outputs: Mapping[str, object],
    *,
    temperature: float = 2.0,
    mask: Tensor | None = None,
    boundary_boost: float = 0.0,
) -> tuple[Tensor, list[Tensor]]:
    """Distill refined, coarse, and decoder-scale predictions from a teacher."""

    student_final = student_outputs.get("seg_logits")
    teacher_final = teacher_outputs.get("seg_logits")
    student_coarse = student_outputs.get("coarse_seg_logits")
    teacher_coarse = teacher_outputs.get("coarse_seg_logits")
    if not all(
        isinstance(value, Tensor)
        for value in (student_final, teacher_final, student_coarse, teacher_coarse)
    ):
        raise TypeError("hierarchical distillation requires final and coarse logits")
    student_deep = _tensor_list(student_outputs["deep_seg_logits"], "deep_seg_logits")
    teacher_deep = _tensor_list(teacher_outputs["deep_seg_logits"], "deep_seg_logits")
    if len(student_deep) != 3 or len(teacher_deep) != 3:
        raise ValueError("hierarchical distillation requires three decoder scales")

    if boundary_boost < 0:
        raise ValueError("boundary_boost must be non-negative")
    pairs = [
        (student_final, teacher_final),
        (student_coarse, teacher_coarse),
        *zip(student_deep, teacher_deep),
    ]
    losses: list[Tensor] = []
    for student, teacher in pairs:
        pixel_weights = None
        if mask is not None and boundary_boost > 0:
            boundary = soft_boundary_target(mask, student)
            pixel_weights = 1.0 + boundary_boost * boundary
        losses.append(
            binary_kl_distillation_loss(
                student,
                teacher,
                temperature=temperature,
                pixel_weights=pixel_weights,
            )
        )
    # The deployed refined output remains primary.  Coarse and increasingly
    # fine decoder predictions transfer how the wider teacher reaches it.
    weights = losses[0].new_tensor((0.50, 0.20, 0.05, 0.10, 0.15))
    return (torch.stack(losses) * weights).sum(), losses


def masked_correlation_loss(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """病灶内部的 ``1-Pearson correlation``，平坦目标不计入损失。"""

    prediction = _for_loss(prediction)
    target = _for_loss(_matched(prediction, target, "target"))
    valid = _for_loss(_matched(prediction, valid_mask, "valid_mask").clamp(0, 1))
    count = valid.sum(dim=(1, 2, 3)).clamp_min(1)
    pred_mean = (prediction * valid).sum(dim=(1, 2, 3)) / count
    target_mean = (target * valid).sum(dim=(1, 2, 3)) / count
    pred_centered = (prediction - pred_mean[:, None, None, None]) * valid
    target_centered = (target - target_mean[:, None, None, None]) * valid
    covariance = (pred_centered * target_centered).sum(dim=(1, 2, 3))
    pred_energy = pred_centered.square().sum(dim=(1, 2, 3))
    target_energy = target_centered.square().sum(dim=(1, 2, 3))
    # Clamp each energy before sqrt. Clamping the product after sqrt leaves an
    # infinite sqrt derivative at zero, which can turn a finite loss into NaN
    # gradients whenever an auxiliary head initially predicts a constant map.
    denominator = (
        pred_energy.clamp_min(epsilon) * target_energy.clamp_min(epsilon)
    ).sqrt()
    correlation = covariance / denominator
    usable = (target_energy > epsilon).to(prediction.dtype)
    return ((1 - correlation.clamp(-1, 1)) * usable).sum() / usable.sum().clamp_min(1)


def gcsas_texture_scale_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """连续纹理回归：掩码 Huber 加弱相关性约束。"""

    valid_mask = _for_loss(_matched(logits, valid_mask, "valid_mask").clamp(0, 1))
    logits, target = _loss_inputs(logits, target)
    probability = torch.sigmoid(logits)
    regression = F.smooth_l1_loss(probability, target, reduction="none")
    regression = (regression * valid_mask).sum() / valid_mask.sum().clamp_min(1)
    return regression + 0.2 * masked_correlation_loss(probability, target, valid_mask)


def _tensor_list(value: object, name: str) -> list[Tensor]:
    if isinstance(value, Tensor) or not isinstance(value, Sequence):
        raise TypeError(f"outputs[{name!r}] must be a sequence of tensors")
    tensors = list(value)
    if not tensors or any(not isinstance(item, Tensor) for item in tensors):
        raise TypeError(f"outputs[{name!r}] must be a non-empty sequence of tensors")
    return tensors


def compute_total_loss(
    outputs: Mapping[str, object],
    mask: Tensor,
    images: Tensor | None = None,
    lambda_boundary: float = 0.2,
    lambda_texture: float = 0.2,
    lambda_shape: float = 0.2,
    max_aux_to_seg_ratio: float = 0.5,
    target_mode: str = "legacy",
    scale_weights: Sequence[float] | None = None,
) -> dict[str, Tensor | list[Tensor]]:
    """计算基线或三特征模型的完整训练损失。"""

    if min(lambda_boundary, lambda_texture, lambda_shape) < 0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if max_aux_to_seg_ratio < 0:
        raise ValueError("max_aux_to_seg_ratio must be non-negative")
    seg_logits = outputs.get("seg_logits")
    if not isinstance(seg_logits, Tensor):
        raise TypeError("outputs['seg_logits'] must be a tensor")
    seg = segmentation_loss(seg_logits, mask)
    zero = _for_loss(seg_logits).sum() * 0.0

    has_boundary = "boundary_logits" in outputs
    has_texture = "texture_logits" in outputs
    if has_boundary != has_texture:
        raise ValueError("boundary_logits and texture_logits must be supplied together")
    if not has_boundary:
        return {
            "total_loss": seg,
            "seg_loss": seg,
            "boundary_loss": zero,
            "texture_loss": zero,
            "ambiguity_loss": zero,
            "weighted_ambiguity_loss": zero,
            "shape_loss": zero,
            "weighted_aux_loss": zero,
            "aux_weight_scale": zero,
            "boundary_scale_losses": [],
            "texture_scale_losses": [],
            "ambiguity_scale_losses": [],
            "shape_scale_losses": [],
            "sdf_loss": zero,
            "distill_loss": zero,
            "edge_loss": zero,
            "deep_supervision_loss": zero,
            "weighted_deep_supervision_loss": zero,
            "deep_supervision_scale_losses": [],
            "sdf_scale_losses": [],
        }
    if images is None:
        raise ValueError("images are required when auxiliary logits are present")

    boundary_logits = _tensor_list(outputs["boundary_logits"], "boundary_logits")
    texture_logits = _tensor_list(outputs["texture_logits"], "texture_logits")
    if len(boundary_logits) != len(texture_logits):
        raise ValueError("boundary and texture auxiliary outputs need equal scale counts")
    for boundary_logit, texture_logit in zip(boundary_logits, texture_logits):
        if boundary_logit.shape != texture_logit.shape:
            raise ValueError(
                "boundary and texture logits must match at each scale, got "
                f"{tuple(boundary_logit.shape)} and {tuple(texture_logit.shape)}"
            )

    if target_mode not in {"legacy", "gcsas"}:
        raise ValueError("target_mode must be 'legacy' or 'gcsas'")
    targets = (
        make_gcsas_targets(mask, images, boundary_logits)
        if target_mode == "gcsas"
        else make_multiscale_targets(mask, images, boundary_logits)
    )
    boundary_losses = [
        boundary_scale_loss(logit, target["boundary"])
        for logit, target in zip(boundary_logits, targets)
    ]
    texture_loss_function = (
        gcsas_texture_scale_loss if target_mode == "gcsas" else texture_scale_loss
    )
    texture_losses = [
        texture_loss_function(logit, target["texture"], target["texture_valid"])
        for logit, target in zip(texture_logits, targets)
    ]
    shape_logits = (
        _tensor_list(outputs["shape_logits"], "shape_logits")
        if "shape_logits" in outputs
        else []
    )
    if shape_logits and len(shape_logits) != len(boundary_logits):
        raise ValueError("shape auxiliary outputs must match the other scale counts")
    shape_losses = [
        shape_scale_loss(logit, target["mask"])
        for logit, target in zip(shape_logits, targets)
    ]
    if scale_weights is None:
        weights = seg.new_ones(len(boundary_losses))
    else:
        if len(scale_weights) != len(boundary_losses) or any(w < 0 for w in scale_weights):
            raise ValueError("scale_weights must match auxiliary scales and be non-negative")
        weights = seg.new_tensor(scale_weights)
    weights = weights / weights.sum().clamp_min(1e-12)
    boundary = (torch.stack(boundary_losses) * weights).sum()
    texture = (torch.stack(texture_losses) * weights).sum()
    shape = (
        (torch.stack(shape_losses) * weights).sum()
        if shape_losses
        else zero
    )
    raw_auxiliary = (
        lambda_boundary * boundary
        + lambda_texture * texture
        + lambda_shape * shape
    )
    with torch.no_grad():
        maximum = max_aux_to_seg_ratio * seg.detach().clamp_min(0)
        scale = torch.where(
            raw_auxiliary.detach() > 0,
            (maximum / raw_auxiliary.detach().clamp_min(1e-12)).clamp(max=1),
            torch.ones_like(raw_auxiliary.detach()),
        )
    weighted_auxiliary = scale * raw_auxiliary
    return {
        "total_loss": seg + weighted_auxiliary,
        "seg_loss": seg,
        "boundary_loss": boundary,
        "texture_loss": texture,
        "ambiguity_loss": zero,
        "weighted_ambiguity_loss": zero,
        "shape_loss": shape,
        "weighted_aux_loss": weighted_auxiliary,
        "aux_weight_scale": scale,
        "boundary_scale_losses": boundary_losses,
        "texture_scale_losses": texture_losses,
        "ambiguity_scale_losses": [],
        "shape_scale_losses": shape_losses,
        "sdf_loss": zero,
        "distill_loss": zero,
        "edge_loss": zero,
        "deep_supervision_loss": zero,
        "weighted_deep_supervision_loss": zero,
        "deep_supervision_scale_losses": [],
        "sdf_scale_losses": [],
    }


def compute_lite_sdf_distill_loss(
    outputs: Mapping[str, object],
    mask: Tensor,
    images: Tensor,
    *,
    teacher_logits: Tensor | None = None,
    lambda_boundary: float = 0.15,
    lambda_texture: float = 0.10,
    lambda_sdf: float = 0.15,
    lambda_distill: float = 0.20,
    lambda_edge: float = 0.05,
    max_aux_to_seg_ratio: float = 0.35,
    auxiliary_outer_scale: float = 1.0,
    distill_temperature: float = 2.0,
) -> dict[str, Tensor | list[Tensor]]:
    """Scale-aware, training-only supervision for the deployable base-20 model.

    Unlike the legacy objective, the schedule multiplies the already-capped
    auxiliary term.  Consequently a falling schedule always reduces the actual
    auxiliary gradient instead of being cancelled by the cap normalizer.
    """

    weights_to_check = (
        lambda_boundary,
        lambda_texture,
        lambda_sdf,
        lambda_distill,
        lambda_edge,
        max_aux_to_seg_ratio,
        auxiliary_outer_scale,
    )
    if min(weights_to_check) < 0 or auxiliary_outer_scale > 1:
        raise ValueError("lite loss weights and scale must be in valid non-negative ranges")

    seg_logits = outputs.get("seg_logits")
    if not isinstance(seg_logits, Tensor):
        raise TypeError("outputs['seg_logits'] must be a tensor")
    boundary_logits = _tensor_list(outputs["boundary_logits"], "boundary_logits")
    texture_logits = _tensor_list(outputs["texture_logits"], "texture_logits")
    sdf_logits = _tensor_list(outputs["sdf_logits"], "sdf_logits")
    scale_counts = (len(boundary_logits), len(texture_logits), len(sdf_logits))
    legacy_scales = scale_counts == (3, 2, 2)
    if not legacy_scales and min(scale_counts) <= 0:
        raise ValueError("semantic supervision requires every concept route to be non-empty")

    seg = segmentation_loss(seg_logits, mask)
    boundary_targets: dict[tuple[int, int], Tensor] = {}
    texture_targets: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
    sdf_targets: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

    def size_of(logit: Tensor) -> tuple[int, int]:
        return int(logit.shape[-2]), int(logit.shape[-1])

    boundary_losses: list[Tensor] = []
    for logit in boundary_logits:
        size = size_of(logit)
        if size not in boundary_targets:
            boundary_targets[size] = soft_boundary_target(mask, size)
        boundary_losses.append(boundary_scale_loss(logit, boundary_targets[size]))
    texture_losses: list[Tensor] = []
    for logit in texture_logits:
        size = size_of(logit)
        if size not in texture_targets:
            texture_targets[size] = texture_energy_target(images, mask, size)
        target, valid = texture_targets[size]
        texture_losses.append(texture_scale_loss(logit, target, valid))
    sdf_losses: list[Tensor] = []
    for logit in sdf_logits:
        size = size_of(logit)
        relative_scale = max(logit.shape[-2:]) / max(mask.shape[-2:])
        radius = max(2, round(12 * relative_scale))
        if size not in sdf_targets:
            sdf_targets[size] = truncated_signed_distance_target(
                mask, size, radius=radius
            )
        target, foreground = sdf_targets[size]
        sdf_losses.append(signed_distance_scale_loss(logit, target, foreground))

    def weighted(values: list[Tensor], coefficients: Sequence[float]) -> Tensor:
        coefficients_tensor = seg.new_tensor(coefficients)
        coefficients_tensor = coefficients_tensor / coefficients_tensor.sum()
        return (torch.stack(values) * coefficients_tensor).sum()

    if legacy_scales:
        boundary_coefficients: Sequence[float] = (0.5, 0.3, 0.2)
        texture_coefficients: Sequence[float] = (0.65, 0.35)
        sdf_coefficients: Sequence[float] = (0.4, 0.6)
    else:
        maximum_resolution = max(max(logit.shape[-2:]) for logit in boundary_logits)
        boundary_coefficients = tuple(
            math.sqrt(max(logit.shape[-2:]) / maximum_resolution)
            for logit in boundary_logits
        )
        texture_coefficients = tuple(
            (max(logit.shape[-2:]) / maximum_resolution) ** 0.25
            for logit in texture_logits
        )
        sdf_coefficients = (1.0,) * len(sdf_logits)
    boundary = weighted(boundary_losses, boundary_coefficients)
    texture = weighted(texture_losses, texture_coefficients)
    sdf = weighted(sdf_losses, sdf_coefficients)
    raw_auxiliary = (
        lambda_boundary * boundary
        + lambda_texture * texture
        + lambda_sdf * sdf
    )
    with torch.no_grad():
        maximum = max_aux_to_seg_ratio * seg.detach().clamp_min(0)
        cap_scale = torch.where(
            raw_auxiliary.detach() > 0,
            (maximum / raw_auxiliary.detach().clamp_min(1e-12)).clamp(max=1),
            torch.ones_like(raw_auxiliary.detach()),
        )
    weighted_auxiliary = auxiliary_outer_scale * cap_scale * raw_auxiliary
    edge = segmentation_edge_loss(seg_logits, mask)
    distill = (
        binary_kl_distillation_loss(
            seg_logits, teacher_logits, temperature=distill_temperature
        )
        if teacher_logits is not None
        else seg_logits.sum() * 0.0
    )
    zero = seg_logits.sum() * 0.0
    total = (
        seg
        + weighted_auxiliary
        + lambda_distill * distill
        + lambda_edge * edge
    )
    return {
        "total_loss": total,
        "seg_loss": seg,
        "boundary_loss": boundary,
        "texture_loss": texture,
        "ambiguity_loss": zero,
        "weighted_ambiguity_loss": zero,
        "shape_loss": zero,
        "sdf_loss": sdf,
        "distill_loss": distill,
        "edge_loss": edge,
        "deep_supervision_loss": zero,
        "weighted_deep_supervision_loss": zero,
        "deep_supervision_scale_losses": [],
        "weighted_aux_loss": weighted_auxiliary,
        "aux_weight_scale": cap_scale,
        "boundary_scale_losses": boundary_losses,
        "texture_scale_losses": texture_losses,
        "ambiguity_scale_losses": [],
        "shape_scale_losses": [],
        "sdf_scale_losses": sdf_losses,
    }


def compute_performance_context_loss(
    outputs: Mapping[str, object],
    mask: Tensor,
    images: Tensor,
    *,
    teacher_logits: Tensor | None = None,
    teacher_outputs: Mapping[str, object] | None = None,
    lambda_boundary: float = 0.15,
    lambda_texture: float = 0.05,
    lambda_sdf: float = 0.15,
    lambda_distill: float = 0.0,
    lambda_edge: float = 0.08,
    lambda_deep_supervision: float = 0.20,
    lambda_ambiguity: float = 0.10,
    max_aux_to_seg_ratio: float = 0.35,
    max_deep_to_seg_ratio: float = 0.25,
    max_ambiguity_to_seg_ratio: float = 0.10,
    auxiliary_outer_scale: float = 1.0,
    distill_temperature: float = 2.0,
    distill_boundary_boost: float = 0.0,
) -> dict[str, Tensor | list[Tensor]]:
    """Accuracy-first objective with concept and decoder deep supervision."""

    if min(
        lambda_deep_supervision,
        max_deep_to_seg_ratio,
        lambda_ambiguity,
        max_ambiguity_to_seg_ratio,
        distill_boundary_boost,
    ) < 0:
        raise ValueError("performance auxiliary weights and caps must be non-negative")
    hierarchical_teacher = teacher_outputs is not None
    result = compute_lite_sdf_distill_loss(
        outputs,
        mask,
        images,
        teacher_logits=None if hierarchical_teacher else teacher_logits,
        lambda_boundary=lambda_boundary,
        lambda_texture=lambda_texture,
        lambda_sdf=lambda_sdf,
        lambda_distill=0.0 if hierarchical_teacher else lambda_distill,
        lambda_edge=lambda_edge,
        max_aux_to_seg_ratio=max_aux_to_seg_ratio,
        auxiliary_outer_scale=auxiliary_outer_scale,
        distill_temperature=distill_temperature,
    )
    deep_logits = _tensor_list(outputs["deep_seg_logits"], "deep_seg_logits")
    coarse = outputs.get("coarse_seg_logits")
    if not isinstance(coarse, Tensor):
        raise TypeError("outputs['coarse_seg_logits'] must be a tensor")
    if len(deep_logits) != 3:
        raise ValueError("performance model requires D4, D3, and D2 supervision")
    deep, deep_losses = deep_segmentation_loss(
        [*deep_logits, coarse],
        mask,
        weights=(0.10, 0.15, 0.25, 0.50),
    )
    seg = result["seg_loss"]
    total = result["total_loss"]
    assert isinstance(seg, Tensor) and isinstance(total, Tensor)
    raw_deep = lambda_deep_supervision * deep
    with torch.no_grad():
        maximum = max_deep_to_seg_ratio * seg.detach().clamp_min(0)
        deep_scale = torch.where(
            raw_deep.detach() > 0,
            (maximum / raw_deep.detach().clamp_min(1e-12)).clamp(max=1),
            torch.ones_like(raw_deep.detach()),
        )
    weighted_deep = auxiliary_outer_scale * deep_scale * raw_deep
    total = total + weighted_deep
    ambiguity_logits = (
        _tensor_list(outputs["ambiguity_logits"], "ambiguity_logits")
        if "ambiguity_logits" in outputs
        else []
    )
    if ambiguity_logits:
        if len(ambiguity_logits) != 3:
            raise ValueError("ambiguity supervision requires E1, E2, and E3 scales")
        reference_logits = (
            teacher_outputs.get("seg_logits")
            if teacher_outputs is not None
            else outputs.get("seg_logits")
        )
        if not isinstance(reference_logits, Tensor):
            raise TypeError("ambiguity supervision requires reference segmentation logits")
        ambiguity_losses = [
            boundary_scale_loss(
                logit,
                ambiguity_target(mask, reference_logits, logit),
            )
            for logit in ambiguity_logits
        ]
        ambiguity_weights = seg.new_tensor((0.5, 0.3, 0.2))
        ambiguity = (torch.stack(ambiguity_losses) * ambiguity_weights).sum()
        raw_ambiguity = lambda_ambiguity * ambiguity
        with torch.no_grad():
            ambiguity_maximum = max_ambiguity_to_seg_ratio * seg.detach().clamp_min(0)
            ambiguity_scale = torch.where(
                raw_ambiguity.detach() > 0,
                (
                    ambiguity_maximum
                    / raw_ambiguity.detach().clamp_min(1e-12)
                ).clamp(max=1),
                torch.ones_like(raw_ambiguity.detach()),
            )
        weighted_ambiguity = (
            auxiliary_outer_scale * ambiguity_scale * raw_ambiguity
        )
        total = total + weighted_ambiguity
        result.update(
            ambiguity_loss=ambiguity,
            weighted_ambiguity_loss=weighted_ambiguity,
            ambiguity_scale_losses=ambiguity_losses,
            weighted_aux_loss=result["weighted_aux_loss"] + weighted_ambiguity,
        )
    if teacher_outputs is not None:
        distill, _ = hierarchical_output_distillation_loss(
            outputs,
            teacher_outputs,
            temperature=distill_temperature,
            mask=mask,
            boundary_boost=distill_boundary_boost,
        )
        total = total + lambda_distill * distill
        result["distill_loss"] = distill
    result.update(
        total_loss=total,
        deep_supervision_loss=deep,
        weighted_deep_supervision_loss=weighted_deep,
        deep_supervision_scale_losses=deep_losses,
    )
    return result


__all__ = [
    "soft_dice_loss",
    "balanced_bce_with_logits",
    "masked_balanced_bce_with_logits",
    "masked_soft_dice_loss",
    "resize_binary_mask",
    "sobel_magnitude",
    "soft_boundary_target",
    "distance_boundary_target",
    "ambiguity_target",
    "texture_energy_target",
    "wavelet_texture_target",
    "boundary_target",
    "make_multiscale_targets",
    "make_gcsas_targets",
    "build_multiscale_targets",
    "segmentation_loss",
    "deep_segmentation_loss",
    "boundary_scale_loss",
    "texture_scale_loss",
    "shape_scale_loss",
    "truncated_signed_distance_target",
    "signed_distance_scale_loss",
    "segmentation_edge_loss",
    "binary_kl_distillation_loss",
    "hierarchical_output_distillation_loss",
    "masked_correlation_loss",
    "gcsas_texture_scale_loss",
    "compute_total_loss",
    "compute_lite_sdf_distill_loss",
    "compute_performance_context_loss",
]
