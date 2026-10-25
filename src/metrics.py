"""Binary segmentation metrics, computed per image before averaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


Reduction = Literal["none", "mean", "sum"]


def _binary(values: Tensor, threshold: float, from_logits: bool) -> Tensor:
    values = torch.as_tensor(values)
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[:, None]
    elif values.ndim != 4 or values.shape[1] != 1:
        raise ValueError(
            "binary masks must have shape [H,W], [B,H,W], or [B,1,H,W]; "
            f"got {tuple(values.shape)}"
        )
    if values.dtype == torch.bool:
        if from_logits:
            raise ValueError("boolean predictions cannot be logits")
        return values
    return (values.sigmoid() if from_logits else values) >= threshold


def _pair(
    prediction: Tensor, target: Tensor, threshold: float, from_logits: bool
) -> tuple[Tensor, Tensor]:
    prediction = _binary(prediction, threshold, from_logits)
    target = _binary(target, 0.5, False)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )
    return prediction, target


def _reduce(scores: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return scores
    if reduction == "mean":
        return scores.mean()
    if reduction == "sum":
        return scores.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def dice_score(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
    reduction: Reduction = "mean",
    epsilon: float = 1e-6,
) -> Tensor:
    """Return per-image Dice scores; two empty masks score one."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    prediction, target = _pair(prediction, target, threshold, from_logits)
    prediction, target = prediction.flatten(1), target.flatten(1)
    intersection = (prediction & target).sum(1, dtype=torch.float32)
    total = prediction.sum(1, dtype=torch.float32) + target.sum(
        1, dtype=torch.float32
    )
    scores = torch.where(
        total > 0,
        2 * intersection / total.clamp_min(epsilon),
        torch.ones_like(total),
    )
    return _reduce(scores, reduction)


def iou_score(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
    reduction: Reduction = "mean",
    epsilon: float = 1e-6,
) -> Tensor:
    """Return per-image IoU scores; two empty masks score one."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    prediction, target = _pair(prediction, target, threshold, from_logits)
    prediction, target = prediction.flatten(1), target.flatten(1)
    intersection = (prediction & target).sum(1, dtype=torch.float32)
    union = (prediction | target).sum(1, dtype=torch.float32)
    scores = torch.where(
        union > 0,
        intersection / union.clamp_min(epsilon),
        torch.ones_like(union),
    )
    return _reduce(scores, reduction)


def accuracy_score(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> Tensor:
    """Return per-image pixel accuracy after thresholding the masks."""

    prediction, target = _pair(prediction, target, threshold, from_logits)
    scores = (prediction == target).flatten(1).float().mean(1)
    return _reduce(scores, reduction)


def _boundary(mask: Tensor) -> Tensor:
    values = mask.float()
    dilated = F.max_pool2d(values, 3, stride=1, padding=1)
    eroded = 1 - F.max_pool2d(1 - values, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1).bool()


def boundary_f1_score(
    prediction: Tensor,
    target: Tensor,
    *,
    tolerance: int = 2,
    distance: Literal["chebyshev", "euclidean"] = "chebyshev",
    threshold: float = 0.5,
    from_logits: bool = False,
    reduction: Reduction = "mean",
    epsilon: float = 1e-6,
) -> Tensor:
    """Match 3x3 morphological-gradient boundaries in evaluation-space pixels.

    Precision counts prediction pixels near any reference boundary pixel;
    recall counts reference pixels near any prediction boundary pixel. Matching
    is many-to-many. BF1 is 2PR/(P+R), computed per image before averaging.
    The legacy square band is preserved unless Euclidean distance is requested.
    """

    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative integer")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if distance not in {"chebyshev", "euclidean"}:
        raise ValueError("distance must be 'chebyshev' or 'euclidean'")
    prediction, target = _pair(prediction, target, threshold, from_logits)
    predicted_boundary, target_boundary = _boundary(prediction), _boundary(target)
    if tolerance and distance == "euclidean":
        coordinates = torch.arange(-tolerance, tolerance + 1, device=prediction.device)
        disk = (coordinates[:, None].square() + coordinates[None, :].square()) <= tolerance**2
        kernel = disk.float()[None, None]
        # Dilate both boundaries with an integer-grid Euclidean disk, not a square.
        predicted_band = F.conv2d(predicted_boundary.float(), kernel, padding=tolerance) > 0
        target_band = F.conv2d(target_boundary.float(), kernel, padding=tolerance) > 0
    elif tolerance:
        kernel = 2 * tolerance + 1
        predicted_band = F.max_pool2d(
            predicted_boundary.float(), kernel, stride=1, padding=tolerance
        ).bool()
        target_band = F.max_pool2d(
            target_boundary.float(), kernel, stride=1, padding=tolerance
        ).bool()
    else:
        predicted_band, target_band = predicted_boundary, target_boundary

    predicted_count = predicted_boundary.flatten(1).sum(1, dtype=torch.float32)
    target_count = target_boundary.flatten(1).sum(1, dtype=torch.float32)
    precision = (predicted_boundary & target_band).flatten(1).sum(
        1, dtype=torch.float32
    ) / predicted_count.clamp_min(epsilon)
    recall = (target_boundary & predicted_band).flatten(1).sum(
        1, dtype=torch.float32
    ) / target_count.clamp_min(epsilon)
    scores = 2 * precision * recall / (precision + recall).clamp_min(epsilon)

    # 空预测对非空病灶为 0；双方都空且掩膜相同才记为 1，绝不产生 NaN。
    both_without_boundary = (predicted_count == 0) & (target_count == 0)
    masks_equal = (prediction == target).flatten(1).all(1)
    scores = torch.where(
        both_without_boundary & masks_equal, torch.ones_like(scores), scores
    )
    return _reduce(scores, reduction)


def compute_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
    boundary_tolerance: int = 2,
    boundary_distance: Literal["chebyshev", "euclidean"] = "chebyshev",
    return_per_image: bool = False,
) -> dict[str, float] | dict[str, Tensor]:
    """Compute Dice, IoU, pixel accuracy, and BF1 with one threshold."""

    common = dict(
        threshold=threshold, from_logits=from_logits, reduction="none"
    )
    scores = {
        "dice": dice_score(prediction, target, **common),
        "iou": iou_score(prediction, target, **common),
        "accuracy": accuracy_score(prediction, target, **common),
        "bf1": boundary_f1_score(
            prediction, target, tolerance=boundary_tolerance, distance=boundary_distance, **common
        ),
    }
    if return_per_image:
        return scores
    return {name: float(values.mean()) for name, values in scores.items()}


@dataclass
class MetricAccumulator:
    """Accumulate per-image metrics independently of DataLoader batch size."""

    threshold: float = 0.5
    from_logits: bool = True
    boundary_tolerance: int = 2
    totals: dict[str, float] = field(
        default_factory=lambda: {
            "dice": 0.0,
            "iou": 0.0,
            "accuracy": 0.0,
            "bf1": 0.0,
        }
    )
    count: int = 0
    boundary_distance: Literal["chebyshev", "euclidean"] = "chebyshev"

    @torch.no_grad()
    def update(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        scores = compute_metrics(
            prediction,
            target,
            threshold=self.threshold,
            from_logits=self.from_logits,
            boundary_tolerance=self.boundary_tolerance,
            boundary_distance=self.boundary_distance,
            return_per_image=True,
        )
        for name, values in scores.items():
            self.totals[name] += float(values.sum())
        self.count += scores["dice"].numel()
        return scores

    def compute(self) -> dict[str, float]:
        if not self.count:
            raise RuntimeError("cannot compute metrics before adding images")
        return {name: total / self.count for name, total in self.totals.items()}

    def reset(self) -> None:
        self.totals = {
            "dice": 0.0,
            "iou": 0.0,
            "accuracy": 0.0,
            "bf1": 0.0,
        }
        self.count = 0
