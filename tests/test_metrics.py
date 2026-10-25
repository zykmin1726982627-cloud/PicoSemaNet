import math

import pytest
import torch
from torch.utils.data import DataLoader

from evaluate import build_parser as build_evaluate_parser
from evaluate import evaluate_model, load_model_from_checkpoint
from src.metrics import (
    MetricAccumulator,
    accuracy_score,
    boundary_f1_score,
    compute_metrics,
    dice_score,
    iou_score,
)
from src.models import TriFeatureUNet, UNet
from src.utils import count_inference_parameters, count_trainable_parameters


def test_overlap_metrics_are_computed_per_image_before_mean() -> None:
    target = torch.zeros(2, 1, 8, 8)
    prediction = torch.zeros_like(target)
    target[0, 0, 2, 2] = 1
    prediction[0, 0, 2, 2] = 1
    target[1, 0, 2:4, 2:4] = 1

    per_image_dice = dice_score(prediction, target, reduction="none")
    per_image_iou = iou_score(prediction, target, reduction="none")

    assert per_image_dice.tolist() == pytest.approx([1.0, 0.0])
    assert per_image_iou.tolist() == pytest.approx([1.0, 0.0])
    assert dice_score(prediction, target).item() == pytest.approx(0.5)
    assert iou_score(prediction, target).item() == pytest.approx(0.5)


def test_accuracy_is_computed_per_image_before_reduction() -> None:
    target = torch.tensor(
        [
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    prediction = torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ]
    )

    per_image = accuracy_score(prediction, target, reduction="none")

    assert per_image.tolist() == pytest.approx([0.75, 0.0])
    assert accuracy_score(prediction, target, reduction="mean").item() == pytest.approx(
        0.375
    )
    assert accuracy_score(prediction, target, reduction="sum").item() == pytest.approx(
        0.75
    )


def test_accuracy_empty_masks_use_natural_pixel_correctness() -> None:
    empty = torch.zeros(1, 1, 4, 4)
    lesion = empty.clone()
    lesion[..., :2, :2] = 1

    assert accuracy_score(empty, empty).item() == pytest.approx(1.0)
    assert accuracy_score(empty, lesion).item() == pytest.approx(0.75)
    assert accuracy_score(lesion, empty).item() == pytest.approx(0.75)


def test_accuracy_supports_logits_and_custom_threshold() -> None:
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    probabilities = torch.tensor([[[[0.2, 0.7], [0.6, 0.4]]]])
    logits = torch.logit(probabilities)

    assert accuracy_score(logits, target, from_logits=True).item() == pytest.approx(
        1.0
    )
    assert accuracy_score(
        probabilities, target, threshold=0.65
    ).item() == pytest.approx(0.75)


@pytest.mark.parametrize("metric", [dice_score, iou_score, boundary_f1_score])
def test_empty_mask_cases_are_finite_and_explicit(metric) -> None:
    empty = torch.zeros(1, 1, 16, 16)
    lesion = empty.clone()
    lesion[..., 4:12, 4:12] = 1

    both_empty = metric(empty, empty).item()
    empty_prediction = metric(empty, lesion).item()
    empty_target = metric(lesion, empty).item()

    assert both_empty == pytest.approx(1.0)
    assert empty_prediction == pytest.approx(0.0)
    assert empty_target == pytest.approx(0.0)
    assert all(math.isfinite(value) for value in (both_empty, empty_prediction, empty_target))


def test_boundary_f1_honours_two_pixel_tolerance() -> None:
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 8:20] = 1
    shifted_two = torch.zeros_like(target)
    shifted_two[..., 8:24, 10:22] = 1
    shifted_four = torch.zeros_like(target)
    shifted_four[..., 8:24, 12:24] = 1

    within_tolerance = boundary_f1_score(
        shifted_two, target, tolerance=2
    ).item()
    outside_tolerance = boundary_f1_score(
        shifted_four, target, tolerance=2
    ).item()

    assert within_tolerance == pytest.approx(1.0)
    assert 0.0 <= outside_tolerance < within_tolerance


def test_logits_threshold_and_accumulator_match_direct_metrics() -> None:
    target = torch.tensor(
        [
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ]
    )
    logits = torch.where(target.bool(), torch.tensor(8.0), torch.tensor(-8.0))
    direct = compute_metrics(logits, target, from_logits=True)
    accumulator = MetricAccumulator(from_logits=True)
    accumulator.update(logits[:1], target[:1])
    accumulator.update(logits[1:], target[1:])

    assert direct == pytest.approx(
        {"dice": 1.0, "iou": 1.0, "accuracy": 1.0, "bf1": 1.0}
    )
    assert accumulator.compute() == pytest.approx(direct)
    assert accumulator.count == 2

    accumulator.reset()
    assert accumulator.count == 0
    assert accumulator.totals == {
        "dice": 0.0,
        "iou": 0.0,
        "accuracy": 0.0,
        "bf1": 0.0,
    }


@pytest.mark.parametrize("tolerance", [0, 1, 2])
def test_euclidean_bf1_matches_pairwise_distance_reference(tolerance) -> None:
    # A diagonal displacement distinguishes an Euclidean disk from a square.
    from src.metrics import _boundary

    target = torch.zeros(2, 1, 20, 20)
    prediction = torch.zeros_like(target)
    target[0, 0, 4:11, 4:11] = 1
    prediction[0, 0, 6:13, 6:13] = 1
    target[1, 0, 3:9, 4:14] = 1
    prediction[1, 0, 3:9, 6:16] = 1
    expected = []
    for pred, ref in zip(_boundary(prediction), _boundary(target)):
        points_pred = torch.nonzero(pred[0]).float()
        points_ref = torch.nonzero(ref[0]).float()
        distances = torch.cdist(points_pred, points_ref)
        precision = (distances.min(1).values <= tolerance).float().mean()
        recall = (distances.min(0).values <= tolerance).float().mean()
        expected.append(float(2 * precision * recall / (precision + recall).clamp_min(1e-6)))
    actual = boundary_f1_score(prediction, target, tolerance=tolerance, distance="euclidean", reduction="none")
    assert actual.tolist() == pytest.approx(expected)
    accumulator = MetricAccumulator(boundary_tolerance=tolerance, from_logits=False, boundary_distance="euclidean")
    accumulator.update(prediction, target)
    assert accumulator.compute()["bf1"] == pytest.approx(sum(expected) / len(expected))
    if tolerance == 2:
        square = boundary_f1_score(prediction, target, tolerance=2, reduction="none")
        assert actual[0] < square[0]


@pytest.mark.parametrize("distance", ["chebyshev", "euclidean"])
def test_boundary_distance_empty_cases(distance) -> None:
    empty = torch.zeros(1, 1, 16, 16)
    lesion = empty.clone()
    lesion[..., 4:12, 4:12] = 1
    assert boundary_f1_score(empty, empty, distance=distance).item() == 1
    assert boundary_f1_score(empty, lesion, distance=distance).item() == 0


def test_inference_parameter_count_excludes_only_tri_auxiliary_heads() -> None:
    baseline = UNet(base_channels=4)
    tri = TriFeatureUNet(base_channels=4)

    assert count_inference_parameters(tri) == count_trainable_parameters(baseline)
    assert count_trainable_parameters(tri) > count_inference_parameters(tri)


def test_checkpoint_loading_and_test_evaluation(tmp_path) -> None:
    model = UNet(base_channels=4)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_name": "unet",
            "model_state_dict": model.state_dict(),
            "config": {"model": "unet", "base_channels": 4, "seed": 43},
            "seed": 43,
            "epoch": 2,
            "best_val_dice": 0.75,
        },
        checkpoint_path,
    )
    restored, _, metadata = load_model_from_checkpoint(checkpoint_path, "cpu")
    samples = [
        {
            "image": torch.zeros(3, 16, 16),
            "mask": torch.zeros(1, 16, 16),
            "id": "ISIC_0000000",
        },
        {
            "image": torch.ones(3, 16, 16),
            "mask": torch.ones(1, 16, 16),
            "id": "ISIC_0000001",
        },
    ]
    metrics = evaluate_model(
        restored, DataLoader(samples, batch_size=2), "cpu", return_aux=False
    )

    assert metadata["model_name"] == "unet"
    assert metadata["seed"] == 43
    assert metrics["num_images"] == 2
    assert all(
        math.isfinite(float(metrics[name]))
        for name in ("dice", "iou", "accuracy", "bf1")
    )


@pytest.mark.parametrize(
    ("base_channels", "record_width"),
    [(32, False), (24, True), (16, True), (12, True), (8, True)],
)
def test_tri_checkpoint_loader_supports_old_and_sweep_widths(
    tmp_path, base_channels: int, record_width: bool
) -> None:
    model = TriFeatureUNet(base_channels=base_channels)
    config = {"model": "tri_unet", "seed": 42}
    if record_width:
        config["base_channels"] = base_channels
    checkpoint_path = tmp_path / f"tri_base{base_channels}.pt"
    torch.save(
        {
            "model_name": "tri_unet",
            "model_state_dict": model.state_dict(),
            "config": config,
            "seed": 42,
        },
        checkpoint_path,
    )

    restored, _, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
    assert isinstance(restored, TriFeatureUNet)
    assert restored.backbone.base_channels == base_channels
    restored.eval()
    with torch.inference_mode():
        output = restored(torch.zeros(1, 3, 16, 16), return_aux=False)
    assert output["seg_logits"].shape == (1, 1, 16, 16)


def test_checkpoint_loader_rejects_width_metadata_mismatch(tmp_path) -> None:
    model = TriFeatureUNet(base_channels=8)
    checkpoint_path = tmp_path / "mismatched.pt"
    torch.save(
        {
            "model_name": "tri_unet",
            "model_state_dict": model.state_dict(),
            "config": {"model": "tri_unet", "base_channels": 12, "seed": 42},
            "seed": 42,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="does not match"):
        load_model_from_checkpoint(checkpoint_path, "cpu")


def test_evaluation_cli_disables_auxiliary_heads_by_default() -> None:
    parser = build_evaluate_parser()
    base_args = ["--checkpoint", "best.pt", "--data-root", "data/ISIC2018"]
    default = parser.parse_args(base_args)
    explicit_no_aux = parser.parse_args([*base_args, "--no-aux"])
    explicit_with_aux = parser.parse_args([*base_args, "--with-aux"])

    assert not default.with_aux
    assert not explicit_no_aux.with_aux
    assert explicit_with_aux.with_aux
