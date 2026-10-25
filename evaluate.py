"""Restore a training checkpoint and evaluate it on a fixed ISIC split."""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.dataset import ISICDataset
from src.metrics import MetricAccumulator
from src.models import build_model
from src.utils import (
    count_inference_parameters,
    count_trainable_parameters,
    device_name,
    resolve_device,
    save_json,
    seed_worker,
    set_seed,
)


RESULT_CSV_FIELDS = (
    "model",
    "base_channels",
    "seed",
    "split",
    "dice",
    "iou",
    "accuracy",
    "bf1",
    "num_images",
    "best_val_dice",
    "training_elapsed_seconds",
    "epochs_completed",
    "seconds_per_epoch",
    "inference_total_seconds",
    "inference_num_images",
    "inference_ms_per_image",
    "inference_images_per_second",
    "trainable_parameters",
    "inference_parameters",
    "checkpoint",
    "checkpoint_epoch",
    "device",
    "device_name",
    "threshold",
    "boundary_tolerance",
    "boundary_distance",
    "benchmark_scope",
    "benchmark_warmup_batches",
    "benchmark_batches",
    "benchmark_repetitions",
    "benchmark_batch_size",
    "benchmark_image_height",
    "benchmark_image_width",
    "benchmark_device",
    "benchmark_auxiliary_outputs_enabled",
)


def save_metrics_csv(
    result: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Save one completed evaluation as a stable, spreadsheet-friendly CSV row."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = dict(result)
    row["model"] = row.get("model", row.get("model_name", ""))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_CSV_FIELDS)
        writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in RESULT_CSV_FIELDS})
    return output


def _positive_int(value: str) -> int:
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(value: str) -> int:
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Build the saved model and restore its current checkpoint format strictly."""

    device = resolve_device(device)
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        # Optimizer/scheduler state can be large and is not needed on the GPU for evaluation.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 2.0 does not expose weights_only.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    checkpoint = dict(checkpoint)
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint 'config' must be a mapping")
    config = dict(config)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise ValueError("checkpoint 'model_state_dict' must map names to tensors")

    model_name = checkpoint.get("model_name", config.get("model"))
    if not isinstance(model_name, str):
        raise ValueError("checkpoint must contain a model name")
    first = state.get("backbone.encoder1.0.weight")
    last = state.get("backbone.segmentation_head.weight")
    inferred_base_channels = int(first.shape[0]) if first is not None else None
    configured_base_channels = config.get("base_channels")
    if (
        configured_base_channels is not None
        and inferred_base_channels is not None
        and int(configured_base_channels) != inferred_base_channels
    ):
        raise ValueError(
            "checkpoint base_channels does not match the first encoder weight: "
            f"config={configured_base_channels}, weights={inferred_base_channels}"
        )
    constructor = {
        "in_channels": int(config.get("in_channels", first.shape[1] if first is not None else 3)),
        "out_channels": int(config.get("out_channels", last.shape[0] if last is not None else 1)),
        "base_channels": int(
            configured_base_channels
            if configured_base_channels is not None
            else inferred_base_channels or 32
        ),
    }
    model = build_model(model_name, **constructor)
    model.load_state_dict(state, strict=True)
    model.to(device)
    metadata = {
        "model_name": model_name,
        "config": config,
        "seed": checkpoint.get("seed", config.get("seed", 42)),
        "epoch": checkpoint.get("epoch"),
        "best_val_dice": checkpoint.get("best_val_dice"),
    }
    return model, checkpoint, metadata


def _batch(batch: Any) -> tuple[Tensor, Tensor, Any]:
    if not isinstance(batch, Mapping) or "image" not in batch or "mask" not in batch:
        raise TypeError("loader batches must contain 'image' and 'mask'")
    images, masks = batch["image"], batch["mask"]
    if not isinstance(images, Tensor) or not isinstance(masks, Tensor):
        raise TypeError("batch images and masks must be tensors")
    return images, masks, batch.get("id")


def _forward(model: nn.Module, images: Tensor, return_aux: bool) -> Tensor:
    output = model(images, return_aux=return_aux)
    if not isinstance(output, Mapping) or not isinstance(output.get("seg_logits"), Tensor):
        raise TypeError("model output must contain tensor 'seg_logits'")
    return output["seg_logits"]


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Any],
    device: str | torch.device,
    *,
    threshold: float = 0.5,
    boundary_tolerance: int = 2,
    boundary_distance: str = "chebyshev",
    return_aux: bool = False,
) -> dict[str, float | int]:
    """Average Dice, IoU, pixel accuracy, and BF1 over images."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    device = resolve_device(device)
    was_training = model.training
    model.eval()
    metrics = MetricAccumulator(threshold, True, boundary_tolerance, boundary_distance=boundary_distance)
    try:
        for batch in loader:
            images, masks, _ = _batch(batch)
            metrics.update(
                _forward(model, images.to(device, non_blocking=True), return_aux),
                masks.to(device, non_blocking=True),
            )
    finally:
        model.train(was_training)
    result: dict[str, float | int] = metrics.compute()
    result["num_images"] = metrics.count
    return result


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA kernels so wall-clock measurements are accurate."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_inference(
    model: nn.Module,
    loader: DataLoader[Any],
    device: str | torch.device,
    *,
    warmup_batches: int = 5,
    benchmark_batches: int = 20,
    repetitions: int = 3,
    return_aux: bool = False,
) -> dict[str, float | int | str | bool]:
    """Measure model-forward throughput under a reproducible input workload.

    Selected image batches are first copied to ``device`` and then reused, so the
    reported time excludes data loading, preprocessing, and host-to-device transfer.
    CUDA is synchronized immediately before and after the timed region. Auxiliary
    heads are disabled by default, which measures the actual TriFeatureUNet inference
    path and keeps it comparable with the baseline U-Net.
    """

    if warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")
    if benchmark_batches <= 0:
        raise ValueError("benchmark_batches must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    device = resolve_device(device)
    cached: list[Tensor] = []
    input_shape: tuple[int, int, int] | None = None
    for index, batch in enumerate(loader):
        if index >= benchmark_batches:
            break
        images, _, _ = _batch(batch)
        if images.ndim != 4 or images.shape[0] <= 0:
            raise ValueError("benchmark images must be shaped [B, C, H, W]")
        shape = tuple(int(value) for value in images.shape[1:])
        if input_shape is None:
            input_shape = shape
        elif shape != input_shape:
            raise ValueError(
                "all benchmark batches must have the same channel and image size"
            )
        cached.append(images.to(device, non_blocking=True))

    if not cached or input_shape is None:
        raise ValueError("benchmark loader must contain at least one batch")

    # Complete asynchronous copies before warmup; transfers are outside the timer.
    _synchronize(device)
    was_training = model.training
    model.eval()
    try:
        for index in range(warmup_batches):
            _forward(model, cached[index % len(cached)], return_aux)
        _synchronize(device)

        start = time.perf_counter()
        for _ in range(repetitions):
            for images in cached:
                _forward(model, images, return_aux)
        _synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        model.train(was_training)

    images_per_repeat = sum(int(images.shape[0]) for images in cached)
    total_images = images_per_repeat * repetitions
    batch_size = loader.batch_size
    if not isinstance(batch_size, int) or batch_size <= 0:
        batch_size = int(cached[0].shape[0])
    channels, height, width = input_shape
    return {
        "inference_total_seconds": elapsed,
        "inference_num_images": total_images,
        "inference_ms_per_image": elapsed * 1000.0 / total_images,
        "inference_images_per_second": total_images / elapsed,
        "benchmark_scope": "model_forward_only",
        "benchmark_warmup_batches": warmup_batches,
        "benchmark_batches": len(cached),
        "benchmark_repetitions": repetitions,
        "benchmark_images_per_repeat": images_per_repeat,
        "benchmark_batch_size": batch_size,
        "benchmark_input_channels": channels,
        "benchmark_image_height": height,
        "benchmark_image_width": width,
        "benchmark_device": str(device),
        "benchmark_auxiliary_outputs_enabled": return_aux,
    }


def _chw(value: Tensor, name: str) -> Tensor:
    value = value.detach().cpu()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3:
        raise ValueError(f"{name} must represent one image")
    return value


def save_prediction_grid(
    image: Tensor,
    target: Tensor,
    probability: Tensor,
    output_path: str | Path,
    *,
    threshold: float = 0.5,
    sample_id: str | None = None,
) -> Path:
    """Save image, ground truth, probability, and binary prediction panels."""

    import matplotlib.pyplot as plt

    image, target, probability = (
        _chw(image, "image"),
        _chw(target, "target"),
        _chw(probability, "probability"),
    )
    if image.shape[0] != 3 or target.shape[0] != 1 or probability.shape[0] != 1:
        raise ValueError("expected RGB image and single-channel target/probability")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb = (image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    probability = probability[0].clamp(0, 1).numpy()
    panels = (
        (rgb, "Image", None),
        (target[0].numpy(), "Ground truth", "gray"),
        (probability, "Probability", "viridis"),
        (probability >= threshold, f"Prediction (t={threshold:g})", "gray"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for axis, (values, title, colour_map) in zip(axes, panels):
        axis.imshow(values, cmap=colour_map, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    if sample_id:
        figure.suptitle(sample_id)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def fixed_sample_indices(dataset_size: int, count: int) -> list[int]:
    """Choose deterministic, evenly spaced positions for cross-run visualizations."""

    if dataset_size < 0 or count < 0:
        raise ValueError("dataset_size and count must be non-negative")
    count = min(dataset_size, count)
    if not count:
        return []
    if count == 1:
        return [0]
    return [index * (dataset_size - 1) // (count - 1) for index in range(count)]


@torch.inference_mode()
def save_fixed_predictions(
    model: nn.Module,
    dataset: Dataset[Any],
    device: str | torch.device,
    output_dir: str | Path,
    *,
    count: int = 5,
    threshold: float = 0.5,
    return_aux: bool = False,
) -> list[Path]:
    """Save the same fixed dataset positions for every model run."""

    device = resolve_device(device)
    was_training = model.training
    model.eval()
    paths: list[Path] = []
    try:
        for number, index in enumerate(fixed_sample_indices(len(dataset), count), 1):
            sample = dataset[index]
            image, mask, sample_id = _batch(sample)
            logits = _forward(model, image[None].to(device), return_aux)
            path = Path(output_dir) / f"sample_{number:03d}.png"
            paths.append(
                save_prediction_grid(
                    image,
                    mask,
                    logits.sigmoid()[0],
                    path,
                    threshold=threshold,
                    sample_id=str(sample_id) if sample_id is not None else f"index_{index}",
                )
            )
    finally:
        model.train(was_training)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--splits-path", type=Path,
        default=Path(__file__).resolve().parent / "data/ISIC2018/paper_split_70_10_20_seed20260912.json",
        help="locked paper split; overrides the historical splits.json default",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--num-workers", type=_non_negative_int, default=0)
    parser.add_argument("--image-size", type=_positive_int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=_non_negative_int, default=2)
    parser.add_argument("--boundary-distance", choices=("chebyshev", "euclidean"), default="euclidean")
    parser.add_argument("--num-visualizations", type=_non_negative_int, default=5)
    parser.add_argument(
        "--benchmark-warmup-batches",
        type=_non_negative_int,
        default=5,
        help="untimed forward batches used to warm up the selected device",
    )
    parser.add_argument(
        "--benchmark-batches",
        type=_positive_int,
        default=20,
        help="maximum cached batches timed in each benchmark repetition",
    )
    parser.add_argument(
        "--benchmark-repetitions",
        type=_positive_int,
        default=3,
        help="number of times to time the selected cached batches",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a GPU such as cuda:0"
    )
    auxiliary = parser.add_mutually_exclusive_group()
    auxiliary.add_argument("--no-aux", dest="with_aux", action="store_false")
    auxiliary.add_argument("--with-aux", action="store_true")
    parser.set_defaults(with_aux=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    device = resolve_device(args.device)
    print(
        f"device={device} name={device_name(device)} "
        f"cuda={torch.version.cuda if device.type == 'cuda' else 'n/a'}"
    )
    model, _, metadata = load_model_from_checkpoint(args.checkpoint, device)
    config = metadata["config"]
    seed = int(metadata["seed"])
    set_seed(seed)
    dataset = ISICDataset(
        args.data_root,
        split=args.split,
        splits_path=args.splits_path,
        image_size=args.image_size or int(config.get("image_size", 256)),
        augment=False,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    # 正式预测默认关闭仅用于训练监督的辅助读取头。
    metrics = evaluate_model(
        model,
        loader,
        device,
        threshold=args.threshold,
        boundary_tolerance=args.boundary_tolerance,
        boundary_distance=args.boundary_distance,
        return_aux=args.with_aux,
    )
    benchmark = benchmark_inference(
        model,
        loader,
        device,
        warmup_batches=args.benchmark_warmup_batches,
        benchmark_batches=args.benchmark_batches,
        repetitions=args.benchmark_repetitions,
        return_aux=args.with_aux,
    )
    output_dir = args.output_dir or Path("outputs/evaluation") / args.checkpoint.parent.name
    predictions = save_fixed_predictions(
        model,
        dataset,
        device,
        output_dir / "predictions",
        count=args.num_visualizations,
        threshold=args.threshold,
        return_aux=args.with_aux,
    )
    result = {
        "model_name": metadata["model_name"],
        "base_channels": model.backbone.base_channels,
        "seed": seed,
        "split": args.split,
        **metrics,
        **benchmark,
        "trainable_parameters": count_trainable_parameters(model),
        "inference_parameters": count_inference_parameters(model),
        "auxiliary_outputs_enabled": args.with_aux,
        "threshold": args.threshold,
        "boundary_tolerance": args.boundary_tolerance,
        "boundary_distance": args.boundary_distance,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": metadata["epoch"],
        "best_val_dice": metadata["best_val_dice"],
        "device": str(device),
        "device_name": device_name(device),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if device.type == "cuda" else "n/a",
        "prediction_files": [str(path) for path in predictions],
    }
    stem = "test_metrics" if args.split == "test" else f"{args.split}_metrics"
    json_output = save_json(result, output_dir / f"{stem}.json")
    csv_output = save_metrics_csv(result, output_dir / f"{stem}.csv")
    print(f"Saved {args.split} metrics to {json_output} and {csv_output}")
    print(
        {
            name: result[name]
            for name in (
                "dice",
                "iou",
                "accuracy",
                "bf1",
                "inference_ms_per_image",
                "inference_images_per_second",
            )
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
