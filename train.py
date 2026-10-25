"""以相同设置训练 U-Net 或 TriFeatureUNet。"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import ISICDataset, seed_worker
from src.losses import (
    compute_lite_sdf_distill_loss,
    compute_performance_context_loss,
    compute_total_loss,
    make_gcsas_targets,
    make_multiscale_targets,
)
from src.metrics import MetricAccumulator, boundary_f1_score
from src.models import (
    AUXILIARY_MODEL_NAMES,
    BASELINE_MODEL_NAMES,
    RETIRING_AUXILIARY_MODEL_NAMES,
    TriFeatureUNet,
    build_model,
)
from src.utils import (
    count_inference_parameters,
    count_trainable_parameters,
    device_name,
    resolve_device,
    save_json,
    set_seed,
)


SUPPORTED_BASE_CHANNELS = (2, 4, 6, 8, 9, 12, 16, 18, 20, 24, 32)
MODEL_NAMES = (*BASELINE_MODEL_NAMES, *AUXILIARY_MODEL_NAMES)


@dataclass(frozen=True)
class TrainConfig:
    """一次训练的完整配置。"""

    model: str
    data_root: Path
    output_dir: Path
    splits_path: Path | None = None
    base_channels: int | None = None
    seed: int = 42
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 15
    lambda_boundary: float = 0.2
    lambda_texture: float = 0.2
    lambda_shape: float = 0.2
    lambda_sdf: float = 0.15
    lambda_distill: float = 0.20
    lambda_edge: float = 0.05
    lambda_deep_supervision: float = 0.20
    max_deep_to_seg_ratio: float = 0.25
    lambda_ambiguity: float = 0.10
    max_ambiguity_to_seg_ratio: float = 0.10
    distill_temperature: float = 2.0
    distill_boundary_boost: float = 0.0
    teacher_checkpoint: Path | None = None
    initial_checkpoint: Path | None = None
    ema_decay: float = 0.995
    auxiliary_floor: float = 0.15
    max_aux_to_seg_ratio: float = 0.5
    aux_warmup_epochs: int = 5
    gcsas_aux_start_epochs: int = 5
    gcsas_aux_multiplier: float = 0.5
    gcsas_max_aux_to_seg_ratio: float = 0.25
    aux_retire_start: float = 0.35
    aux_retire_end: float = 0.75
    aux_retire_start_epoch: int | None = None
    aux_retire_end_epoch: int | None = None
    aux_diagnostics: bool = False
    num_workers: int = 0
    amp: bool = False
    device: str = "auto"
    threshold: float = 0.5
    boundary_tolerance: int = 2
    boundary_distance: str = "chebyshev"
    num_predictions: int = 4
    benchmark_warmup_batches: int = 5
    benchmark_batches: int = 20
    benchmark_repetitions: int = 3
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    batch_cooldown_ms: float = 0.0
    allow_existing_output: bool = False


LOSS_FIELDS = (
    "total_loss",
    "seg_loss",
    "boundary_loss",
    "texture_loss",
    "ambiguity_loss",
    "weighted_ambiguity_loss",
    "shape_loss",
    "sdf_loss",
    "distill_loss",
    "edge_loss",
    "deep_supervision_loss",
    "weighted_deep_supervision_loss",
    "weighted_aux_loss",
    "aux_weight_scale",
    *(f"boundary_loss_e{i}" for i in range(1, 4)),
    *(f"texture_loss_e{i}" for i in range(1, 4)),
    *(f"ambiguity_loss_e{i}" for i in range(1, 4)),
    *(f"shape_loss_e{i}" for i in range(1, 4)),
    *(f"sdf_loss_e{i}" for i in range(1, 4)),
)
VAL_FIELDS = (
    *LOSS_FIELDS,
    "dice",
    "iou",
    "accuracy",
    "bf1",
    "boundary_head_bf1",
    "texture_head_mae",
    *(f"boundary_head_bf1_e{i}" for i in range(1, 4)),
    *(f"texture_head_mae_e{i}" for i in range(1, 4)),
)
LOG_FIELDS = (
    "epoch",
    "learning_rate",
    "epoch_seconds",
    "train_seconds",
    "validation_seconds",
    "cumulative_seconds",
    "auxiliary_scale",
    *(f"train_{name}" for name in LOSS_FIELDS),
    *(f"val_{name}" for name in VAL_FIELDS),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits-path",
        type=Path,
        help="explicit immutable split JSON; defaults to <data-root>/splits.json",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        choices=SUPPORTED_BASE_CHANNELS,
        default=None,
        help="explicit U-Net base width; omit to use the model default",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--lambda-boundary", type=float, default=0.2)
    parser.add_argument(
        "--lambda-texture",
        "--lambda-region",
        dest="lambda_texture",
        type=float,
        default=0.2,
        help="texture auxiliary weight; --lambda-region is a legacy alias",
    )
    parser.add_argument("--lambda-shape", type=float, default=0.2)
    parser.add_argument("--lambda-sdf", type=float, default=0.15)
    parser.add_argument("--lambda-distill", type=float, default=0.20)
    parser.add_argument("--lambda-edge", type=float, default=0.05)
    parser.add_argument("--lambda-deep-supervision", type=float, default=0.20)
    parser.add_argument("--max-deep-to-seg-ratio", type=float, default=0.25)
    parser.add_argument("--lambda-ambiguity", type=float, default=0.10)
    parser.add_argument("--max-ambiguity-to-seg-ratio", type=float, default=0.10)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--distill-boundary-boost",
        type=float,
        default=0.0,
        help="extra hierarchical-KD weight on ground-truth boundary pixels",
    )
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--auxiliary-floor", type=float, default=0.15)
    parser.add_argument("--max-aux-to-seg-ratio", type=float, default=0.5)
    parser.add_argument("--aux-warmup-epochs", type=int, default=5)
    parser.add_argument("--gcsas-aux-start-epochs", type=int, default=5)
    parser.add_argument("--gcsas-aux-multiplier", type=float, default=0.5)
    parser.add_argument("--gcsas-max-aux-to-seg-ratio", type=float, default=0.25)
    parser.add_argument(
        "--aux-retire-start",
        type=float,
        default=0.35,
        help="fraction of training where retiring auxiliary supervision starts",
    )
    parser.add_argument(
        "--aux-retire-end",
        type=float,
        default=0.75,
        help="fraction of training where retiring auxiliary supervision reaches zero",
    )
    parser.add_argument("--aux-retire-start-epoch", type=int)
    parser.add_argument("--aux-retire-end-epoch", type=int)
    parser.add_argument(
        "--aux-diagnostics",
        action="store_true",
        help="also execute auxiliary heads during validation (slower)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp", action="store_true", help="use CUDA mixed precision to save GPU memory"
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a GPU such as cuda:0"
    )
    parser.add_argument("--num-predictions", type=int, default=4)
    parser.add_argument(
        "--benchmark-warmup-batches",
        type=int,
        default=5,
        help="untimed forward batches used to warm up the inference device",
    )
    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=20,
        help="maximum cached batches timed in each inference repetition",
    )
    parser.add_argument(
        "--benchmark-repetitions",
        type=int,
        default=3,
        help="number of inference timing repetitions",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--batch-cooldown-ms",
        type=float,
        default=0.0,
        help="optional idle time after each training batch to limit sustained GPU heat",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--boundary-distance", choices=("chebyshev", "euclidean"), default="chebyshev")
    return parser


def _validate(config: TrainConfig) -> None:
    if not 0 <= config.threshold <= 1 or config.boundary_tolerance < 0:
        raise ValueError("invalid metric threshold or boundary tolerance")
    if config.boundary_distance not in {"chebyshev", "euclidean"}:
        raise ValueError("invalid boundary distance")
    if config.model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    if (
        config.base_channels is not None
        and config.base_channels not in SUPPORTED_BASE_CHANNELS
    ):
        raise ValueError(
            f"--base-channels must be one of {SUPPORTED_BASE_CHANNELS}"
        )
    if config.image_size <= 0 or config.image_size % 16:
        raise ValueError("--image-size must be positive and divisible by 16")
    if min(config.batch_size, config.epochs) <= 0:
        raise ValueError("--batch-size and --epochs must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("invalid learning rate or weight decay")
    if not 0 < config.scheduler_factor < 1:
        raise ValueError("--scheduler-factor must be between 0 and 1")
    if config.scheduler_patience < 0 or config.early_stopping_patience <= 0:
        raise ValueError("invalid patience")
    if config.min_learning_rate <= 0:
        raise ValueError("--min-learning-rate must be positive")
    if min(
        config.lambda_boundary,
        config.lambda_texture,
        config.lambda_shape,
        config.lambda_sdf,
        config.lambda_distill,
        config.lambda_edge,
        config.lambda_deep_supervision,
        config.max_deep_to_seg_ratio,
        config.lambda_ambiguity,
        config.max_ambiguity_to_seg_ratio,
    ) < 0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if config.distill_temperature <= 0:
        raise ValueError("--distill-temperature must be positive")
    if config.distill_boundary_boost < 0:
        raise ValueError("--distill-boundary-boost must be non-negative")
    if not 0 <= config.ema_decay < 1 or not 0 <= config.auxiliary_floor <= 1:
        raise ValueError("EMA decay and auxiliary floor must be in valid ranges")
    if config.model in {
        "lite_sdf_distill",
        "ultralite_sdf_distill",
        "nanolite_sdf_distill",
        "microlite_sdf_distill",
        "picolite_sdf_distill",
        "microlite_fullpath_distill",
        "picolite_fullpath_distill",
        "picolite_selective_fullpath_distill",
        "picolite_shallow_semantic_distill",
        "picolite_learned_head_fullpath_distill",
        "picolite_encoder_semantic_distill",
        "compressed_context",
        "compressed_context_fullpath",
        "ambiguity_context",
    }:
        if config.lambda_distill > 0 and config.teacher_checkpoint is None:
            raise ValueError("--teacher-checkpoint is required for lite_sdf_distill")
        if config.teacher_checkpoint is not None and not config.teacher_checkpoint.is_file():
            raise FileNotFoundError(config.teacher_checkpoint)
    if config.initial_checkpoint is not None and not config.initial_checkpoint.is_file():
        raise FileNotFoundError(config.initial_checkpoint)
    if config.max_aux_to_seg_ratio < 0:
        raise ValueError("--max-aux-to-seg-ratio must be non-negative")
    if min(config.aux_warmup_epochs, config.gcsas_aux_start_epochs) < 0:
        raise ValueError("auxiliary warmup/start epochs must be non-negative")
    if config.gcsas_aux_multiplier < 0 or config.gcsas_max_aux_to_seg_ratio < 0:
        raise ValueError("GCSAS auxiliary multiplier and cap must be non-negative")
    if not 0 <= config.aux_retire_start < config.aux_retire_end <= 1:
        raise ValueError("auxiliary retirement fractions must satisfy 0 <= start < end <= 1")
    retire_epochs = (config.aux_retire_start_epoch, config.aux_retire_end_epoch)
    if (retire_epochs[0] is None) != (retire_epochs[1] is None):
        raise ValueError("absolute auxiliary retirement epochs must be supplied together")
    if retire_epochs[0] is not None and not (
        0 <= retire_epochs[0] < retire_epochs[1] <= config.epochs
    ):
        raise ValueError("absolute auxiliary retirement epochs must satisfy 0 <= start < end <= epochs")
    if min(
        config.num_workers,
        config.num_predictions,
        config.benchmark_warmup_batches,
    ) < 0:
        raise ValueError(
            "worker, prediction, and benchmark warmup counts must be non-negative"
        )
    if min(config.benchmark_batches, config.benchmark_repetitions) <= 0:
        raise ValueError(
            "--benchmark-batches and --benchmark-repetitions must be positive"
        )
    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        value = getattr(config, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if config.batch_cooldown_ms < 0:
        raise ValueError("--batch-cooldown-ms must be non-negative")


def _view(dataset: Dataset[Any], limit: int | None) -> Dataset[Any]:
    return dataset if limit is None or limit >= len(dataset) else Subset(dataset, range(limit))


def build_loaders(
    config: TrainConfig, device: torch.device
) -> tuple[ISICDataset, dict[str, DataLoader[Any]]]:
    """三个加载器始终读取同一个 ``splits.json``。"""

    limits = {
        "train": config.max_train_samples,
        "val": config.max_val_samples,
        "test": config.max_test_samples,
    }
    datasets = {
        split: ISICDataset(
            config.data_root,
            split=split,
            image_size=config.image_size,
            augment=split == "train",
            seed=config.seed,
            splits_path=config.splits_path,
            require_standard_split_counts=(False if config.splits_path else None),
            # 数据完整性由 prepare_data.py 一次性检查；每次实验不重复打开全部文件。
            validate_files=False,
        )
        for split in limits
    }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        split: DataLoader(
            _view(dataset, limits[split]),
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker if config.num_workers else None,
            generator=generator if split == "train" else None,
            # 重建 worker，确保 set_epoch() 的增强轮次生效。
            persistent_workers=False,
        )
        for split, dataset in datasets.items()
    }
    return datasets["train"], loaders


def _make_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class ModelEMA:
    """Maintain one deployable exponential-moving-average copy of a model."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Avoid a long-lived initialization bias during the first few updates.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            current = source[name].detach()
            if value.is_floating_point():
                value.mul_(decay).add_(current, alpha=1 - decay)
            else:
                value.copy_(current)


def auxiliary_ramp(epoch: int, warmup_epochs: int) -> float:
    """Linearly introduce auxiliary gradients, then keep their full weight."""

    if epoch < 0 or warmup_epochs < 0:
        raise ValueError("epoch and warmup_epochs must be non-negative")
    if warmup_epochs == 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


def auxiliary_schedule(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    retire_start: float,
    retire_end: float,
    *,
    retire: bool,
) -> float:
    """先线性引入辅助监督；实验候选可再用余弦曲线将其退役。"""

    ramp = auxiliary_ramp(epoch, warmup_epochs)
    if not retire:
        return ramp
    progress = epoch / max(total_epochs, 1)
    if progress <= retire_start:
        return ramp
    if progress >= retire_end:
        return 0.0
    phase = (progress - retire_start) / (retire_end - retire_start)
    return ramp * 0.5 * (1.0 + math.cos(math.pi * phase))


def gcsas_auxiliary_schedule(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    start_epochs: int,
    retire_start: float,
    retire_end: float,
) -> float:
    """Delay GCSAS until segmentation has formed, then ramp and retire it."""

    if start_epochs < 0:
        raise ValueError("GCSAS start epochs must be non-negative")
    if start_epochs >= total_epochs:
        return 0.0
    effective_epoch = max(0, epoch - start_epochs)
    effective_total = total_epochs - start_epochs
    return auxiliary_schedule(
        effective_epoch,
        effective_total,
        warmup_epochs,
        retire_start,
        retire_end,
        retire=True,
    )


def absolute_auxiliary_schedule(
    epoch: int,
    warmup_epochs: int,
    retire_start_epoch: int,
    retire_end_epoch: int,
    *,
    delay_epochs: int = 0,
) -> float:
    """按固定轮次引入并退役辅助监督，与总训练轮数解耦。"""

    if min(epoch, warmup_epochs, retire_start_epoch, retire_end_epoch, delay_epochs) < 0:
        raise ValueError("absolute auxiliary schedule values must be non-negative")
    if retire_start_epoch >= retire_end_epoch:
        raise ValueError("absolute auxiliary retirement start must precede its end")
    ramp = auxiliary_ramp(max(0, epoch - delay_epochs), warmup_epochs)
    if epoch <= retire_start_epoch:
        return ramp
    if epoch >= retire_end_epoch:
        return 0.0
    phase = (epoch - retire_start_epoch) / (retire_end_epoch - retire_start_epoch)
    return ramp * 0.5 * (1.0 + math.cos(math.pi * phase))


def _loss_stats(losses: Mapping[str, Tensor | list[Tensor]]) -> dict[str, float]:
    stats = {
        name: float(value.detach())
        for name in (
            "total_loss",
            "seg_loss",
            "boundary_loss",
            "texture_loss",
            "ambiguity_loss",
            "weighted_ambiguity_loss",
            "shape_loss",
            "sdf_loss",
            "distill_loss",
            "edge_loss",
            "deep_supervision_loss",
            "weighted_deep_supervision_loss",
            "weighted_aux_loss",
            "aux_weight_scale",
        )
        if isinstance((value := losses[name]), Tensor)
    }
    for family in ("boundary", "texture", "ambiguity", "shape", "sdf"):
        scales = losses[f"{family}_scale_losses"]
        assert isinstance(scales, list)
        for i in range(3):
            stats[f"{family}_loss_e{i + 1}"] = (
                float(scales[i].detach()) if i < len(scales) else 0.0
            )
    return stats


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    config: TrainConfig,
    *,
    amp: bool,
    auxiliary_scale: float = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    teacher: nn.Module | None = None,
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    """训练和验证共用一次前向流程；有 optimizer 时执行反向传播。"""

    training = optimizer is not None
    model.train(training)
    use_aux = isinstance(model, TriFeatureUNet) or bool(
        getattr(model, "supports_auxiliary", False)
    )
    # Hierarchical KD still needs the student's decoder logits after the
    # semantic/deep auxiliary objectives have retired to zero.
    distillation_active = training and teacher is not None and config.lambda_distill > 0
    output_aux = use_aux and (
        (training and (auxiliary_scale > 0 or distillation_active))
        or config.aux_diagnostics
    )
    target_mode = "gcsas" if config.model == "gcsas_unet" else "legacy"
    scale_weights = (0.5, 0.3, 0.2) if config.model == "gcsas_unet" else None
    auxiliary_multiplier = (
        config.gcsas_aux_multiplier if config.model == "gcsas_unet" else 1.0
    )
    auxiliary_cap = (
        min(config.max_aux_to_seg_ratio, config.gcsas_max_aux_to_seg_ratio)
        if config.model == "gcsas_unet"
        else config.max_aux_to_seg_ratio
    )
    totals: dict[str, float] = {}
    count = 0
    metrics = None if training else MetricAccumulator(
        threshold=config.threshold,
        from_logits=True,
        boundary_tolerance=config.boundary_tolerance,
        boundary_distance=config.boundary_distance,
    )
    boundary_sum = [0.0] * 3
    boundary_count = [0] * 3
    texture_error = [0.0] * 3
    texture_count = [0.0] * 3

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)  # [B,3,H,W]
        masks = batch["mask"].to(device, non_blocking=True)  # [B,1,H,W]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=amp
        ):
            outputs = model(images, return_aux=output_aux)
            if config.model in {
                "lite_sdf_distill",
                "ultralite_sdf_distill",
                "nanolite_sdf_distill",
                "microlite_sdf_distill",
                "picolite_sdf_distill",
                "microlite_fullpath_distill",
                "picolite_fullpath_distill",
                "picolite_selective_fullpath_distill",
                "picolite_shallow_semantic_distill",
                "picolite_learned_head_fullpath_distill",
                "picolite_encoder_semantic_distill",
                "performance_context",
                "compressed_context",
                "compressed_context_fullpath",
                "ambiguity_context",
            } and output_aux:
                teacher_logits = None
                teacher_outputs = None
                if training and teacher is not None and config.lambda_distill > 0:
                    with torch.no_grad():
                        if config.model in {
                            "compressed_context",
                            "compressed_context_fullpath",
                            "ambiguity_context",
                            "picolite_selective_fullpath_distill",
                        }:
                            teacher_outputs = teacher(images, return_aux=True)
                            teacher_logits = teacher_outputs["seg_logits"].detach()
                        else:
                            teacher_logits = teacher(images, return_aux=False)[
                                "seg_logits"
                            ].detach()
                shared = dict(
                    teacher_logits=teacher_logits,
                    lambda_boundary=config.lambda_boundary,
                    lambda_texture=config.lambda_texture,
                    lambda_sdf=config.lambda_sdf,
                    lambda_distill=config.lambda_distill if training else 0.0,
                    lambda_edge=config.lambda_edge if training else 0.0,
                    max_aux_to_seg_ratio=auxiliary_cap,
                    auxiliary_outer_scale=auxiliary_scale,
                    distill_temperature=config.distill_temperature,
                )
                if config.model in {
                    "performance_context",
                    "compressed_context",
                    "compressed_context_fullpath",
                    "ambiguity_context",
                    "microlite_fullpath_distill",
                    "picolite_fullpath_distill",
                    "picolite_selective_fullpath_distill",
                    "picolite_shallow_semantic_distill",
                    "picolite_learned_head_fullpath_distill",
                    "picolite_encoder_semantic_distill",
                }:
                    losses = compute_performance_context_loss(
                        outputs,
                        masks,
                        images,
                        teacher_outputs=teacher_outputs,
                        lambda_deep_supervision=config.lambda_deep_supervision,
                        max_deep_to_seg_ratio=config.max_deep_to_seg_ratio,
                        lambda_ambiguity=config.lambda_ambiguity,
                        max_ambiguity_to_seg_ratio=config.max_ambiguity_to_seg_ratio,
                        distill_boundary_boost=config.distill_boundary_boost,
                        **shared,
                    )
                else:
                    losses = compute_lite_sdf_distill_loss(
                        outputs,
                        masks,
                        images,
                        **shared,
                    )
            else:
                losses = compute_total_loss(
                    outputs,
                    masks,
                    images,
                    lambda_boundary=(
                        config.lambda_boundary * auxiliary_scale * auxiliary_multiplier
                    ),
                    lambda_texture=(
                        config.lambda_texture * auxiliary_scale * auxiliary_multiplier
                    ),
                    lambda_shape=(
                        config.lambda_shape * auxiliary_scale * auxiliary_multiplier
                    ),
                    max_aux_to_seg_ratio=auxiliary_cap,
                    target_mode=target_mode,
                    scale_weights=scale_weights,
                )
            total = losses["total_loss"]
            assert isinstance(total, Tensor)
        if not torch.isfinite(total).item():
            raise FloatingPointError("Non-finite loss encountered")
        if training:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            if config.batch_cooldown_ms:
                time.sleep(config.batch_cooldown_ms / 1000.0)

        batch_size = len(images)
        for name, value in _loss_stats(losses).items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        count += batch_size

        if not training:
            assert metrics is not None
            metrics.update(outputs["seg_logits"].float(), masks)
            if output_aux:
                boundary_logits = outputs["boundary_logits"]
                texture_logits = outputs["texture_logits"]
                assert isinstance(boundary_logits, list) and isinstance(texture_logits, list)
                targets = (
                    make_gcsas_targets(masks, images, boundary_logits)
                    if target_mode == "gcsas"
                    else make_multiscale_targets(masks, images, boundary_logits)
                )
                for i, (boundary, texture, target) in enumerate(
                    zip(boundary_logits, texture_logits, targets)
                ):
                    tolerance = max(
                        1,
                        round(
                            config.boundary_tolerance
                            * boundary.shape[-1]
                            / masks.shape[-1]
                        ),
                    )
                    scores = boundary_f1_score(
                        boundary.float(),
                        target["boundary"],
                        tolerance=tolerance,
                        distance=config.boundary_distance,
                        threshold=config.threshold,
                        from_logits=True,
                        reduction="none",
                    )
                    boundary_sum[i] += float(scores.sum())
                    boundary_count[i] += scores.numel()
                    valid = target["texture_valid"].float()
                    absolute_error = (
                        torch.sigmoid(texture.float()) - target["texture"]
                    ).abs()
                    texture_error[i] += float((absolute_error * valid).sum())
                    texture_count[i] += float(valid.sum())

    if not count:
        raise RuntimeError("empty data loader")
    result = {name: value / count for name, value in totals.items()}
    if training:
        return result

    assert metrics is not None
    result.update(metrics.compute())
    boundary_scores = [
        boundary_sum[i] / boundary_count[i] if boundary_count[i] else 0.0
        for i in range(3)
    ]
    texture_scores = [
        texture_error[i] / texture_count[i] if texture_count[i] else 0.0
        for i in range(3)
    ]
    for i in range(3):
        result[f"boundary_head_bf1_e{i + 1}"] = boundary_scores[i]
        result[f"texture_head_mae_e{i + 1}"] = texture_scores[i]
    result["boundary_head_bf1"] = sum(boundary_scores) / 3 if output_aux else 0.0
    result["texture_head_mae"] = sum(texture_scores) / 3 if output_aux else 0.0
    return result


def _convergence_point(
    history: list[dict[str, float | int]], fraction: float, best_dice: float
) -> tuple[int, float]:
    """返回首次达到自身最佳验证 Dice 某一比例的 epoch 和累计秒数。"""

    target = fraction * best_dice
    for row in history:
        if float(row["val_dice"]) >= target:
            return int(row["epoch"]), float(row["cumulative_seconds"])
    return 0, float("nan")


def run_training(config: TrainConfig) -> dict[str, Any]:
    """仅按验证 Dice 选优，训练结束后才评估测试集。"""

    _validate(config)
    output = config.output_dir
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.is_dir() and any(output.iterdir()) and not config.allow_existing_output:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(config.device)
    amp = config.amp and device.type == "cuda"
    if config.amp and not amp:
        print("Warning: AMP requires CUDA; using FP32.")
    resolved_name = device_name(device)
    cuda_version = torch.version.cuda if device.type == "cuda" else "n/a"
    print(
        f"device={device} name={resolved_name} "
        f"cuda={cuda_version} amp={'on' if amp else 'off'}"
    )

    train_dataset, loaders = build_loaders(config, device)
    model_kwargs = (
        {} if config.base_channels is None else {"base_channels": config.base_channels}
    )
    model = build_model(config.model, **model_kwargs).to(device)
    if config.initial_checkpoint is not None:
        try:
            initial = torch.load(
                config.initial_checkpoint, map_location="cpu", weights_only=False
            )
        except TypeError:
            initial = torch.load(config.initial_checkpoint, map_location="cpu")
        state = initial.get("model_state_dict") if isinstance(initial, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("initial checkpoint must contain model_state_dict")
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = (
            ("ambiguity_auxiliary.",) if config.model == "ambiguity_context" else ()
        )
        invalid_missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith(allowed_missing)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise ValueError(
                "initial checkpoint is incompatible: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        print(
            f"initialized_from={config.initial_checkpoint} "
            f"new_parameters={len(incompatible.missing_keys)}"
        )
    base_channels = model.backbone.base_channels
    trainable_parameters = count_trainable_parameters(model)
    inference_parameters = count_inference_parameters(model)
    print(
        f"model={config.model} base_channels={base_channels} "
        f"trainable_parameters={trainable_parameters} "
        f"inference_parameters={inference_parameters}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    scaler = _make_scaler(amp)
    teacher: nn.Module | None = None
    ema: ModelEMA | None = None
    if config.model in {
        "lite_sdf_distill",
        "ultralite_sdf_distill",
        "nanolite_sdf_distill",
        "microlite_sdf_distill",
        "picolite_sdf_distill",
        "microlite_fullpath_distill",
        "picolite_fullpath_distill",
        "picolite_selective_fullpath_distill",
        "picolite_shallow_semantic_distill",
        "picolite_learned_head_fullpath_distill",
        "picolite_encoder_semantic_distill",
        "performance_context",
        "compressed_context",
        "compressed_context_fullpath",
        "ambiguity_context",
    }:
        from evaluate import load_model_from_checkpoint

        teacher_metadata = None
        if config.teacher_checkpoint is not None:
            try:
                raw_teacher = torch.load(
                    config.teacher_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                raw_teacher = torch.load(config.teacher_checkpoint, map_location="cpu")
            if isinstance(raw_teacher, Mapping) and raw_teacher.get("kind") in {
                "directional_ambiguity_refiner",
                "hybrid_directional_shape_refiner",
            }:
                from src.directional_refiner import load_directional_refiner_checkpoint

                teacher, directional_metadata = load_directional_refiner_checkpoint(
                    config.teacher_checkpoint, device
                )
                teacher_metadata = {
                    "model_name": str(raw_teacher["kind"]),
                    "epoch": directional_metadata.get("epoch"),
                }
            else:
                teacher, _, teacher_metadata = load_model_from_checkpoint(
                    config.teacher_checkpoint, device
                )
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        ema = ModelEMA(model, config.ema_decay)
        teacher_description = (
            "none"
            if teacher_metadata is None
            else f"{teacher_metadata['model_name']}@{teacher_metadata['epoch']}"
        )
        print(f"teacher={teacher_description} ema_decay={config.ema_decay}")
    payload = asdict(config) | {
        "data_root": str(config.data_root),
        "output_dir": str(output),
        "splits_path": None if config.splits_path is None else str(config.splits_path),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "device_name": resolved_name,
        "resolved_device": str(device),
        "amp_enabled": amp,
        "base_channels": base_channels,
        "trainable_parameters": trainable_parameters,
        "inference_parameters": inference_parameters,
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    save_json(payload, output / "config.json")

    started = time.perf_counter()
    best = float("-inf")
    stale = 0
    completed_epoch = 0
    history: list[dict[str, float | int]] = []
    if config.initial_checkpoint is not None:
        initial_validation_model = ema.model if ema is not None else model
        initial_stats = _run_epoch(
            initial_validation_model,
            loaders["val"],
            device,
            config,
            amp=amp,
            auxiliary_scale=0.0,
        )
        best = initial_stats["dice"]
        history.append({"epoch": 0, "val_dice": best, "cumulative_seconds": 0.0})
        scheduler.step(best)
        initial_best = {
            "model_name": config.model,
            "model_state_dict": initial_validation_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": 0,
            "best_val_dice": best,
            "config": payload,
            "seed": config.seed,
            "training_elapsed_seconds": 0.0,
        }
        torch.save(initial_best, output / "best.pt")
        print(
            f"epoch=000 initialized_val_dice={best:.5f} "
            f"val_bf1={initial_stats['bf1']:.5f}"
        )
    with (output / "train_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            train_dataset.set_epoch(epoch - 1)
            epoch_started = time.perf_counter()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            if config.model in BASELINE_MODEL_NAMES:
                auxiliary_scale = 0.0
            elif config.model == "picolite_selective_fullpath_distill":
                # Semantic routes establish an inductive bias early, then fully
                # retire so the final low-LR phase can focus on segmentation and
                # hierarchical teacher transfer.  Plateau-driven decay may lower
                # the pressure even earlier when the validation curve stalls.
                plateau_scale = math.sqrt(learning_rate / config.learning_rate)
                retirement_scale = auxiliary_schedule(
                    epoch,
                    config.epochs,
                    config.aux_warmup_epochs,
                    config.aux_retire_start,
                    config.aux_retire_end,
                    retire=True,
                )
                auxiliary_scale = min(plateau_scale, retirement_scale)
            elif config.model in {
                "lite_sdf_distill",
                "ultralite_sdf_distill",
                "nanolite_sdf_distill",
                "microlite_sdf_distill",
                "picolite_sdf_distill",
                "microlite_fullpath_distill",
                "picolite_fullpath_distill",
                "picolite_shallow_semantic_distill",
                "picolite_learned_head_fullpath_distill",
                "picolite_encoder_semantic_distill",
                "performance_context",
                "compressed_context",
                "compressed_context_fullpath",
                "ambiguity_context",
            }:
                # Reduce auxiliary pressure only after validation plateaus have
                # caused LR reductions; retain a semantic floor throughout.
                plateau_scale = max(
                    config.auxiliary_floor,
                    math.sqrt(learning_rate / config.learning_rate),
                )
                auxiliary_scale = (
                    auxiliary_ramp(epoch, config.aux_warmup_epochs)
                    * plateau_scale
                )
            elif config.model == "gcsas_unet":
                if config.aux_retire_start_epoch is not None:
                    assert config.aux_retire_end_epoch is not None
                    auxiliary_scale = absolute_auxiliary_schedule(
                        epoch,
                        config.aux_warmup_epochs,
                        config.aux_retire_start_epoch,
                        config.aux_retire_end_epoch,
                        delay_epochs=config.gcsas_aux_start_epochs,
                    )
                else:
                    auxiliary_scale = gcsas_auxiliary_schedule(
                        epoch,
                        config.epochs,
                        config.aux_warmup_epochs,
                        config.gcsas_aux_start_epochs,
                        config.aux_retire_start,
                        config.aux_retire_end,
                    )
            else:
                retire = config.model in RETIRING_AUXILIARY_MODEL_NAMES
                if retire and config.aux_retire_start_epoch is not None:
                    assert config.aux_retire_end_epoch is not None
                    auxiliary_scale = absolute_auxiliary_schedule(
                        epoch,
                        config.aux_warmup_epochs,
                        config.aux_retire_start_epoch,
                        config.aux_retire_end_epoch,
                    )
                else:
                    auxiliary_scale = auxiliary_schedule(
                        epoch,
                        config.epochs,
                        config.aux_warmup_epochs,
                        config.aux_retire_start,
                        config.aux_retire_end,
                        retire=retire,
                    )
            train_started = time.perf_counter()
            train_stats = _run_epoch(
                model,
                loaders["train"],
                device,
                config,
                amp=amp,
                auxiliary_scale=auxiliary_scale,
                optimizer=optimizer,
                scaler=scaler,
                teacher=teacher,
                ema=ema,
            )
            train_seconds = time.perf_counter() - train_started
            validation_started = time.perf_counter()
            validation_model = ema.model if ema is not None else model
            val_stats = _run_epoch(
                validation_model,
                loaders["val"],
                device,
                config,
                amp=amp,
                auxiliary_scale=auxiliary_scale,
            )
            validation_seconds = time.perf_counter() - validation_started
            val_dice = val_stats["dice"]
            scheduler.step(val_dice)
            completed_epoch = epoch
            improved = val_dice > best
            best, stale = (val_dice, 0) if improved else (best, stale + 1)

            epoch_seconds = time.perf_counter() - epoch_started
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "epoch_seconds": epoch_seconds,
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "cumulative_seconds": time.perf_counter() - started,
                "auxiliary_scale": auxiliary_scale,
                **{f"train_{name}": train_stats[name] for name in LOSS_FIELDS},
                **{f"val_{name}": val_stats[name] for name in VAL_FIELDS},
            }
            writer.writerow(row)
            handle.flush()
            history.append(
                {
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "cumulative_seconds": row["cumulative_seconds"],
                }
            )
            checkpoint = {
                "model_name": config.model,
                # EMA is the only student copy used for selection and deployment.
                "model_state_dict": validation_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "best_val_dice": best,
                "config": payload,
                "seed": config.seed,
                "training_elapsed_seconds": time.perf_counter() - started,
            }
            torch.save(checkpoint, output / "last.pt")
            if improved:
                torch.save(checkpoint, output / "best.pt")
            print(
                f"epoch={epoch:03d} train_loss={train_stats['total_loss']:.5f} "
                f"val_loss={val_stats['total_loss']:.5f} "
                f"val_dice={val_dice:.5f} "
                f"val_accuracy={val_stats['accuracy']:.5f} "
                f"val_bf1={val_stats['bf1']:.5f} "
                f"lr={learning_rate:.3g}"
            )
            if stale >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    elapsed = time.perf_counter() - started
    seconds_per_epoch = elapsed / completed_epoch
    best_epoch = int(max(history, key=lambda row: float(row["val_dice"]))["epoch"])
    epoch90, seconds90 = _convergence_point(history, 0.90, best)
    epoch95, seconds95 = _convergence_point(history, 0.95, best)
    payload |= {
        "training_elapsed_seconds": elapsed,
        "epochs_completed": completed_epoch,
        "seconds_per_epoch": seconds_per_epoch,
        "best_val_dice": best,
        "best_epoch": best_epoch,
        "epochs_to_90pct_own_best": epoch90,
        "seconds_to_90pct_own_best": seconds90,
        "epochs_to_95pct_own_best": epoch95,
        "seconds_to_95pct_own_best": seconds95,
    }
    save_json(payload, output / "config.json")
    try:
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(output / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    from evaluate import (
        benchmark_inference,
        evaluate_model,
        save_fixed_predictions,
        save_metrics_csv,
    )

    metrics = evaluate_model(
        model,
        loaders["test"],
        device,
        threshold=config.threshold,
        boundary_tolerance=config.boundary_tolerance,
        boundary_distance=config.boundary_distance,
        return_aux=False,
    )
    benchmark = benchmark_inference(
        model,
        loaders["test"],
        device,
        warmup_batches=config.benchmark_warmup_batches,
        benchmark_batches=config.benchmark_batches,
        repetitions=config.benchmark_repetitions,
        return_aux=False,
    )
    metrics |= {
        "model": config.model,
        "seed": config.seed,
        "split": "test",
        "checkpoint": str(output / "best.pt"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_dice": best,
        "best_epoch": best_epoch,
        "epochs_to_90pct_own_best": epoch90,
        "seconds_to_90pct_own_best": seconds90,
        "epochs_to_95pct_own_best": epoch95,
        "seconds_to_95pct_own_best": seconds95,
        "base_channels": base_channels,
        "trainable_parameters": trainable_parameters,
        "inference_parameters": inference_parameters,
        "inference_model_size_mb_fp32": inference_parameters * 4 / 1024**2,
        "device": str(device),
        "device_name": resolved_name,
        "threshold": config.threshold,
        "boundary_tolerance": config.boundary_tolerance,
        "boundary_distance": config.boundary_distance,
        "training_elapsed_seconds": elapsed,
        "epochs_completed": completed_epoch,
        "seconds_per_epoch": seconds_per_epoch,
        **benchmark,
    }
    payload |= benchmark
    save_json(payload, output / "config.json")
    save_json(metrics, output / "test_metrics.json")
    save_metrics_csv(metrics, output / "test_metrics.csv")
    if config.num_predictions:
        save_fixed_predictions(
            model,
            loaders["test"].dataset,
            device,
            output / "predictions",
            count=config.num_predictions,
            threshold=config.threshold,
            return_aux=False,
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        metrics = run_training(TrainConfig(**vars(parser.parse_args(argv))))
    except (FileNotFoundError, FileExistsError, NotADirectoryError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"test dice={metrics['dice']:.5f} iou={metrics['iou']:.5f} "
        f"accuracy={metrics['accuracy']:.5f} bf1={metrics['bf1']:.5f}"
    )
    print(
        f"training={metrics['training_elapsed_seconds']:.2f}s "
        f"({metrics['seconds_per_epoch']:.2f}s/epoch) "
        f"inference={metrics['inference_ms_per_image']:.3f}ms/image "
        f"({metrics['inference_images_per_second']:.2f} images/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
