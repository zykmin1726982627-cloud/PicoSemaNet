"""Tiny end-to-end checks for the shared training entry point."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import train as training
from src.losses import compute_total_loss
from src.models import build_model
from train import TrainConfig, build_parser, run_training


def _tiny_isic_dataset(root: Path) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    ids = [f"ISIC_{index:07d}" for index in range(6)]
    for index, image_id in enumerate(ids):
        mask = np.zeros((32, 32), dtype=np.uint8)
        start = 5 + index % 3
        mask[start : start + 16, 7:23] = 255
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = 30 + index
        image[..., 1] = 40
        image[..., 2] = 50
        image[mask > 0] = (190, 120, 100)
        Image.fromarray(image).save(images / f"{image_id}.jpg")
        Image.fromarray(mask).save(masks / f"{image_id}_segmentation.png")
    splits = {"train": ids[:4], "val": [ids[4]], "test": [ids[5]]}
    (root / "splits.json").write_text(
        json.dumps(splits), encoding="utf-8"
    )


def test_training_cli_accepts_only_supported_sweep_widths() -> None:
    parser = build_parser()
    required = [
        "--model",
        "unet",
        "--data-root",
        "data/ISIC2018",
        "--output-dir",
        "runs/example",
    ]
    assert parser.parse_args(required).base_channels is None
    assert parser.parse_args([*required, "--base-channels", "12"]).base_channels == 12
    assert parser.parse_args([*required, "--base-channels", "9"]).base_channels == 9
    assert parser.parse_args([*required, "--base-channels", "4"]).base_channels == 4
    assert parser.parse_args([*required, "--base-channels", "20"]).base_channels == 20
    assert parser.parse_args([*required, "--batch-cooldown-ms", "25"]).batch_cooldown_ms == 25
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--base-channels", "10"])


def test_auxiliary_cli_and_warmup_have_safe_defaults_and_legacy_alias() -> None:
    required = [
        "--model",
        "tri_unet",
        "--data-root",
        "data/ISIC2018",
        "--output-dir",
        "runs/example",
    ]
    parser = build_parser()

    current = parser.parse_args(
        [
            *required,
            "--lambda-texture",
            "0.3",
            "--max-aux-to-seg-ratio",
            "0.4",
            "--aux-warmup-epochs",
            "4",
        ]
    )
    legacy = parser.parse_args([*required, "--lambda-region", "0.25"])

    assert current.lambda_texture == pytest.approx(0.3)
    assert current.max_aux_to_seg_ratio == pytest.approx(0.4)
    assert current.aux_warmup_epochs == 4
    assert legacy.lambda_texture == pytest.approx(0.25)
    assert training.auxiliary_ramp(0, 5) == 0
    assert training.auxiliary_ramp(1, 5) == pytest.approx(0.2)
    assert training.auxiliary_ramp(5, 5) == 1
    assert training.auxiliary_ramp(8, 5) == 1
    assert training.auxiliary_ramp(1, 0) == 1


def test_unified_trainer_runs_both_models_and_writes_required_artifacts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "ISIC2018"
    _tiny_isic_dataset(data_root)
    results: dict[str, dict[str, object]] = {}
    expected_widths = {"unet": 32, "tri_unet": 24}

    for model_name in ("unet", "tri_unet"):
        output_dir = tmp_path / model_name
        results[model_name] = run_training(
            TrainConfig(
                model=model_name,
                data_root=data_root,
                output_dir=output_dir,
                image_size=32,
                batch_size=2,
                epochs=1,
                early_stopping_patience=1,
                num_predictions=1,
                benchmark_warmup_batches=0,
                benchmark_batches=1,
                benchmark_repetitions=1,
                device="cpu",
            )
        )
        for name in (
            "config.json",
            "train_log.csv",
            "best.pt",
            "last.pt",
            "test_metrics.json",
            "test_metrics.csv",
        ):
            assert (output_dir / name).is_file()
        assert (output_dir / "predictions" / "sample_001.png").is_file()
        assert results[model_name]["num_images"] == 1

        with (output_dir / "test_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            metric_rows = list(csv.DictReader(handle))
        assert len(metric_rows) == 1
        assert metric_rows[0]["model"] == model_name
        assert int(metric_rows[0]["base_channels"]) == expected_widths[model_name]
        assert metric_rows[0]["split"] == "test"
        assert float(metric_rows[0]["accuracy"]) == pytest.approx(
            float(results[model_name]["accuracy"])
        )
        assert float(metric_rows[0]["training_elapsed_seconds"]) >= 0
        assert float(metric_rows[0]["inference_ms_per_image"]) >= 0

        with (output_dir / "train_log.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert math.isfinite(float(rows[0]["train_total_loss"]))
        assert math.isfinite(float(rows[0]["val_dice"]))
        assert math.isfinite(float(rows[0]["val_accuracy"]))
        if model_name == "tri_unet":
            assert float(rows[0]["train_boundary_loss_e1"]) > 0
            assert float(rows[0]["train_texture_loss_e1"]) > 0
            assert 0 <= float(rows[0]["train_aux_weight_scale"]) <= 1
            assert float(rows[0]["val_texture_head_mae"]) >= 0

        try:
            checkpoint = torch.load(
                output_dir / "best.pt", map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch 2.0 compatibility
            checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
        assert {
            "model_name",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
            "best_val_dice",
            "config",
            "seed",
        } <= checkpoint.keys()
        assert checkpoint["config"]["base_channels"] == expected_widths[model_name]
        assert (
            checkpoint["model_state_dict"]["backbone.encoder1.0.weight"].shape[0]
            == expected_widths[model_name]
        )
        config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        assert config["torch_version"]
        assert config["device_name"] == "CPU"
        assert config["base_channels"] == expected_widths[model_name]
        assert config["training_elapsed_seconds"] >= 0
        assert config["epochs_completed"] == 1
        assert config["seconds_per_epoch"] == pytest.approx(
            config["training_elapsed_seconds"]
        )
        assert config["inference_ms_per_image"] >= 0

        result = results[model_name]
        assert 0 <= float(result["accuracy"]) <= 1
        assert result["epochs_completed"] == 1
        assert float(result["seconds_per_epoch"]) >= 0
        assert float(result["inference_ms_per_image"]) >= 0
        assert float(result["inference_images_per_second"]) > 0
        assert result["benchmark_scope"] == "model_forward_only"
        assert result["benchmark_auxiliary_outputs_enabled"] is False
        assert result["base_channels"] == expected_widths[model_name]

    assert results["unet"]["trainable_parameters"] == 7_763_041
    assert results["unet"]["inference_parameters"] == 7_763_041
    assert results["tri_unet"]["trainable_parameters"] > 4_368_073
    assert results["tri_unet"]["inference_parameters"] == 4_368_073


def test_training_honors_explicit_base_channels(tmp_path: Path) -> None:
    data_root = tmp_path / "ISIC2018"
    _tiny_isic_dataset(data_root)
    output_dir = tmp_path / "unet_base8"

    result = run_training(
        TrainConfig(
            model="unet",
            base_channels=8,
            data_root=data_root,
            output_dir=output_dir,
            image_size=32,
            batch_size=2,
            epochs=1,
            early_stopping_patience=1,
            num_predictions=0,
            benchmark_warmup_batches=0,
            benchmark_batches=1,
            benchmark_repetitions=1,
            device="cpu",
        )
    )

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (output_dir / "test_metrics.json").read_text(encoding="utf-8")
    )
    assert config["base_channels"] == 8
    assert metrics["base_channels"] == 8
    assert result["base_channels"] == 8
    assert result["trainable_parameters"] == 486_553


def test_both_models_overfit_four_deterministic_synthetic_images() -> None:
    """Manual-integration requirement kept small enough for the local suite."""

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        masks = torch.zeros(4, 1, 32, 32)
        for index in range(4):
            start = 5 + index
            masks[index, :, start : start + 16, 7:23] = 1
        images = torch.cat(
            (
                masks * 2 - 1,
                masks * 1.5 - 0.5,
                0.25 * torch.ones_like(masks),
            ),
            dim=1,
        )
        images += 0.03 * torch.randn_like(images)

        for model_name in ("unet", "tri_unet"):
            torch.manual_seed(42)
            model = build_model(model_name, base_channels=4).train()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=3e-3, weight_decay=0.0
            )
            history: list[float] = []
            for _ in range(40):
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images, return_aux=model_name == "tri_unet")
                total = compute_total_loss(outputs, masks, images)["total_loss"]
                history.append(float(total.detach()))
                total.backward()
                optimizer.step()
            assert history[-1] < 0.8 * history[0], (
                model_name,
                history[0],
                history[-1],
            )
    finally:
        torch.set_num_threads(previous_threads)


def test_training_api_rejects_noncanonical_model_alias(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model must be one of"):
        run_training(
            TrainConfig(
                model="tri-unet",
                data_root=tmp_path / "missing-data",
                output_dir=tmp_path / "run",
            )
        )
