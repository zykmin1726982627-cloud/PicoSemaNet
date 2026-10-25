import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluate import benchmark_inference, build_parser, save_metrics_csv


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[bool] = []

    def forward(
        self, images: torch.Tensor, return_aux: bool = False
    ) -> dict[str, torch.Tensor]:
        self.calls.append(return_aux)
        return {"seg_logits": images[:, :1]}


def _loader(count: int = 5, batch_size: int = 2) -> DataLoader:
    samples = [
        {
            "image": torch.zeros(3, 16, 16),
            "mask": torch.zeros(1, 16, 16),
            "id": f"sample_{index}",
        }
        for index in range(count)
    ]
    return DataLoader(samples, batch_size=batch_size, shuffle=False)


def test_cpu_benchmark_reports_forward_only_workload_and_restores_mode() -> None:
    model = RecordingModel()
    model.train()

    result = benchmark_inference(
        model,
        _loader(),
        "cpu",
        warmup_batches=2,
        benchmark_batches=2,
        repetitions=3,
    )

    assert model.training
    assert model.calls == [False] * 8  # 2 warmup + 2 batches * 3 repetitions
    assert result["benchmark_scope"] == "model_forward_only"
    assert result["benchmark_device"] == "cpu"
    assert result["benchmark_warmup_batches"] == 2
    assert result["benchmark_batches"] == 2
    assert result["benchmark_repetitions"] == 3
    assert result["benchmark_batch_size"] == 2
    assert result["benchmark_image_height"] == 16
    assert result["benchmark_image_width"] == 16
    assert result["inference_num_images"] == 12
    assert float(result["inference_total_seconds"]) > 0
    assert float(result["inference_ms_per_image"]) == pytest.approx(
        float(result["inference_total_seconds"]) * 1000 / 12
    )
    assert float(result["inference_images_per_second"]) == pytest.approx(
        12 / float(result["inference_total_seconds"])
    )
    assert all(
        math.isfinite(float(result[name]))
        for name in (
            "inference_total_seconds",
            "inference_ms_per_image",
            "inference_images_per_second",
        )
    )


def test_benchmark_can_explicitly_measure_auxiliary_path() -> None:
    model = RecordingModel()

    result = benchmark_inference(
        model,
        _loader(count=1, batch_size=1),
        "cpu",
        warmup_batches=0,
        benchmark_batches=1,
        repetitions=1,
        return_aux=True,
    )

    assert model.calls == [True]
    assert result["benchmark_auxiliary_outputs_enabled"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmup_batches": -1}, "warmup_batches"),
        ({"benchmark_batches": 0}, "benchmark_batches"),
        ({"repetitions": 0}, "repetitions"),
    ],
)
def test_benchmark_rejects_invalid_controls(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_inference(RecordingModel(), _loader(), "cpu", **kwargs)


def test_benchmark_rejects_empty_loader() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        benchmark_inference(
            RecordingModel(),
            _loader(count=0),
            "cpu",
            warmup_batches=0,
            benchmark_batches=1,
            repetitions=1,
        )


def test_evaluation_cli_exposes_reproducible_benchmark_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "best.pt",
            "--data-root",
            "data/ISIC2018",
            "--benchmark-warmup-batches",
            "7",
            "--benchmark-batches",
            "11",
            "--benchmark-repetitions",
            "4",
        ]
    )

    assert args.benchmark_warmup_batches == 7
    assert args.benchmark_batches == 11
    assert args.benchmark_repetitions == 4
    assert not args.with_aux


def test_single_run_metrics_are_saved_as_one_csv_row(tmp_path) -> None:
    output = save_metrics_csv(
        {
            "model_name": "unet",
            "seed": 42,
            "split": "test",
            "dice": 0.8,
            "iou": 0.7,
            "accuracy": 0.9,
            "bf1": 0.6,
            "training_elapsed_seconds": 120.0,
            "inference_ms_per_image": 5.0,
        },
        tmp_path / "test_metrics.csv",
    )

    import csv

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["model"] == "unet"
    assert rows[0]["accuracy"] == "0.9"
    assert rows[0]["training_elapsed_seconds"] == "120.0"
    assert rows[0]["inference_ms_per_image"] == "5.0"
