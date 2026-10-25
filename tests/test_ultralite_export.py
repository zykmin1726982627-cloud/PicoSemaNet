from __future__ import annotations

from pathlib import Path

import torch

from export_ultralite import export_ultralite, load_ultralite_deploy
from src.models import build_model


def test_ultralite_export_is_independent_and_below_half_mib(tmp_path: Path) -> None:
    model = build_model("ultralite_sdf_distill", base_channels=18).eval()
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {
            "model_name": "ultralite_sdf_distill",
            "model_state_dict": model.state_dict(),
            "config": {"base_channels": 18, "image_size": 32},
            "seed": 42,
        },
        checkpoint,
    )
    output = tmp_path / "deploy.pt"
    result = export_ultralite(checkpoint, output)
    restored, metadata = load_ultralite_deploy(output)

    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        expected = model(image, return_aux=False)["seg_logits"]
        actual = restored(image)
    assert output.stat().st_size <= 512 * 1024
    assert result["parameters"] == 121_063
    assert metadata["dtype"] == "float16"
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_fullpath_microlite_export_preserves_custom_core(tmp_path: Path) -> None:
    model = build_model("microlite_fullpath_distill", base_channels=6).eval()
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {
            "model_name": "microlite_fullpath_distill",
            "model_state_dict": model.state_dict(),
            "config": {"base_channels": 6, "image_size": 32},
            "seed": 42,
        },
        checkpoint,
    )
    output = tmp_path / "deploy.pt"
    result = export_ultralite(checkpoint, output)
    restored, metadata = load_ultralite_deploy(output)

    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        expected = model(image, return_aux=False)["seg_logits"]
        actual = restored(image)
    assert result["parameters"] == 9_824
    assert metadata["channel_schedule"] == [6, 10, 14, 18, 20]
    assert metadata["refine_output"] is True
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)
