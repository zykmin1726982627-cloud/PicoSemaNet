import pytest
import torch

from src.models import build_model
from src.utils import resolve_device


def test_auto_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_cuda_request_explains_cpu_only_pytorch(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="PyTorch build is CPU-only"):
        resolve_device("cuda")


def test_cuda_request_rejects_invalid_device_index(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="CUDA device 1 does not exist"):
        resolve_device("cuda:1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("model_name", ["unet", "tri_unet"])
def test_model_cuda_optimizer_step(model_name: str) -> None:
    """Exercise one real GPU update for each model on CUDA hosts."""

    model = build_model(model_name, base_channels=4).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 32, 32, device="cuda")

    optimizer.zero_grad(set_to_none=True)
    loss = model(images, return_aux=False)["seg_logits"].square().mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
