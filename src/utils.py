"""Shared reproducibility, device, parameter-count, and JSON utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and every available PyTorch device."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_: int) -> None:
    """Seed a DataLoader worker from the seed assigned by PyTorch."""

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(requested: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto``/CPU/MPS/CUDA and fail early on invalid CUDA requests."""

    if isinstance(requested, torch.device):
        device = requested
    elif requested is None or str(requested).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        return torch.device("mps" if mps is not None and mps.is_available() else "cpu")
    else:
        device = torch.device(str(requested).lower())
    if device.type == "cuda":
        if torch.version.cuda is None:
            raise RuntimeError(
                "CUDA was requested, but the installed PyTorch build is CPU-only "
                "(torch.version.cuda is None). Install a CUDA-enabled PyTorch build."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA was requested and PyTorch was built for CUDA {torch.version.cuda}, "
                "but no usable NVIDIA GPU was found. Check the NVIDIA driver."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device.index} does not exist; "
                f"available device count: {torch.cuda.device_count()}."
            )
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    return device


def device_name(device: str | torch.device) -> str:
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    return "Apple Metal (MPS)" if device.type == "mps" else "CPU"


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_inference_parameters(model: nn.Module) -> int:
    """Count final-path parameters, excluding TriFeatureUNet training-only heads."""

    if bool(getattr(model, "inference_uses_auxiliary", False)):
        return count_trainable_parameters(model)
    # 辅助头只服务训练监督，正式分割推理不执行它们。
    auxiliary = getattr(
        model,
        "training_only_parameter_prefixes",
        ("boundary_auxiliary.", "texture_auxiliary.", "shape_auxiliary.", "sdf_auxiliary."),
    )
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(auxiliary)
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            data,
            handle,
            default=_json_default,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        handle.write("\n")
    return output


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
