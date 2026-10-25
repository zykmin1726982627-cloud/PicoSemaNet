"""Export a trained ultralight model without training-only auxiliary heads."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch

from evaluate import load_model_from_checkpoint
from src.models import UltraLiteCore


FORMAT = "ultralite-core-v1"


def export_ultralite(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    dtype: torch.dtype = torch.float16,
    max_bytes: int = 512 * 1024,
) -> dict[str, Any]:
    """Strip auxiliary heads and save an independently loadable core checkpoint."""

    if dtype not in {torch.float16, torch.float32}:
        raise ValueError("deployment dtype must be float16 or float32")
    model, _, metadata = load_model_from_checkpoint(checkpoint_path, "cpu")
    core = getattr(model, "core", None)
    if not isinstance(core, UltraLiteCore):
        raise ValueError("checkpoint does not contain an UltraLiteCore deployment graph")
    core = core.eval()
    state = {
        name: value.detach().cpu().to(dtype=dtype)
        if value.is_floating_point()
        else value.detach().cpu()
        for name, value in core.state_dict().items()
    }
    payload = {
        "format": FORMAT,
        "dtype": str(dtype).removeprefix("torch."),
        "in_channels": core.in_channels,
        "out_channels": core.out_channels,
        "base_channels": core.base_channels,
        "channel_schedule": list(core.channel_schedule),
        "refine_output": core.refine_output,
        "image_size": int(metadata["config"].get("image_size", 320)),
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "state_dict": state,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    size = output.stat().st_size
    if size > max_bytes:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"export is {size} bytes, exceeding the {max_bytes}-byte deployment budget"
        )
    return {
        "path": str(output.resolve()),
        "bytes": size,
        "mib": size / 1024**2,
        "parameters": sum(parameter.numel() for parameter in core.parameters()),
        "dtype": payload["dtype"],
    }


def load_ultralite_deploy(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[UltraLiteCore, dict[str, Any]]:
    """Load a stripped deployment checkpoint with strict format validation."""

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("format") != FORMAT:
        raise ValueError("invalid ultralite deployment checkpoint")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("deployment checkpoint has no state_dict")
    model = UltraLiteCore(
        in_channels=int(payload["in_channels"]),
        out_channels=int(payload["out_channels"]),
        base_channels=int(payload["base_channels"]),
        channel_schedule=(
            tuple(int(value) for value in payload["channel_schedule"])
            if "channel_schedule" in payload
            else None
        ),
        refine_output=bool(payload.get("refine_output", False)),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-bytes", type=int, default=512 * 1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print(
        export_ultralite(
            args.checkpoint,
            args.output,
            dtype=dtype,
            max_bytes=args.max_bytes,
        )
    )


if __name__ == "__main__":
    main()
