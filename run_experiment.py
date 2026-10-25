"""Launch one archived experiment using portable, explicit paths."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("data_root", "output_dir", "teacher_checkpoint", "device", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value.resolve()) if isinstance(value, Path) else value
    for key in ("data_root", "output_dir", "splits_path", "teacher_checkpoint", "initial_checkpoint"):
        if config.get(key) is not None:
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else ROOT / value)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    from train import TrainConfig, run_training
    for key in ("data_root", "output_dir", "splits_path", "teacher_checkpoint", "initial_checkpoint"):
        if config.get(key) is not None:
            config[key] = Path(config[key])
    run_training(TrainConfig(**config))

if __name__ == "__main__":
    main()
