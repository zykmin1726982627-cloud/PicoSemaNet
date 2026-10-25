"""严格校验 ISIC 2018 Task 1 数据并创建共享固定划分。"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Sequence

from src.dataset import (
    DEFAULT_SPLIT_SEED,
    EXPECTED_PAIR_COUNT,
    SPLIT_SIZES,
    DatasetValidationError,
    create_fixed_splits,
    discover_pairs,
    load_splits,
    save_splits,
)


def prepare_dataset(
    data_root: str | Path,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    overwrite_split: bool = False,
) -> Path:
    """校验全部配对，并创建或复用 ``splits.json``。"""

    root = Path(data_root)
    pairs = discover_pairs(root, validate_readable=True)
    pair_count = len(pairs)
    print(f"Validated {pair_count} image/mask pairs in {root}.")
    if pair_count != EXPECTED_PAIR_COUNT:
        warnings.warn(
            f"ISIC 2018 Task 1 normally contains {EXPECTED_PAIR_COUNT} pairs, "
            f"but found {pair_count}. The fixed {SPLIT_SIZES} partition cannot "
            "be generated from an incomplete or unexpected dataset.",
            RuntimeWarning,
            stacklevel=2,
        )

    split_path = root / "splits.json"
    if split_path.exists() and not overwrite_split:
        splits = load_splits(
            split_path, expected_ids=pairs, require_standard_counts=True
        )
        counts = {name: len(ids) for name, ids in splits.items()}
        print(
            f"Using existing split file without modifying it: {split_path} "
            f"(counts: {counts})."
        )
        return split_path

    if split_path.exists():
        print(
            f"WARNING: --overwrite-split was supplied; replacing {split_path}. "
            "All experiments must now use this newly generated partition.",
            file=sys.stderr,
        )
    splits = create_fixed_splits(pairs, seed=seed)
    save_splits(split_path, splits, overwrite=overwrite_split)
    counts = {name: len(ids) for name, ids in splits.items()}
    print(f"Wrote fixed split to {split_path} with seed {seed}: {counts}.")
    return split_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ISIC 2018 Task 1 image/mask pairs and create the shared "
            "2075/259/260 splits.json file."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing images/ and masks/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"One-time split seed (default: {DEFAULT_SPLIT_SEED}).",
    )
    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Explicitly replace an existing splits.json (normally forbidden).",
    )
    args = parser.parse_args(argv)
    try:
        prepare_dataset(
            args.data_root,
            seed=args.seed,
            overwrite_split=args.overwrite_split,
        )
    except (DatasetValidationError, FileNotFoundError, OSError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
