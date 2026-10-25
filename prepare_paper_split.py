"""Create the immutable preregistered ISIC 2018 paper split.

The split covers all 2,594 local image/mask pairs with 1,815/259/520 samples.
It is stratified by lesion-area decile and keeps exact/very-near duplicate image
groups together.  Existing output files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

from src.dataset import EXPECTED_PAIR_COUNT, discover_pairs, save_splits


PAPER_SPLIT_SIZES = {"train": 1_815, "val": 259, "test": 520}


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.int16)
    bits = (values[:, 1:] >= values[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _lesion_fraction(path: Path) -> float:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.uint8)
    return float(np.count_nonzero(values)) / float(values.size)


def create_paper_split(data_root: Path, output: Path, seed: int) -> dict[str, object]:
    pairs = discover_pairs(data_root, validate_readable=True)
    ids = sorted(pairs)
    if len(ids) != EXPECTED_PAIR_COUNT:
        raise ValueError(f"expected {EXPECTED_PAIR_COUNT} pairs, found {len(ids)}")
    if sum(PAPER_SPLIT_SIZES.values()) != len(ids):
        raise AssertionError("paper split sizes do not cover the dataset")

    union = UnionFind(ids)
    exact: dict[str, str] = {}
    dhashes: dict[str, int] = {}
    areas: dict[str, float] = {}
    for index, image_id in enumerate(ids, start=1):
        pair = pairs[image_id]
        digest = _sha256(pair.image_path)
        if digest in exact:
            union.union(image_id, exact[digest])
        else:
            exact[digest] = image_id
        dhashes[image_id] = _difference_hash(pair.image_path)
        areas[image_id] = _lesion_fraction(pair.mask_path)
        if index % 250 == 0:
            print(f"audited {index}/{len(ids)} images", flush=True)

    # Hamming distance <= 1 catches identical or almost identical low-frequency
    # image layouts without aggressively grouping ordinary similar lesions.
    for left_index, left in enumerate(ids):
        left_hash = dhashes[left]
        for right in ids[left_index + 1 :]:
            if (left_hash ^ dhashes[right]).bit_count() <= 1:
                union.union(left, right)

    grouped: dict[str, list[str]] = defaultdict(list)
    for image_id in ids:
        grouped[union.find(image_id)].append(image_id)
    groups = list(grouped.values())
    groups.sort(key=lambda group: (median(areas[value] for value in group), group[0]))

    strata: list[list[list[str]]] = [[] for _ in range(10)]
    for rank, group in enumerate(groups):
        stratum = min(9, rank * 10 // len(groups))
        strata[stratum].append(group)

    rng = random.Random(seed)
    split_names = tuple(PAPER_SPLIT_SIZES)
    assigned = {name: [] for name in split_names}
    stratum_counts = [{name: 0 for name in split_names} for _ in strata]
    target_fraction = {
        name: PAPER_SPLIT_SIZES[name] / len(ids) for name in split_names
    }
    for stratum_index, stratum_groups in enumerate(strata):
        rng.shuffle(stratum_groups)
        stratum_groups.sort(key=lambda group: -len(group))
        stratum_total = sum(map(len, stratum_groups))
        for group in stratum_groups:
            size = len(group)
            candidates = [
                name
                for name in split_names
                if len(assigned[name]) + size <= PAPER_SPLIT_SIZES[name]
            ]
            if not candidates:
                raise RuntimeError("duplicate groups make exact split sizes impossible")

            def score(name: str) -> tuple[float, float, int]:
                global_need = (
                    PAPER_SPLIT_SIZES[name] - len(assigned[name])
                ) / PAPER_SPLIT_SIZES[name]
                expected_here = stratum_total * target_fraction[name]
                stratum_need = expected_here - stratum_counts[stratum_index][name]
                return stratum_need, global_need, -split_names.index(name)

            destination = max(candidates, key=score)
            assigned[destination].extend(group)
            stratum_counts[stratum_index][destination] += size

    counts = {name: len(values) for name, values in assigned.items()}
    if counts != PAPER_SPLIT_SIZES:
        raise RuntimeError(f"unexpected split counts: {counts}")
    for values in assigned.values():
        values.sort()
    save_splits(output, assigned)
    split_sha256 = _sha256(output)
    manifest = {
        "protocol": "ISIC2018-full-70-10-20-area-stratified-grouped-v1",
        "seed": seed,
        "counts": counts,
        "dataset_pairs": len(ids),
        "group_count": len(groups),
        "multi_image_group_count": sum(len(group) > 1 for group in groups),
        "split_file": str(output.resolve()),
        "split_sha256": split_sha256,
        "stratification": "rank deciles of foreground-mask area fraction",
        "grouping": "exact SHA-256 or 64-bit dHash Hamming distance <= 1",
        "warning": "No patient/lesion metadata was present locally; image-level duplicate grouping is the available leakage control.",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".sha256").write_text(
        f"{split_sha256}  {output.name}\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("paper split or manifest already exists; refusing to overwrite")
    print(create_paper_split(args.data_root, args.output, args.seed))


if __name__ == "__main__":
    main()
