"""ISIC 2018 Task 1 数据发现、校验、固定划分与加载。"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


EXPECTED_PAIR_COUNT = 2_594
DEFAULT_IMAGE_SIZE = 256
DEFAULT_SPLIT_SEED = 42
SPLIT_SIZES = {"train": 2_075, "val": 259, "test": 260}
SPLIT_NAMES = tuple(SPLIT_SIZES)

_ID_PATTERN = re.compile(r"^ISIC_\d+$")
_IMAGE_PATTERN = re.compile(r"^(?P<id>ISIC_\d+)\.(?:jpe?g|png)$", re.IGNORECASE)
_MASK_PATTERN = re.compile(
    r"^(?P<id>ISIC_\d+)_segmentation\.png$", re.IGNORECASE
)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_ROTATIONS = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
)


class DatasetValidationError(ValueError):
    """本地 ISIC 文件或数据划分不一致。"""


@dataclass(frozen=True)
class ImageMaskPair:
    """同一 ISIC ID 对应的图像与掩码路径。"""

    id: str
    image_path: Path
    mask_path: Path


def _index_directory(directory: Path, *, is_mask: bool) -> dict[str, Path]:
    """按 ID 索引文件，并拒绝会掩盖数据缺失的歧义文件名。"""

    pattern = _MASK_PATTERN if is_mask else _IMAGE_PATTERN
    label = "mask" if is_mask else "image"
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            relevant = (
                path.suffix.lower() == ".png"
                if is_mask
                else path.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if relevant and not path.name.startswith("."):
                expected = (
                    "ISIC_<digits>_segmentation.png"
                    if is_mask
                    else "ISIC_<digits>.jpg"
                )
                raise DatasetValidationError(
                    f"Invalid {label} filename '{path.name}' in {directory}; "
                    f"expected {expected}."
                )
            continue

        image_id = match.group("id").upper()
        if image_id in indexed:
            raise DatasetValidationError(
                f"Duplicate {label} ID '{image_id}': "
                f"'{indexed[image_id]}' and '{path}'."
            )
        indexed[image_id] = path

    if not indexed:
        raise DatasetValidationError(f"No valid ISIC {label} files found in {directory}.")
    return indexed


def validate_pair_files(pair: ImageMaskPair) -> None:
    """完整解码一对文件，并校验空间尺寸一致。"""

    sizes: dict[str, tuple[int, int]] = {}
    for label, path in (("image", pair.image_path), ("mask", pair.mask_path)):
        try:
            with Image.open(path) as handle:
                handle.load()
                sizes[label] = handle.size
        except Exception as exc:
            raise DatasetValidationError(
                f"Unreadable {label} file '{path}': {exc}"
            ) from exc

    if sizes["image"] != sizes["mask"]:
        width, height = sizes["image"]
        mask_width, mask_height = sizes["mask"]
        raise DatasetValidationError(
            f"Image/mask size mismatch for ID '{pair.id}': image is "
            f"{width}x{height}, mask is {mask_width}x{mask_height}."
        )


def discover_pairs(
    data_root: str | Path, *, validate_readable: bool = True
) -> dict[str, ImageMaskPair]:
    """按 ID 严格配对 ``images`` 与 ``masks`` 下的文件。"""

    root = Path(data_root)
    image_dir, mask_dir = root / "images", root / "masks"
    for label, directory in (("image", image_dir), ("mask", mask_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Missing {label} directory: {directory}. Place the extracted "
                "ISIC 2018 Task 1 files under '<data-root>/images' and "
                "'<data-root>/masks'."
            )

    images = _index_directory(image_dir, is_mask=False)
    masks = _index_directory(mask_dir, is_mask=True)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        problems = []
        for label, missing in (
            ("missing masks for", missing_masks),
            ("missing images for", missing_images),
        ):
            if missing:
                suffix = " ..." if len(missing) > 10 else ""
                problems.append(f"{label} {', '.join(missing[:10])}{suffix}")
        raise DatasetValidationError(
            "Image/mask ID pairing failed: " + "; ".join(problems) + "."
        )

    pairs = {
        image_id: ImageMaskPair(image_id, images[image_id], masks[image_id])
        for image_id in sorted(images)
    }
    if validate_readable:
        for pair in pairs.values():
            validate_pair_files(pair)
    return pairs


def _validate_ids(ids: Iterable[str], label: str) -> list[str]:
    values = list(ids)
    if not all(isinstance(image_id, str) for image_id in values):
        raise DatasetValidationError(f"{label} must contain only string image IDs.")
    malformed = next(
        (image_id for image_id in values if not _ID_PATTERN.fullmatch(image_id)), None
    )
    if malformed is not None:
        raise DatasetValidationError(f"Malformed image ID in {label}: '{malformed}'.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for image_id in values:
        if image_id in seen and image_id not in duplicates:
            duplicates.append(image_id)
        seen.add(image_id)
    if duplicates:
        raise DatasetValidationError(
            f"Duplicate image IDs in {label}: {', '.join(duplicates[:10])}."
        )
    return values


def validate_splits(
    splits: Mapping[str, Any],
    *,
    expected_ids: Iterable[str] | None = None,
    require_standard_counts: bool = False,
) -> dict[str, list[str]]:
    """校验划分名称、ID 互斥性、覆盖关系和可选的固定计数。"""

    if not isinstance(splits, Mapping):
        raise DatasetValidationError("splits.json must contain a JSON object.")
    if set(splits) != set(SPLIT_NAMES):
        missing = sorted(set(SPLIT_NAMES) - set(splits))
        extra = sorted(set(splits) - set(SPLIT_NAMES))
        raise DatasetValidationError(
            f"splits.json must contain exactly {list(SPLIT_NAMES)}; "
            f"missing={missing}, extra={extra}."
        )

    validated: dict[str, list[str]] = {}
    owners: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        values = splits[split_name]
        if not isinstance(values, list):
            raise DatasetValidationError(
                f"Split '{split_name}' must be a JSON list of image IDs."
            )
        split_ids = _validate_ids(values, f"split '{split_name}'")
        validated[split_name] = split_ids
        for image_id in split_ids:
            if image_id in owners:
                raise DatasetValidationError(
                    f"Image ID '{image_id}' occurs in both "
                    f"'{owners[image_id]}' and '{split_name}' splits."
                )
            owners[image_id] = split_name

    if require_standard_counts:
        counts = {name: len(validated[name]) for name in SPLIT_NAMES}
        if counts != SPLIT_SIZES:
            raise DatasetValidationError(
                f"Fixed split counts must be {SPLIT_SIZES}, got {counts}."
            )

    if expected_ids is not None:
        expected = set(_validate_ids(expected_ids, "dataset IDs"))
        actual = set(owners)
        missing, unknown = sorted(expected - actual), sorted(actual - expected)
        if missing or unknown:
            raise DatasetValidationError(
                "splits.json does not exactly cover the paired dataset IDs: "
                f"missing={missing[:10]}, unknown={unknown[:10]}."
            )
    return validated


def create_fixed_splits(
    ids: Iterable[str], *, seed: int = DEFAULT_SPLIT_SEED
) -> dict[str, list[str]]:
    """从排序后的 ID 创建可复现的固定 2075/259/260 划分。"""

    image_ids = _validate_ids(ids, "dataset IDs")
    if len(image_ids) != EXPECTED_PAIR_COUNT:
        raise DatasetValidationError(
            f"Cannot create the fixed {SPLIT_SIZES} split from "
            f"{len(image_ids)} pairs; expected exactly {EXPECTED_PAIR_COUNT}."
        )

    shuffled = sorted(image_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = SPLIT_SIZES["train"]
    val_end = train_end + SPLIT_SIZES["val"]
    return validate_splits(
        {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        },
        expected_ids=image_ids,
        require_standard_counts=True,
    )


def save_splits(
    split_path: str | Path,
    splits: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """保存仅含 ID 的划分；默认拒绝覆盖。"""

    path = Path(split_path)
    payload = json.dumps(validate_splits(splits), indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as file:
            file.write(payload)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Split file already exists and was not overwritten: {path}"
        ) from exc
    return path


def load_splits(
    split_path: str | Path,
    *,
    expected_ids: Iterable[str] | None = None,
    require_standard_counts: bool = False,
) -> dict[str, list[str]]:
    """读取并严格校验已有划分。"""

    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run 'python prepare_data.py "
            f"--data-root {path.parent}' first."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Cannot read split file '{path}': {exc}") from exc
    return validate_splits(
        raw,
        expected_ids=expected_ids,
        require_standard_counts=require_standard_counts,
    )


class ISICDataset(Dataset[dict[str, Any]]):
    """加载成对图像和二值掩码；增强由 ``(seed, epoch, index)`` 唯一决定。"""

    def __init__(
        self,
        data_root: str | Path,
        split: str | None = None,
        *,
        ids: Sequence[str] | None = None,
        image_size: int | Sequence[int] = DEFAULT_IMAGE_SIZE,
        augment: bool | None = None,
        seed: int = DEFAULT_SPLIT_SEED,
        splits_path: str | Path | None = None,
        require_standard_split_counts: bool | None = None,
        validate_files: bool = True,
    ) -> None:
        if (split is None) == (ids is None):
            raise ValueError("Provide exactly one of 'split' or 'ids'.")
        if isinstance(image_size, int):
            height = width = image_size
        else:
            try:
                height, width = map(int, image_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "image_size must be an int or a (height, width) pair."
                ) from exc
        if height <= 0 or width <= 0:
            raise ValueError("image_size dimensions must be positive.")

        self.data_root = Path(data_root)
        self.height, self.width = height, width
        self.seed, self.epoch = int(seed), 0
        self.split = split
        self.augment = split == "train" if augment is None else bool(augment)

        pairs = discover_pairs(self.data_root, validate_readable=validate_files)
        if split is not None:
            if split not in SPLIT_NAMES:
                raise ValueError(
                    f"Unknown split '{split}'; choose one of {list(SPLIT_NAMES)}."
                )
            path = Path(splits_path) if splits_path else self.data_root / "splits.json"
            require_counts = (
                len(pairs) == EXPECTED_PAIR_COUNT and splits_path is None
                if require_standard_split_counts is None
                else require_standard_split_counts
            )
            selected_ids = load_splits(
                path,
                expected_ids=pairs,
                # The legacy default split retains its historical fixed counts.
                # Explicit paper splits may use another preregistered protocol,
                # while still requiring disjoint IDs and complete coverage.
                require_standard_counts=require_counts,
            )[split]
        else:
            assert ids is not None
            selected_ids = _validate_ids(ids, "selected dataset IDs")
            missing = sorted(set(selected_ids) - set(pairs))
            if missing:
                raise DatasetValidationError(
                    f"Requested IDs are absent from the paired dataset: {missing[:10]}."
                )

        self.ids = list(selected_ids)
        self.samples = [pairs[image_id] for image_id in self.ids]

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """切换到指定轮次的确定性增强流。"""

        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = int(epoch)

    def _augment_pair(
        self, image: Image.Image, mask: Image.Image, index: int
    ) -> tuple[Image.Image, Image.Image]:
        # 固定整数混合保证所有 DataLoader worker 复现同一几何增强。
        mixed = (
            (self.seed & 0xFFFFFFFFFFFFFFFF)
            ^ ((self.epoch + 1) * 0x9E3779B97F4A7C15)
            ^ ((index + 1) * 0xBF58476D1CE4E5B9)
        ) & 0xFFFFFFFFFFFFFFFF
        rng = random.Random(mixed)
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rotation = _ROTATIONS[rng.randrange(4)]
        if rotation is not None:
            image, mask = image.transpose(rotation), mask.transpose(rotation)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.samples[index]
        try:
            with Image.open(pair.image_path) as file:
                image = file.convert("RGB")
            with Image.open(pair.mask_path) as file:
                mask = file.convert("L")
        except Exception as exc:
            raise DatasetValidationError(
                f"Failed to read paired files for ID '{pair.id}': {exc}"
            ) from exc
        if image.size != mask.size:
            raise DatasetValidationError(
                f"Image/mask size mismatch for ID '{pair.id}': "
                f"image={image.size}, mask={mask.size}."
            )

        # 图像和掩码共享几何增强，但按各自要求插值。
        if self.augment:
            image, mask = self._augment_pair(image, mask, index)
        size = (self.width, self.height)  # PIL 使用 (宽, 高)。
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - 0.5) / 0.5
        mask_array = (np.asarray(mask) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return {"image": image_tensor, "mask": mask_tensor, "id": pair.id}


def seed_worker(_: int) -> None:
    """用 PyTorch worker 种子同步 Python 与 NumPy 随机流。"""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
