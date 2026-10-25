from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import prepare_data
from src.dataset import (
    EXPECTED_PAIR_COUNT,
    SPLIT_SIZES,
    DatasetValidationError,
    ISICDataset,
    ImageMaskPair,
    create_fixed_splits,
    discover_pairs,
    load_splits,
    save_splits,
    seed_worker,
)


def _make_dirs(root: Path) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)


def _write_pair(
    root: Path,
    image_id: str,
    *,
    size: tuple[int, int] = (19, 13),
    image_array: np.ndarray | None = None,
    mask_array: np.ndarray | None = None,
    image_extension: str = ".jpg",
) -> None:
    _make_dirs(root)
    width, height = size
    if image_array is None:
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        image_array[..., 0] = 220
        image_array[..., 1] = 80
    if mask_array is None:
        mask_array = np.zeros((height, width), dtype=np.uint8)
        mask_array[2 : max(3, height - 2), 3 : max(4, width - 3)] = 255

    Image.fromarray(image_array, mode="RGB").save(
        root / "images" / f"{image_id}{image_extension}"
    )
    Image.fromarray(mask_array, mode="L").save(
        root / "masks" / f"{image_id}_segmentation.png"
    )


def _official_ids() -> list[str]:
    return [f"ISIC_{index:07d}" for index in range(EXPECTED_PAIR_COUNT)]


def test_pairs_are_matched_by_id_not_directory_order(tmp_path: Path) -> None:
    _write_pair(tmp_path, "ISIC_0000002")
    _write_pair(tmp_path, "ISIC_0000001")

    pairs = discover_pairs(tmp_path)

    assert list(pairs) == ["ISIC_0000001", "ISIC_0000002"]
    for image_id, pair in pairs.items():
        assert pair.image_path.stem == image_id
        assert pair.mask_path.stem == f"{image_id}_segmentation"


def test_missing_mask_raises_clear_error(tmp_path: Path) -> None:
    _make_dirs(tmp_path)
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "ISIC_0000001.jpg")
    # Keep the mask directory non-empty so discovery reaches the pairing check.
    Image.new("L", (8, 8)).save(
        tmp_path / "masks" / "ISIC_0000002_segmentation.png"
    )

    with pytest.raises(DatasetValidationError, match="missing masks.*ISIC_0000001"):
        discover_pairs(tmp_path)


def test_duplicate_image_id_raises(tmp_path: Path) -> None:
    _write_pair(tmp_path, "ISIC_0000001")
    Image.new("RGB", (19, 13)).save(tmp_path / "images" / "ISIC_0000001.png")

    with pytest.raises(DatasetValidationError, match="Duplicate image ID"):
        discover_pairs(tmp_path)


def test_unreadable_pil_file_raises_during_validation(tmp_path: Path) -> None:
    _make_dirs(tmp_path)
    (tmp_path / "images" / "ISIC_0000001.jpg").write_bytes(b"not a jpeg")
    Image.new("L", (8, 8)).save(
        tmp_path / "masks" / "ISIC_0000001_segmentation.png"
    )

    with pytest.raises(DatasetValidationError, match="Unreadable image file"):
        discover_pairs(tmp_path)


def test_mismatched_image_mask_dimensions_raise(tmp_path: Path) -> None:
    _make_dirs(tmp_path)
    Image.new("RGB", (8, 9)).save(tmp_path / "images" / "ISIC_0000001.jpg")
    Image.new("L", (8, 8)).save(
        tmp_path / "masks" / "ISIC_0000001_segmentation.png"
    )

    with pytest.raises(DatasetValidationError, match="size mismatch"):
        discover_pairs(tmp_path)


def test_default_preprocessing_shapes_normalization_and_binary_mask(
    tmp_path: Path,
) -> None:
    mask = np.zeros((11, 17), dtype=np.uint8)
    mask[1:5, 2:8] = 17
    mask[6:, 9:] = 255
    _write_pair(tmp_path, "ISIC_0000001", size=(17, 11), mask_array=mask)

    dataset = ISICDataset(
        tmp_path, ids=["ISIC_0000001"], augment=False, seed=123
    )
    sample = dataset[0]

    assert sample["id"] == "ISIC_0000001"
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert sample["image"].min() >= -1.0
    assert sample["image"].max() <= 1.0
    assert set(torch.unique(sample["mask"]).tolist()) == {0.0, 1.0}


def test_augmentation_is_synchronized_and_reproducible(tmp_path: Path) -> None:
    size = 256
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[9:77, 31:103] = 255
    mask[141:211, 188:241] = 255
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = mask
    _write_pair(
        tmp_path,
        "ISIC_0000001",
        size=(size, size),
        image_array=image,
        mask_array=mask,
        image_extension=".png",
    )

    first = ISICDataset(
        tmp_path, ids=["ISIC_0000001"], augment=True, seed=42
    )
    second = ISICDataset(
        tmp_path, ids=["ISIC_0000001"], augment=True, seed=42
    )
    first_sample = first[0]
    second_sample = second[0]

    assert torch.equal(first_sample["image"], second_sample["image"])
    assert torch.equal(first_sample["mask"], second_sample["mask"])
    # Red is foreground and black is background, so an exact match proves that
    # flips and the 0/90/180/270-degree rotation were shared by both modalities.
    assert torch.equal(first_sample["image"][0] > 0, first_sample["mask"][0].bool())

    first.set_epoch(3)
    second.set_epoch(3)
    assert torch.equal(first[0]["image"], second[0]["image"])
    assert torch.equal(first[0]["mask"], second[0]["mask"])


def test_dataset_can_read_split_file_directly(tmp_path: Path) -> None:
    ids = ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"]
    for image_id in ids:
        _write_pair(tmp_path, image_id)
    save_splits(
        tmp_path / "splits.json",
        {"train": [ids[1]], "val": [ids[0]], "test": [ids[2]]},
    )

    dataset = ISICDataset(tmp_path, split="train", augment=False)

    assert len(dataset) == 1
    assert dataset[0]["id"] == ids[1]


def test_fixed_split_counts_disjointness_and_seed_reproducibility() -> None:
    ids = _official_ids()

    first = create_fixed_splits(ids, seed=42)
    same = create_fixed_splits(reversed(ids), seed=42)
    different = create_fixed_splits(ids, seed=43)

    assert {name: len(values) for name, values in first.items()} == SPLIT_SIZES
    assert first == same
    assert first != different
    all_ids = first["train"] + first["val"] + first["test"]
    assert len(all_ids) == len(set(all_ids)) == EXPECTED_PAIR_COUNT
    assert set(all_ids) == set(ids)


def test_fixed_split_rejects_non_official_count() -> None:
    with pytest.raises(DatasetValidationError, match="expected exactly 2594"):
        create_fixed_splits(["ISIC_0000001"], seed=42)


def test_split_save_refuses_overwrite_and_preserves_contents(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    original = {
        "train": ["ISIC_0000001"],
        "val": ["ISIC_0000002"],
        "test": ["ISIC_0000003"],
    }
    replacement = {
        "train": ["ISIC_0000003"],
        "val": ["ISIC_0000002"],
        "test": ["ISIC_0000001"],
    }
    save_splits(path, original)
    before = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns

    with pytest.raises(FileExistsError, match="not overwritten"):
        save_splits(path, replacement)

    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == before_mtime
    save_splits(path, replacement, overwrite=True)
    assert load_splits(path) == replacement


def test_split_loader_rejects_overlap_and_incomplete_coverage(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "train": ["ISIC_0000001"],
                "val": ["ISIC_0000001"],
                "test": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="occurs in both"):
        load_splits(path)

    path.write_text(
        json.dumps(
            {
                "train": ["ISIC_0000001"],
                "val": [],
                "test": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="does not exactly cover"):
        load_splits(
            path,
            expected_ids=["ISIC_0000001", "ISIC_0000002"],
        )


def test_prepare_data_warns_for_non_official_pair_count(tmp_path: Path) -> None:
    _write_pair(tmp_path, "ISIC_0000001")

    with pytest.warns(RuntimeWarning, match="normally contains 2594"):
        with pytest.raises(DatasetValidationError, match="Cannot create the fixed"):
            prepare_data.prepare_dataset(tmp_path)


def test_prepare_data_does_not_rewrite_an_existing_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _official_ids()
    fake_pairs = {
        image_id: ImageMaskPair(
            image_id,
            tmp_path / "images" / f"{image_id}.jpg",
            tmp_path / "masks" / f"{image_id}_segmentation.png",
        )
        for image_id in ids
    }
    monkeypatch.setattr(prepare_data, "discover_pairs", lambda *args, **kwargs: fake_pairs)
    original = create_fixed_splits(ids, seed=42)
    path = save_splits(tmp_path / "splits.json", original)
    before = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns

    result = prepare_data.prepare_dataset(tmp_path, seed=999)

    assert result == path
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == before_mtime


def test_seed_worker_seeds_python_and_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "initial_seed", lambda: 1_234_567_890)
    seed_worker(17)
    python_value = random.random()
    numpy_value = float(np.random.random())

    random.seed(1_234_567_890)
    np.random.seed(1_234_567_890)
    assert python_value == random.random()
    assert numpy_value == float(np.random.random())
