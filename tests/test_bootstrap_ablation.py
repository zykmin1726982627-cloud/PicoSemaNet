"""Tests for paired per-image bootstrap statistics."""

from __future__ import annotations

import numpy as np
import pytest

from bootstrap_ablation import paired_bootstrap


def test_paired_bootstrap_reports_positive_matched_difference() -> None:
    reference = np.asarray([0.8, 0.7, 0.9, 0.6])
    comparator = reference - 0.1
    result = paired_bootstrap(reference, comparator, samples=2_000, seed=9)
    assert result["num_images"] == 4
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["ci95_low"] == pytest.approx(0.1)
    assert result["ci95_high"] == pytest.approx(0.1)
    assert result["fraction_images_reference_better"] == 1.0


@pytest.mark.parametrize(
    "reference,comparator",
    [([], []), ([1.0], [1.0, 2.0]), ([[1.0]], [[1.0]])],
)
def test_paired_bootstrap_rejects_unmatched_vectors(reference, comparator) -> None:
    with pytest.raises(ValueError):
        paired_bootstrap(np.asarray(reference), np.asarray(comparator), samples=10, seed=1)
