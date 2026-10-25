"""Per-image evaluation and paired bootstrap for the final PicoLite ablations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import load_model_from_checkpoint
from src.dataset import ISICDataset
from src.metrics import compute_metrics
from src.utils import resolve_device, save_json, seed_worker


METRICS = ("dice", "iou", "accuracy", "bf1")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    family: str
    seed: int
    checkpoint: Path


def paired_bootstrap(
    reference: np.ndarray,
    comparator: np.ndarray,
    *,
    samples: int,
    seed: int,
    chunk_size: int = 1_000,
) -> dict[str, float | int]:
    """Bootstrap the paired per-image mean difference, reference minus comparator."""

    reference = np.asarray(reference, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    if reference.ndim != 1 or reference.shape != comparator.shape or not len(reference):
        raise ValueError("paired bootstrap arrays must be non-empty matched vectors")
    if samples <= 0 or chunk_size <= 0:
        raise ValueError("bootstrap sample and chunk counts must be positive")
    difference = reference - comparator
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    written = 0
    while written < samples:
        count = min(chunk_size, samples - written)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        boot[written : written + count] = difference[indices].mean(axis=1)
        written += count
    lower_tail = (np.count_nonzero(boot <= 0) + 1) / (samples + 1)
    upper_tail = (np.count_nonzero(boot >= 0) + 1) / (samples + 1)
    return {
        "num_images": int(len(difference)),
        "bootstrap_samples": int(samples),
        "reference_mean": float(reference.mean()),
        "comparator_mean": float(comparator.mean()),
        "mean_difference": float(difference.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "two_sided_p": float(min(1.0, 2 * min(lower_tail, upper_tail))),
        "fraction_images_reference_better": float(np.mean(difference > 0)),
    }


def model_specs(project: Path) -> list[ModelSpec]:
    return [
        ModelSpec("main_seed42", "fullpath_nds", 42, project / "formal_paper_seed42_v1/ablation_a7_no_deep_seed42/best.pt"),
        ModelSpec("main_seed43", "fullpath_nds", 43, project / "formal_paper_multiseed_v1/ablation_a7_no_deep_seed43/best.pt"),
        ModelSpec("main_seed44", "fullpath_nds", 44, project / "formal_paper_multiseed_v1/ablation_a7_no_deep_seed44/best.pt"),
        ModelSpec("budget_main_seed42", "budget_fullpath_nds", 42, project / "formal_paper_ablation_supplement_v1/s1_budget_matched_fullpath_seed42/best.pt"),
        ModelSpec("m2_seed42", "no_fullpath", 42, project / "formal_paper_ablation_nds_v1/m2_no_fullpath_seed42/best.pt"),
        ModelSpec("m2_seed43", "no_fullpath", 43, project / "formal_paper_ablation_supplement_v1/s2_no_fullpath_seed43/best.pt"),
        ModelSpec("m2_seed44", "no_fullpath", 44, project / "formal_paper_ablation_priority_v1/m2_no_fullpath_seed44/best.pt"),
        ModelSpec("m3_seed42", "shallow_semantic", 42, project / "formal_paper_ablation_nds_v1/m3_shallow_semantic_seed42/best.pt"),
        ModelSpec("m3_seed43", "shallow_semantic", 43, project / "formal_paper_ablation_priority_v1/m3_shallow_semantic_seed43/best.pt"),
        ModelSpec("m3_seed44", "shallow_semantic", 44, project / "formal_paper_ablation_priority_v1/m3_shallow_semantic_seed44/best.pt"),
        ModelSpec("m4_seed42", "no_boundary", 42, project / "formal_paper_ablation_nds_v1/m4_no_boundary_seed42/best.pt"),
        ModelSpec("m5_seed42", "no_texture", 42, project / "formal_paper_ablation_nds_v1/m5_no_texture_seed42/best.pt"),
        ModelSpec("m6_seed42", "no_sdf", 42, project / "formal_paper_ablation_nds_v1/m6_no_sdf_seed42/best.pt"),
    ]


@torch.inference_mode()
def evaluate_per_image(
    spec: ModelSpec,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model, _, _ = load_model_from_checkpoint(spec.checkpoint, device)
    model.eval()
    rows: dict[str, dict[str, float]] = {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images, return_aux=False)["seg_logits"]
        scores = compute_metrics(
            logits,
            masks,
            threshold=0.5,
            from_logits=True,
            boundary_tolerance=2,
            return_per_image=True,
        )
        identifiers = list(batch["id"])
        for index, identifier in enumerate(identifiers):
            rows[str(identifier)] = {
                name: float(scores[name][index].detach().cpu()) for name in METRICS
            }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _vectors(
    records: dict[str, dict[str, dict[str, float]]],
    labels: tuple[str, ...],
    identifiers: list[str],
    metric: str,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray([records[label][identifier][metric] for identifier in identifiers])
            for label in labels
        ]
    ).mean(axis=0)


def run(args: argparse.Namespace) -> None:
    project = Path(args.project_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = model_specs(project)
    missing = [str(spec.checkpoint) for spec in specs if not spec.checkpoint.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints:\n" + "\n".join(missing))

    device = resolve_device(args.device)
    dataset = ISICDataset(
        project / "data/ISIC2018",
        split="test",
        image_size=320,
        augment=False,
        seed=42,
        splits_path=project / "data/ISIC2018/paper_split_70_10_20_seed20260912.json",
        require_standard_split_counts=False,
        validate_files=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if args.num_workers else None,
        persistent_workers=False,
    )

    records: dict[str, dict[str, dict[str, float]]] = {}
    with (output / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("label", "family", "seed", "image_id", *METRICS)
        )
        writer.writeheader()
        for spec in specs:
            print(f"EVALUATE {spec.label}: {spec.checkpoint}", flush=True)
            rows = evaluate_per_image(spec, loader, device)
            records[spec.label] = rows
            for identifier in sorted(rows):
                writer.writerow(
                    {
                        "label": spec.label,
                        "family": spec.family,
                        "seed": spec.seed,
                        "image_id": identifier,
                        **rows[identifier],
                    }
                )

    identifiers = sorted(records["main_seed42"])
    for label, rows in records.items():
        if sorted(rows) != identifiers:
            raise RuntimeError(f"test image IDs do not align for {label}")

    comparisons: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ("main_vs_m2_seed42", ("main_seed42",), ("m2_seed42",)),
        ("main_vs_m2_seed43", ("main_seed43",), ("m2_seed43",)),
        ("main_vs_m2_seed44", ("main_seed44",), ("m2_seed44",)),
        ("main_vs_m3_seed42", ("main_seed42",), ("m3_seed42",)),
        ("main_vs_m3_seed43", ("main_seed43",), ("m3_seed43",)),
        ("main_vs_m3_seed44", ("main_seed44",), ("m3_seed44",)),
        ("main_3seed_mean_vs_m2_3seed_mean", ("main_seed42", "main_seed43", "main_seed44"), ("m2_seed42", "m2_seed43", "m2_seed44")),
        ("main_3seed_mean_vs_m3_3seed_mean", ("main_seed42", "main_seed43", "main_seed44"), ("m3_seed42", "m3_seed43", "m3_seed44")),
        ("budget_main_vs_m4_seed42", ("budget_main_seed42",), ("m4_seed42",)),
        ("budget_main_vs_m5_seed42", ("budget_main_seed42",), ("m5_seed42",)),
        ("budget_main_vs_m6_seed42", ("budget_main_seed42",), ("m6_seed42",)),
    ]
    results: list[dict[str, Any]] = []
    for comparison_index, (name, reference_labels, comparator_labels) in enumerate(comparisons):
        for metric_index, metric in enumerate(METRICS):
            result = paired_bootstrap(
                _vectors(records, reference_labels, identifiers, metric),
                _vectors(records, comparator_labels, identifiers, metric),
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + comparison_index * len(METRICS) + metric_index,
            )
            results.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "reference_labels": ";".join(reference_labels),
                    "comparator_labels": ";".join(comparator_labels),
                    **result,
                }
            )

    fields = (
        "comparison", "metric", "reference_labels", "comparator_labels",
        "num_images", "bootstrap_samples", "reference_mean", "comparator_mean",
        "mean_difference", "ci95_low", "ci95_high", "two_sided_p",
        "fraction_images_reference_better",
    )
    with (output / "bootstrap_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    save_json(
        {
            "test_images": len(identifiers),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "models": [
                {
                    "label": spec.label,
                    "family": spec.family,
                    "seed": spec.seed,
                    "checkpoint": str(spec.checkpoint),
                }
                for spec in specs
            ],
            "comparisons": results,
        },
        output / "bootstrap_results.json",
    )
    print(f"WROTE {len(results)} metric comparisons for {len(identifiers)} images", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="formal_paper_ablation_bootstrap_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
