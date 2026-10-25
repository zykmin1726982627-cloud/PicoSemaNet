"""Recompute archived legacy-EC bootstrap statistics without rerunning inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bootstrap_ablation import METRICS, paired_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bootstrap_legacy_recomputed.json")
    args = parser.parse_args()
    original = json.loads((ROOT / "formal_paper_ablation_bootstrap_v1/bootstrap_results.json").read_text())
    records = {}
    with (ROOT / "formal_paper_ablation_bootstrap_v1/per_image_metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            records.setdefault(row["label"], {})[row["image_id"]] = row
    identifiers = sorted(next(iter(records.values())))
    if any(sorted(values) != identifiers for values in records.values()):
        raise ValueError("Per-image IDs are not paired")

    def vector(labels, metric):
        return np.stack([
            [float(records[label][image_id][metric]) for image_id in identifiers]
            for label in labels.split(";")
        ]).mean(axis=0)

    comparisons = []
    ordered_names = list(dict.fromkeys(r["comparison"] for r in original["comparisons"]))
    for row in original["comparisons"]:
        comparison_index = ordered_names.index(row["comparison"])
        metric_index = METRICS.index(row["metric"])
        result = paired_bootstrap(
            vector(row["reference_labels"], row["metric"]),
            vector(row["comparator_labels"], row["metric"]),
            samples=original["bootstrap_samples"],
            seed=original["bootstrap_seed"] + comparison_index * len(METRICS) + metric_index,
        )
        comparisons.append({"comparison": row["comparison"], "metric": row["metric"], **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scope": "legacy EC-containing objectives; original Chebyshev BF1 inputs",
        "bootstrap_samples": original["bootstrap_samples"],
        "bootstrap_seed": original["bootstrap_seed"], "comparisons": comparisons,
    }, indent=2))
    print(f"Saved {len(comparisons)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
