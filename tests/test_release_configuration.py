import json
from pathlib import Path

from evaluate import build_parser

ROOT = Path(__file__).resolve().parents[1]

def test_primary_and_matched_objectives_follow_manuscript():
    for group in ("primary_full", "matched100_full"):
        for seed in (42, 43, 44):
            config = json.loads((ROOT / f"configs/{group}_seed{seed}.json").read_text())
            assert config["lambda_edge"] == config["lambda_deep_supervision"] == 0
            assert (config["lambda_boundary"], config["lambda_texture"], config["lambda_sdf"]) == (0.18, 0.05, 0.12)
            assert config["lambda_distill"] == 0.25
            assert config["splits_path"].endswith("paper_split_70_10_20_seed20260912.json")
            if group == "matched100_full":
                assert config["epochs"] == 100 and config["early_stopping_patience"] == 101

def test_standalone_evaluation_defaults_to_paper_split_and_euclidean_bf1():
    args = build_parser().parse_args(["--checkpoint", "unused.pt", "--data-root", "unused-data"])
    assert args.boundary_distance == "euclidean"
    split = json.loads(args.splits_path.read_text())
    assert [len(split[k]) for k in ("train", "val", "test")] == [1815, 259, 520]
    sets = [set(split[k]) for k in ("train", "val", "test")]
    assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3))
