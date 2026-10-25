"""Re-evaluate archived checkpoints on the locked test split in FP32."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--group", default="primary_full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/euclidean_bf1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from evaluate import load_model_from_checkpoint
    from src.dataset import ISICDataset
    from src.metrics import compute_metrics
    records = json.loads((ROOT / "artifacts/experiments.json").read_text())
    selected = [r for r in records if r["group"] == args.group or args.group == "all"]
    if not selected:
        parser.error("Unknown group")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ISICDataset(args.data_root, split="test", image_size=320, augment=False,
                          splits_path=ROOT / "data/ISIC2018/paper_split_70_10_20_seed20260912.json")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for record in selected:
        model, _, _ = load_model_from_checkpoint(ROOT / record["checkpoint"], args.device)
        model.eval()
        rows = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(batch["image"].to(args.device), return_aux=False)["seg_logits"]
                metrics = compute_metrics(logits, batch["mask"].to(args.device), from_logits=True,
                                          boundary_distance="euclidean", boundary_tolerance=2,
                                          return_per_image=True)
                for i, image_id in enumerate(batch["id"]):
                    rows.append({"id": image_id, **{k: float(v[i].cpu()) for k, v in metrics.items()}})
        result = {"run_id": record["run_id"], "boundary_distance": "euclidean",
                  "boundary_tolerance": 2, "image_size": 320, "device": args.device,
                  "torch_version": torch.__version__, "num_images": len(rows),
                  "checkpoint_sha256": record["checkpoint_sha256"],
                  "means": {k: sum(r[k] for r in rows) / len(rows) for k in ("dice", "iou", "bf1", "accuracy")},
                  "per_image": rows}
        (args.output_dir / (record["run_id"] + ".json")).write_text(json.dumps(result, indent=2))
        print(record["run_id"], result["means"])
        del model

if __name__ == "__main__":
    main()
