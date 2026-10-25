# PicoSemaNet: Full-Path Fixed-Channel Supervision

Code and experiment artifacts collected for **Full-Path Fixed-Channel Supervision for Compact Skin Lesion Segmentation**.

The manuscript's primary model is `picolite_fullpath_distill`: **4,844 inference parameters**, trained with segmentation + FPFCS + KD, with **edge consistency (EC) and decoder deep supervision disabled**. The older no-deep-supervision model that retains EC is an ablation, not the primary model in this manuscript.

## Setup

Use Python 3.10 or later and install a PyTorch build appropriate for your device, then:

```bash
python -m pip install -r requirements.txt
```

Original training used Python 3.10.12, PyTorch 2.11.0+cu128, CUDA 12.8 and an RTX A5000. The paper's later historical BF1 recomputation used CPU FP32 / PyTorch 2.14.0. Requirements retain the server's broad version ranges; cross-version bitwise reproduction is not guaranteed.

## Data

Provide the ISIC 2018 Task 1 images and masks separately:

```text
data/ISIC2018/images/ISIC_*.jpg
data/ISIC2018/masks/ISIC_*_segmentation.png
```

The exact split, its generation manifest, and SHA-256 are supplied in `data/ISIC2018/`: 1,815 train / 259 validation / 520 test images. Use this supplied split. The dataset class's historical default `splits.json` is a different split. Images, masks, and manuscript figures are not redistributed.

## Checkpoints

Download `PicoSemaNet_paper_checkpoints.zip` from the [v1.0.0 release](https://github.com/zykmin1726982627-cloud/PicoSemaNet/releases/tag/v1.0.0). It contains original best-validation checkpoints for the experiments referenced by the manuscript. Extract it into this directory to create `checkpoints/`, then run:

```bash
python tools/verify_checkpoints.py
```

Large binary weights are intentionally excluded from the code ZIP and Git tracking. Training checkpoints retain original optimizer state and original path metadata; inference does not need optimizer state. Checkpoint loading follows the original trusted-checkpoint format.

## Primary model

```bash
python run_experiment.py --config configs/primary_full_seed42.json --dry-run
python run_experiment.py --config configs/primary_full_seed42.json --device cuda:0
python evaluate.py --checkpoint checkpoints/primary_full_seed42/best.pt --data-root data/ISIC2018 --device cpu --output-dir outputs/primary_seed42_evaluation
```

Repeat the training command with seed43 and seed44 configurations for the three primary runs. All commands use the locked paper split; evaluation defaults to two-pixel Euclidean BF1. Training configurations preserve the original training-time BF1 definition, which did not select checkpoints (validation Dice did). The primary 200-epoch-cap model and the exactly-100-epoch comparison are separate experiments.

## Matched 100-epoch experiment

```bash
python run_experiment.py --config configs/matched100_full_seed42.json
python run_experiment.py --config configs/matched100_no_fpfcs_seed42.json
```

Repeat for seeds 43 and 44. Both groups retain KD, disable EC/DS, and use exactly 100 epochs with early-stop patience 101. Only the three FPFCS loss weights differ.

## Reproducing the tables

`artifacts/paper_groups.json` maps each objective to run IDs. `artifacts/experiments.json` maps those IDs to portable configs, checkpoints, server run paths, and original scores. `docs/PAPER_MAPPING.md` explains Tables 1-3. Original per-run logs and raw test metrics are in `results/original/`.

**Metric provenance:** historical server raw BF1 is Chebyshev. The manuscript replaced it with CPU-recomputed Euclidean BF1. Those later recomputation files and figure/MAC scripts were not located in the server project. Do not substitute the historical BF1 values for the paper's Euclidean BF1. A new evaluation entry point is supplied:

```bash
python tools/recompute_euclidean_bf1.py --data-root data/ISIC2018 --group primary_full --device cpu
```

This writes new results under `outputs/`; it does not overwrite original metrics or silently replace original GPU Dice/IoU. Use `--group all` to evaluate all collected runs. This collection step did not re-run full test-set evaluation or training.

`bootstrap_ablation.py` and `formal_paper_ablation_bootstrap_v1/` preserve the original **legacy EC-containing** bootstrap protocol and per-image inputs. Its `--project-root` expects the original formal directory layout, available in the archival backup; it is not an EC-free significance analysis. A portable input-only recomputation is available with:

```bash
python tools/bootstrap_legacy_inputs.py
```

The portable script reproduces all 44 archived metric comparisons exactly in the checked server environment. See `docs/REPRODUCIBILITY.md`.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The server's relevant model, loss, dataset, metric, export and training tests are included. Release-specific checks cover the locked split, the manuscript's primary objective and evaluation CLI defaults. Validation completed with **114 passed, 2 CUDA tests skipped** on CPU; all 53 configs passed the training validator, and all three primary checkpoints produced finite 320x320 predictions with exactly 4,844 inference parameters. The original server project was not modified. See `artifacts/release_validation.json`.

## License and provenance

The code is distributed under the [MIT License](LICENSE). ISIC images and masks are not redistributed; their dataset terms remain separate.

Source provenance and changes from the original training project are documented in `artifacts/source_manifest.json` and `docs/CHANGES_FROM_SERVER.md`. The manuscript is identified by title and SHA-256 in `artifacts/manuscript.json`; publication identifiers can be added when available.
