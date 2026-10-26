# Additional reproducibility files (26 October 2026)

[Download the complete supplement](PicoSemaNet_missing_files_20261026.zip)

This addition-only package contains **374 files (7,399,269 uncompressed bytes)**,
including 371 previously absent local evidence files and three navigation and
verification files. No file that existed in the repository at commit
`d16c386052b59ab86e28630b4ee1f99f093265f8` is modified or deleted.

## What is added

- Original later CPU Euclidean-BF1 scripts, per-image metrics, summaries and provenance.
- EC-free manuscript configuration mapping, original figure scripts and data, and four figures.
- Original model-parameter/MAC verification script and recorded output.
- Matched 100-epoch control audit, logs and three distinct last checkpoints absent from the existing best-checkpoint release.
- Portable replay helpers, validation records and category-specific license texts.

The root README and docs/PAPER_MAPPING.md describe the earlier server collection.
Their references to unavailable later BF1 and figure/MAC artifacts are historical;
this package supplies those later local artifacts without rewriting those files.

## Extract and verify

Extract the ZIP into the repository root. Every archive entry starts with the
new `supplementary/` directory; none overlaps the 272 pre-existing repository
files. Inside it, README.md describes the evidence and SHA256SUMS.txt verifies the
supplement contents. FILE_MAP.json maps all 700 full-archive paths to added files,
identical existing Git files, existing release checkpoints or the excluded cache.

The external [manifest](PicoSemaNet_missing_files_20261026.manifest.json) lists all
374 paths, sizes and SHA-256 hashes; the [.sha256 file](PicoSemaNet_missing_files_20261026.zip.sha256)
verifies the ZIP. Identical content already in the Git tree or declared in the
existing checkpoint release was not copied again. One generated .pyc was excluded.

This supplement is not a complete standalone copy of the original archive.
For an intact runnable archival tree, use the fixed-version Zenodo archive at
https://doi.org/10.5281/zenodo.23643500 and its REPRODUCE.md. Historical scripts and
manifests retain their original layout assumptions. No experiments or result
values were changed by this addition.

## Licensing

Original software is MIT. Author-created experiment records, figures and weights
are CC BY 4.0; third-party rights remain unchanged. The package contains RIGHTS.md,
LICENSE-CODE.txt and LICENSE-DATA.txt describing the exact scope. The repository's
original root LICENSE, README, CITATION.cff, MANIFEST.sha256, code and configurations
remain unchanged. Original ISIC images/masks, manuscript sources, author portraits,
credentials and local workspace audit notes are not included.
