# Reproducibility status

- Included: latest model/loss/train/evaluation code; locked split and its manifest; per-run portable training configs; original test metrics and epoch logs; original best checkpoints in a separate archive; legacy EC bootstrap inputs and code; source/checkpoint SHA-256 records.
- Primary model: EC=0 and DS=0. The +EC variant is a labelled ablation. All KD experiments use the same seed42 Context Teacher.
- Historical metrics: original GPU Dice/IoU are retained. Historical Chebyshev BF1 must not be used as the paper's CPU Euclidean BF1. New results belong under outputs/ and must be labelled with their execution environment.
- Not found on the server: later CPU Euclidean BF1 result artifacts and exact recomputation environment; paper-specific qualitative-figure scripts; forward-hook MAC-count scripts. These omissions are recorded, not replaced with fabricated results.
- Data images and masks are excluded. The exact split is included and should not be regenerated for reported experiments.
- Legacy bootstrap: bootstrap_ablation.py uses the original EC-containing run layout and seed-averaged paired per-image vectors, 10000 resamples, bootstrap seed20260915. Existing input CSV/JSON files are retained. It does not measure uncertainty across independent training seeds.
- To evaluate the EC-free proposed model use evaluate.py or tools/recompute_euclidean_bf1.py with primary_full checkpoints.
- Original broad dependency ranges are preserved. The original training environment and later CPU evaluation environment differ; numerical equivalence across versions is not established.
- The code uses the MIT License selected by the project owner. Dataset terms remain separate; images and masks are not redistributed. Publication identifiers can be added when available.
