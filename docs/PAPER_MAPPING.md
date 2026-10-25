# Manuscript-to-artifact mapping

The supplied manuscript identifies **EC-free** `Seg + FPFCS + KD` with decoder DS disabled as PicoSemaNet (full). Its 200-epoch-cap primary runs are:

| Seed | Server run | Public config |
| --- | --- | --- |
| 42 | formal_paper_ablation_supplement_v1/s3b_no_edge_seed42 | primary_full_seed42.json |
| 43 | formal_paper_last30h_v1/no_edge_seed43 | primary_full_seed43.json |
| 44 | formal_paper_last30h_v1/no_edge_seed44 | primary_full_seed44.json |

Their original GPU Dice values average to 87.39% (sample SD rounds to 0.09%). The historical `best_under0005m_no_deep` model is the **+EC ablation**, with 87.30% mean Dice, not the primary model.

## Table 1

Use groups `context_teacher`, `unet`, `contextunet_3m`, `microsemanet`, `matched_size_unet`, and `primary_full` in `artifacts/paper_groups.json`. The teacher uses seed42 only; the other groups use seeds42-44.

## Table 2

Use seed42 from `seg_only`, `primary_no_fpfcs`, `legacy_no_fpfcs_ec`, `legacy_shallow_ec`, `legacy_encoder_only_ec`, `legacy_learned_projection_ec`, `legacy_no_kd_ec`, `legacy_deep_supervision_ec`, `ec_ablation`, and `primary_full`.

## Table 3

Use `matched100_full` and `matched100_no_fpfcs` for the exactly-100-epoch comparison; `primary_full` and `ec_ablation` for the primary configuration and EC ablation; `legacy_no_fpfcs_ec`, `legacy_shallow_ec`, `legacy_learned_projection_ec`, `legacy_budget_control_ec`, `legacy_no_boundary_ec`, `legacy_no_texture_ec`, `legacy_no_distance_ec`, and `legacy_no_kd_ec` for the labelled legacy studies.

The three target-removal groups retain EC and have 100-epoch caps; the paired100 groups have exactly100 epochs. Never pool these budgets.

## BF1 and unavailable later artifacts

All historical server files are preserved with their original BF1 values. Except for matched100, these are Chebyshev values and do not equal the manuscript's recomputed Euclidean BF1. Later per-image CPU-recomputed BF1 outputs, the exact CPU recomputation environment, paper figure scripts, and convolution/MAC-counting scripts were not found in this server project. The provided new BF1 entry point computes the same documented distance rule, but is not labelled as the missing original script or output.

The manuscript PDF is not included in the public code directory. Its title and SHA-256 are recorded in `artifacts/manuscript.json` for identification.
