# Changes from the collected server sources

The model, losses, dataset and training loop are copied unchanged from the latest server source snapshot. `artifacts/source_manifest.json` records original SHA-256 values.

Only `evaluate.py` is adjusted: it accepts `--splits-path`, defaults to the locked 1815/259/520 paper split, passes that split to ISICDataset, defaults to Euclidean BF1, and sends default evaluation outputs to outputs/evaluation instead of the original checkpoint directory. The server CLI otherwise used the historical splits.json default and could evaluate the wrong test partition when used standalone.

New release files include this documentation, portable per-run configs, run_experiment.py, a checkpoint verifier, a Euclidean BF1 recomputation entry point, and release-specific tests. Original configs are normalized only to accepted TrainConfig fields and portable paths; common training settings are preserved. Missing historical boundary_distance fields are explicitly labelled chebyshev.

No server source, configuration, result, checkpoint, or dataset was changed. Newly generated code is separately identifiable from collected source. No full training or test-set recomputation was run to make this collection.
