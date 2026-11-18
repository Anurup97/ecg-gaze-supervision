# ECG gaze supervision: matched-control proof-of-concept

**Research code deposit — original analysis sources and all 15 experiment manifests; not independently rerun end-to-end.**

This repository organises the original analysis code supporting the manuscript **“Clinician gaze as a supervisory signal for electrocardiogram deep learning: a matched-control proof-of-concept study”** (Mukherjee, Shergill and Ang; manuscript under review). It concerns clinician eye-gaze alignment of a pretrained ECG CNN and the methodological need to distinguish **target engagement**, **case-specific gaze learning** and **diagnostic preservation**.

## What's in the repository

| Location | Contents |
|---|---|
| `notebooks/final/` | Original target-building, core-analysis and **executed combined-loss** notebooks, outputs stripped |
| `scripts/core/` | Numbered, verbatim extractions of original notebook cells (CV, full training, PTB-XL, LUDB, figures) |
| `scripts/controls/` | Original reviewer matched controls, repeated-permutation and lead-occlusion scripts, and the **actual executed combined-loss source** |
| `scripts/reconstruction/` | Historical final V9 reconstruction patch (not a complete from-scratch reconstruction) |
| `config/` | **15 author-supplied JSON manifests:** frozen original experiment (Steps 1–3), PTB-XL/LUDB evaluations, reviewer controls, and an explicitly segregated historical configuration; see `docs/manifest_index.md` |
| `annotations/legacy/` | Historical image annotation configuration; **not complete final V9 annotations** |
| `results/` | Non-patient-level point estimates, training gaze-fit and bootstrap comparisons parsed from the **executed final notebook output** |
| `docs/` | Reproduction sequence, data sources, audit, missing assets, GitHub/Zenodo publishing |

## Core experimental settings

- **10** gaze-labelled ECG stimuli, each representing a distinct clinical category.
- Consultant gaze: 9 readers; all-clinician gaze: 20 readers.
- `add_3` activation-derived 12-lead profile; **not** a causal attention mechanism.
- Original gaze-only fine-tuning: `conv2d_10` only, 25 epochs, Adam LR `1e-4`, 40 Gaussian-noise copies per ECG, 5 ECG-level CV folds.
- PTB-XL v1.0.3 stratified fold 10: 2,198 ECGs / 1,904 patients, 23 diagnostic subclasses; source-domain held-out evaluation.
- Post-review matched controls: training-fold mean gaze, uniform targets, incorrect gaze mappings, ten further derangements per population, and a lead-zeroing test.
- Post-review joint-loss experiment: PTB-XL development folds 1–9 (19,601 ECGs), training with BCE and fixed-weight gaze MSE; executed code now included; 4 PTB epochs, frozen λ=27.10707427, six fitted arms.

## Quick start

1. Use Python 3.11 or a version verified to work with your TensorFlow installation. Install the dependencies in `requirements.txt`. On Apple silicon, use a supported TensorFlow environment.
2. Obtain the datasets and pretrained model from their original sources, and lay out local folders as recorded in `docs/data_sources.md` (do not commit them).
3. Verify that your local files match the frozen manifest: `python scripts/00_verify_frozen_inputs.py`.
4. Read `docs/reproduction.md` to run the corresponding final scripts and notebook cells in their original project-root working directory.

**Important:** the provided `requirements.txt` is a compatibility guide, not a recreation of the exact historical environment. Original raw inputs, the base model, and the raw ECG/model inputs and exact environment lock are absent. This deposit should not be described as a fully executable end-to-end reproduction without those additions and a validation run.

## Manifest audit and remaining provenance gaps

Fifteen author-supplied JSON files were integrated without altering their contents. A static cross-manifest consistency check passed all 27 specified checks for the original frozen inputs, reader populations, training settings, diagnostic class ordering, PTB-XL and LUDB splits, and evaluation manifests. This does **not** establish that the omitted raw inputs, saved weights, and exact code environment can be independently reproduced. See [`docs/manifest_index.md`](docs/manifest_index.md) and run `python scripts/audit_manifest_consistency.py`.

## Final combined-loss source and remaining provenance gaps

The author's **executed** combined-loss notebook is now included as an output-stripped notebook and an exact `.py` extraction. Its captured log and derived summary CSVs are in `results/`; see [detailed provenance](results/combined_loss_run_provenance.md). The `diagnostic_only` arm was resumed from completed saved weights in this notebook run.

Now included: author-supplied `step1_manifest.json`, `step2_manifest.json`, `final_model_manifest.json`, PTB-XL/LUDB evaluation and reviewer-control manifests. **Both previously missing frozen reviewer manifests are now present and verified by their original recorded SHA-256 hashes.** Still needed for an independent end-to-end rerun: full original result CSVs, complete final V9 reconstruction annotations, external input/model files and exact environment lock. They should be supplied or described honestly before making an **end-to-end reproducibility** claim. This package is a **transparent code deposit with recorded results**, not an independently rerun reproduction.

## Licence and data restrictions

The original analysis code is distributed under the [MIT License](LICENSE); this does **not** license PTB-XL, the eye-tracking dataset, LUDB, upstream model weights or any third-party code under separate terms. Check upstream redistribution permissions before uploading a public release. This code package intentionally excludes the uploaded MIMIC-style record-list and machine-measurement CSVs containing study/subject IDs, clinical text and timestamps; they are not part of this study's code release.

## Related documents

- [Source-file audit and missing assets](docs/source_inventory.md)
- [Reproduction sequence](docs/reproduction.md)
- [Original datasets/model](docs/data_sources.md)
- [GitHub and Zenodo publishing](docs/publish_github_zenodo.md)
