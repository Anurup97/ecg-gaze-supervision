# Reproduction plan: final and reviewer-directed experiments

**Working directory:** run Python from the root where `ptb-xl/`, `eye_tracking/`, `definitive_clean_gaze/` and `final_digitisation_v9/` are visible as sibling folders. Scripts use relative paths from that root, as they did in the author's original notebooks.

Before any training: run `python scripts/audit_manifest_consistency.py` to cross-check the committed JSON manifests. Then `python scripts/00_verify_frozen_inputs.py` checks the source data, pseudo-waveform, gaze targets, and base model against `config/step1_manifest.json`. The raw files are deliberately not in the GitHub repository.

## Canonical original notebook sequence

1. Source reconstruction was frozen as V9; `scripts/reconstruction/00_v9_final_lbbb_patch.py` is historical provenance and cannot be rerun from scratch without missing V8/final anchor JSON inputs.
2. `notebooks/final/01_prepare_frozen_targets_original.ipynb` performed source target construction and frozen-manifest audit. Its code contains both target-assembly and follow-on conditioning steps; do not assume the extracted `01b` file runs independently.
3. `scripts/core/02_heldout_gaze_cv.py` trains five cross-validation models per population on eight gaze-labelled ECGs with the other two held out. Original seed 2026, 25 epochs, 40 noise copies/ECG, only `conv2d_10` trainable, fixed BatchNorm.
4. `scripts/core/03_full_gaze_finetuning.py` trains full-data consultant and all-clinician models, and `03b_freeze_model_hashes.py` records hashes.
5. `scripts/core/04a`—`04d` audit PTB-XL fold 10, calculate source-domain performance (2,198 ECGs) and patient-cluster bootstrap.
6. `scripts/core/05a`—`05d` implement the exploratory LUDB representation transfer. The manuscript places these in Supplementary Material.
7. `scripts/controls/07_matched_controls_original_notebook.py` contains original reviewer controls including the mean-gaze baseline, uniform and one fixed permuted target. It is a historical original long notebook cell, so inspect outputs/parameters before running it again.
8. `scripts/controls/08_repeated_permutation_sensitivity.py` performs ten further unique frozen derangements per gaze population. It expects the earlier reviewer-control outputs in `definitive_clean_gaze/reviewer_controls/`.
9. `scripts/controls/09_lead_occlusion_validation.py` reads the frozen base model and PTB-XL fold-10 cache; it does not train a model.
10. **Combined diagnostic + gaze loss:** `scripts/controls/10_combined_loss_executed.py` is the **actual code cell** from the successful final notebook (also preserved as `notebooks/final/03_combined_diagnostic_gaze_loss_executed.ipynb`, stripped of outputs). It uses PTB-XL folds 1–9 (19,601 ECGs), frozen λ 27.10707427, four epochs, six training arms, trainable `conv2d_10` onward except frozen BN, and untouched fold 10. This notebook execution *resumed* `diagnostic_only` weights and trained the remaining five arms. Full console output and parsed summaries are in `results/`; original manifest is now committed under `config/manifests/reviewer/`; original weights and datasets still must be supplied separately for an independent rerun.

The original authors used only the 10 gaze ECGs for gaze objectives, separate from diagnostic PTB-XL folds 1–9. LUDB n=200 was transfer analysis and not involved in the joint-loss optimisation.

## Software environment

`requirements.txt` lists dependencies derived from imports in the uploaded analysis scripts. It does *not* reconstruct a locked historical environment. For an accurate release, record Python, TensorFlow/Keras, NumPy, WFDB, sklearn etc. versions from the environment used to generate the manuscript figures and tables.

## Validation status

All extracted `.py` modules have been syntax-checked. They have **not been executed end-to-end** in this isolated packaging environment because original ECG waveforms, model weights, gaze targets and other inputs were not uploaded. Avoid stating they have been independently reproduced until such a run succeeds.

## Manifest placement versus original on-disk paths

The JSONs in `config/` are verbatim archival copies. Their names and original data-root destinations are mapped in [`docs/manifest_index.md`](manifest_index.md). For example, the original notebook expects `definitive_clean_gaze/step2_cv/step2_manifest.json` on the *local project working directory*, whereas the public archival copy is `config/step2_manifest.json`. A reader must restore files to those paths, rather than assume `config/` replaces the working data directory. Copy files as needed; never overwrite an existing author-generated manifest without checking its hash and provenance.


## Restoring reviewer manifests from public archive

The original uploaded JSON contents are preserved verbatim, including both SHA-256-verified manifests:

- Copy `config/manifests/reviewer/combined_loss_manifest.json` to `definitive_clean_gaze/reviewer_controls/combined_loss_experiment/combined_loss_manifest.json` before running the executed joint-loss script. The script resumes the frozen λ and checks the existing manifest rather than redesigning the experiment.
- Copy `config/manifests/reviewer/permutation_manifest.json` to `definitive_clean_gaze/reviewer_controls/repeated_permutation_sensitivity/permutation_manifest.json` before rerunning the repeated-permutation analysis. Use the original 10 mappings; do not regenerate new ones after viewing model results.

**Do not overwrite** already-existing local originals unless their raw SHA-256 matches the values in `docs/manifest_index.md`. The repository contains the analysis sources and these authentic configuration manifests, but cannot reproduce the entire study without the external datasets, model and prepared input artifacts.
