# Executed combined-loss notebook: provenance and scope

The original notebook `Untitled3.ipynb` was supplied by the study author after the initial repository draft.
Its **one nonempty code cell** is preserved byte-for-byte (source text) in
`scripts/controls/10_combined_loss_executed.py`, and also in the output-stripped
`notebooks/final/03_combined_diagnostic_gaze_loss_executed.ipynb`.

- Exact code-cell source SHA-256 (UTF-8): `9033560f383bc903a785e4f3787c0ad5b2d3292d7b1ac4f179a4ea91fe7d1f90`
- Notebook user environment metadata (informational): Python 3.12.7, conda base kernel (stripped from distributable notebook)
- Original executed console output: `results/combined_loss_executed_console.txt`
- Results parsed mechanically from that **executed notebook output**, with no rerun or re-estimation:
  `combined_loss_point_estimates_from_executed_notebook.csv`,
  `combined_loss_training_gaze_fit_from_executed_notebook.csv`, and
  `combined_loss_bootstrap_from_executed_notebook.csv`.

## What the log establishes

The code verifies the base-model, gaze-input, and target checksums; uses 19,601 PTB-XL
folds-1–9 ECGs, 2,198 fold-10 evaluation ECGs, frozen lambda 27.10707427,
4 PTB-XL training epochs (24.64 equivalent gaze passes), learning rate 1e-5,
BatchNorm frozen, `conv2d_10` onward otherwise trainable, and 2,276 valid of 3,000
patient-cluster bootstrap resamples.

**Important resume detail:** this execution loaded **existing completed weights** for the
`diagnostic_only` arm. It trained and logged four epochs for each of the five joint/control
arms. The `diagnostic_only` training *code* is present in the same script, but its original
training console/history was not in this uploaded notebook.

The console reports that `combined_loss_manifest.json` was previously frozen with
SHA-256 `1f4f86ba41dae20f3d6387accc04b7d8a322919b6688a62cc4af3b103623ef27`.
**The original manifest has now been uploaded and preserved verbatim** at `config/manifests/reviewer/combined_loss_manifest.json`, and its bytes match the reported SHA-256 exactly.

## What is not established

Static source checks, captured output and packaging do not establish independent
rerun reproducibility. Saved model weights, raw waveforms, and precise dependency versions remain on the author's local machine.
No results are regenerated or selectively filtered in this repository preparation.
