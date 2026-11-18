# Source-file audit and publication decisions

The uploaded archive (`Code and JSON for eye-gaze.zip`) contained **10 notebook files**, numerous JSONs, and two very large CSVs. The two CSVs were excluded from this proposed software repository:

- `record_list.csv` (~70 MB uncompressed), a MIMIC-style ECG record index with subject/study IDs and dates;
- `machine_measurements.csv` (~183 MB uncompressed), containing medical record IDs, machine ECG reports, measurements and timestamps.

These files are unrelated to the ten-image gaze study's public code deposit, may be subject to clinical-data terms, and should not be shared publicly as part of this code release. No rows from them are included here.

## Relevant authoritative originals

- `New training 30.08.ipynb`: source input audit, target generation and frozen manifest.
- `latest gaze training 31.08.ipynb`: held-out cross-validation, final models, PTB-XL evaluation, LUDB transfer, figure scripts.
- `extra analysis.ipynb`: original reviewer control experiment / training-fold mean-gaze comparisons.
- `repeated_permutation_sensitivity.py`: post-review repeated derangements, copied from the earlier finished analysis file available in this conversation.
- `activation_profile_occlusion_validation.py`: the lead-zeroing check, copied likewise.
- `Untitled3.ipynb`: **actual executed six-arm combined-loss script plus recorded console output**, added after the original ZIP audit; preserved output-free under `notebooks/final/` and as an exact code-source extraction under `scripts/controls/`.

Each `scripts/core/*.py` is extracted unmodified from the named original notebook code cell and retains its original assumptions. The notebooks are also preserved in `notebooks/final/` after removing outputs and execution numbers.

## Archive / historical workflows

Older `16.08.26`, `28.08.26`, `Re-analysis`, manual-annotation and QC notebooks are under `notebooks/archive_exploratory/` for provenance only. They **are not** a recommended alternative execution pipeline. All saved outputs were removed. Hard-coded personal home-directory strings in historical notebooks were redacted to `LOCAL_USER_HOME` and therefore those historical notebooks are not runnable unmodified; this was done intentionally before prospective public release.

`experiment_config.json` was not copied into active config because it includes **seed 42**, earlier processing and evaluations that were superseded by the frozen definitive manifest (**seed 2026**).

`metadata.json` and its numbered variants were not included because they refer to historical intermediate reconstructions and one contains a personal absolute path. The `step1_manifest.json` is the authoritative frozen input record.

`ecg_anchor_annotations_targeted.json` is a historical patch annotation file, **not** proof of inclusion of every final reconstruction annotation. The final LBBB V2 anchor JSON was not present in the uploaded archive.

## Remaining for a fully reproducible final publication release

1. The **actual combined-loss source is now provided**. The notebook output shows that `diagnostic_only` loaded a completed checkpoint; its original training history/weights were not supplied.
2. Original on-disk `combined_loss_manifest.json` and full original CSV outputs, if available. Exact values were transcribed from the **actual executed notebook log** into `results/`; see `results/combined_loss_run_provenance.md`. The reported manifest SHA is **not** a replacement for the original manifest.
3. Primary fixed permutation and train/split manifests, plus additional `permutation_manifest.json`, to audit frozen randomisation without retraining.
4. Original frozen V9 reconstructed inputs and final digitisation annotations, where redistribution rights permit. The requested `step2_manifest.json` is now included under `config/`.
5. Required external PTB-XL v1.0.3, gaze v1.0.0, LUDB, and base Elyamani CNN weights: download separately, and do not commit third-party data/weights without checking their terms.
6. Exact environment versions, software licence, coauthor approval and an actual independently executed verification on the necessary inputs.

The GitHub package is therefore **an auditable source-code deposit with a captured successful joint-loss run, not yet a full independent reproduction**.

## Manifests added after the first repository draft

The author subsequently supplied 13 standalone JSON files. These are preserved byte-for-byte in their final directory locations shown in `docs/manifest_index.md`. They add the original Step-2 and Step-3 audit trail, model and PTB-XL checksums, LUDB evaluation provenance and matched reviewer-control definitions. `experiment_config.json` is intentionally in `config/archive_exploratory/` because its seed 42, early digitisation recipe and PTB-XL evaluation plan do not define the definitive analysis.

The combined-loss output log includes a SHA-256 for `combined_loss_manifest.json` but **the JSON itself was not supplied**. Likewise, the ten-derangement log has a frozen manifest hash, but `permutation_manifest.json` itself was not supplied. These were not reconstructed, renamed or fabricated for this deposit.
