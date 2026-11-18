# Manifest index and consistency audit

Fifteen author-supplied JSON manifest/configuration files are deposited **verbatim** (no content edits). The table maps the public repository copy to the intended original analysis stage. Paths described as “verify local basename” are organisational examples, not assertions that the original workflow used exactly that filename at that path.

| Uploaded file | Repository path | Role | Original working directory context |
|---|---|---|---|
| `step1_manifest.json` | `config/step1_manifest.json` | Definitive Step 1: gaze targets and frozen input checksums | `definitive_clean_gaze/step1_manifest.json` |
| `step2_manifest.json` | `config/step2_manifest.json` | Definitive Step 2: five-fold held-out gaze fine-tuning | `definitive_clean_gaze/step2_cv/step2_manifest.json` |
| `final_model_manifest.json` | `config/final_model_manifest.json` | Definitive Step 3: final full-data gaze-only model weights | `definitive_clean_gaze/final_models/final_model_manifest.json` |
| `ptbxl_audit_manifest.json` | `config/manifests/ptbxl/ptbxl_audit_manifest.json` | Step 4A: PTB-XL metadata, labels, and partition audit | `definitive_clean_gaze/ptbxl_evaluation/ (verify local basename)` |
| `ptbxl_evaluation_manifest.json` | `config/manifests/ptbxl/ptbxl_evaluation_manifest.json` | Step 4B: frozen PTB-XL fold-10 evaluation | `definitive_clean_gaze/ptbxl_evaluation/ (verify local basename)` |
| `ptbxl_patient_cluster_bootstrap_manifest.json` | `config/manifests/ptbxl/ptbxl_patient_cluster_bootstrap_manifest.json` | Step 4C: PTB-XL patient-level bootstrap | `definitive_clean_gaze/ptbxl_evaluation/ (verify local basename)` |
| `ptbxl_reviewer_control_bootstrap_manifest.json` | `config/manifests/reviewer/ptbxl_reviewer_control_bootstrap_manifest.json` | Reviewer-control PTB-XL patient bootstrap | `definitive_clean_gaze/reviewer_controls/ (verify local basename)` |
| `reviewer_control_manifest.json` | `config/manifests/reviewer/reviewer_control_manifest.json` | Reviewer matched controls: uniform and primary fixed permuted targets | `definitive_clean_gaze/reviewer_controls/ (verify local basename)` |
| `ludb_audit_manifest.json` | `config/manifests/ludb/ludb_audit_manifest.json` | Step 5A: LUDB provenance and endpoint audit | `definitive_clean_gaze/ludb_evaluation/ (verify local basename)` |
| `ludb_step5b_manifest.json` | `config/manifests/ludb/ludb_step5b_manifest.json` | Step 5B: frozen LUDB endpoint definitions | `definitive_clean_gaze/ludb_evaluation/ (verify local basename)` |
| `ludb_step5c_manifest.json` | `config/manifests/ludb/ludb_step5c_manifest.json` | Step 5C: LUDB frozen-representation transfer | `definitive_clean_gaze/ludb_evaluation/ (verify local basename)` |
| `ludb_step5d_manifest.json` | `config/manifests/ludb/ludb_step5d_manifest.json` | Step 5D: LUDB record-level bootstrap | `definitive_clean_gaze/ludb_evaluation/ (verify local basename)` |
| `combined_loss_manifest.json` | `config/manifests/reviewer/combined_loss_manifest.json` | Reviewer joint diagnostic–gaze training: fixed λ, trainable tail, six trained arms, and PTB-XL folds | `definitive_clean_gaze/reviewer_controls/combined_loss_experiment/combined_loss_manifest.json` |
| `permutation_manifest.json` | `config/manifests/reviewer/permutation_manifest.json` | Ten additional prespecified fixed-point-free gaze derangements; both populations use identical mappings | `definitive_clean_gaze/reviewer_controls/repeated_permutation_sensitivity/permutation_manifest.json` |
| `experiment_config.json` | `config/archive_exploratory/experiment_config.json` | HISTORICAL exploratory config; seed 42; NOT definitive | `historical local workflow (not applicable)` |

## Provenance and contradictions

- `step1_manifest.json` in the initial repository was semantically identical to the new author-uploaded version, but differed **only** by the presence of a final newline. The uploaded bytes have now been used. This changes the SHA-256 of the JSON *file itself* but does not change any experimental field or the frozen data hashes inside it.
- `experiment_config.json` records **seed 42**, an earlier image-based extraction method and historical PTB-XL out-of-fold probing; the definitive 2026 experiments instead use **seed 2026**, audited V9 inputs, PTB-XL fold-10 evaluation, and the Step-2/3 frozen settings. The historical file is excluded from the active experiment configuration.
- LUDB bootstrap uses **5,000** record-level resamples; PTB-XL main/reviewer bootstrap uses **3,000** patient-level resamples. These numbers differ intentionally across analyses.
- The combined-loss experiment is **separate** from original gaze-only Step 3. Its executed source/log documents 4 diagnostic-training epochs and fixed λ=27.10707427. The **original on-disk `combined_loss_manifest.json` is now included**, byte-for-byte; its SHA-256 matches the executed notebook log. It is separate from the Step-3 gaze-only manifest.
- The repeated-permutation sensitivity experiment references a 10-derangement frozen manifest hash and the **original `permutation_manifest.json` is now included**, with an exact fingerprint match. The reviewer-control manifest documents the *primary fixed* derangement; the newly included `permutation_manifest.json` records all ten additional derangements, their mappings, and their prior freeze status.
- Hashes in the manifest files attest to the identity of inputs and weights recorded at execution. They cannot themselves authenticate absent waveform files, weights, or dataset access in the repository.

## Cross-manifest automated validation

Run `python scripts/audit_manifest_consistency.py` from the repository root. It uses Python standard-library modules only and checks internal agreement of the provided JSONs, without training or loading ECG data.

## Previously missing frozen manifests: resolved

Both original author-uploaded manifests are present and matched **exactly** the SHA-256 fingerprints printed in the experiment logs:

1. `combined_loss_manifest.json`: `1f4f86ba41dae20f3d6387accc04b7d8a322919b6688a62cc4af3b103623ef27`.
2. `permutation_manifest.json`: `3d6a6bb721c0a9e88d426af3fefad87789fd81f34315cb5aefee2e5a108d68db`.

The automated audit also verifies the joint-loss λ formula, fixed derangement against the original reviewer-control manifest, and each additional permutation against its named mapping. The remaining barriers to **independent end-to-end execution** are access to source datasets, frozen reconstructed V9 inputs/full reconstruction anchors, upstream model file and (for exact reproductions) relevant saved weights and an exact dependency lock. Full original result CSVs were not provided; captured summary outputs are retained.

## Uploaded file integrity (raw SHA-256)

| File | SHA-256 |
|---|---|
| `step1_manifest.json` | `f3240c74fb78dcaf7a79d18b32d1505602334da8dacf87a0bfb96dc3f5c58933` |
| `step2_manifest.json` | `d9d45aa44288f2b48c96c34430a91afd18a77cb7f84a78bceb3ae80a73b74af6` |
| `final_model_manifest.json` | `b1912acd4a3eb8c707cde3b32171af482d1cfd289287de4608514a01d305ff72` |
| `ptbxl_audit_manifest.json` | `ab5b84370051fbcbb8503f6e1749b365296c0d6906682406b7942b767713f7ca` |
| `ptbxl_evaluation_manifest.json` | `3d587bcdb15585f434df19edbd96d4a424dc454a43acf882f7ca9aa515f0c306` |
| `ptbxl_patient_cluster_bootstrap_manifest.json` | `7f4806d22a2554d8eab2c6a04dd2e73a196f19b682dbc9d5dd576937904acb68` |
| `ptbxl_reviewer_control_bootstrap_manifest.json` | `8b9ba7b09f3b7a14fa9cbf1eee456f8bd18b3cfec14cd70b5c3233eb186560c4` |
| `reviewer_control_manifest.json` | `811e78aee5e3ffe2326ef707b8513c4ef6f45d60c7e45d21b5303f374c302b0f` |
| `ludb_audit_manifest.json` | `fb54982ba6baf88e8b2cb5327ca06b9741416dfc7103bd97d1adaccfe0570c1a` |
| `ludb_step5b_manifest.json` | `c0038f61a81a8bcb4f93ab59e08821ce65e0c8f842bf9dd7234de93ebc6fd9c9` |
| `ludb_step5c_manifest.json` | `5e8bc79e17d8ccd4a499baa87e47fe146869236d63ccf333f4cd21769b19eada` |
| `ludb_step5d_manifest.json` | `2944a4146762a3de015b531443ea5a546a5efae18923e4f5a071e8b3b2ebbc49` |
| `combined_loss_manifest.json` | `config/manifests/reviewer/combined_loss_manifest.json` | Reviewer joint diagnostic–gaze training: fixed λ, trainable tail, six trained arms, and PTB-XL folds | `definitive_clean_gaze/reviewer_controls/combined_loss_experiment/combined_loss_manifest.json` |
| `permutation_manifest.json` | `config/manifests/reviewer/permutation_manifest.json` | Ten additional prespecified fixed-point-free gaze derangements; both populations use identical mappings | `definitive_clean_gaze/reviewer_controls/repeated_permutation_sensitivity/permutation_manifest.json` |
| `experiment_config.json` | `a70c4d8177bb9b25f6691ab82bd9026164ec685399c2dcc285ecfd9345f34cbb` |

| `combined_loss_manifest.json` | `1f4f86ba41dae20f3d6387accc04b7d8a322919b6688a62cc4af3b103623ef27` |
| `permutation_manifest.json` | `3d6a6bb721c0a9e88d426af3fefad87789fd81f34315cb5aefee2e5a108d68db` |
