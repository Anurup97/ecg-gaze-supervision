"""
EHJ-DH reviewer sensitivity analysis: repeated permuted-gaze controls
===================================================================

Purpose
-------
Reviewer 1 asked why only one fixed gaze permutation was used. This script
pre-specifies 10 additional UNIQUE derangements, then reruns the exact full-data
control-training regime for both consultant and all-clinician gaze targets.

IMPORTANT
---------
* Set N_PERMUTATIONS BEFORE running and do not change it after seeing results.
* The permutation manifest is written BEFORE any model is trained and is reused
  on restart. This prevents post-hoc permutation selection.
* The script is checkpointed/resumable. Completed model/metric pairs are skipped.
* It does NOT alter the original fixed-permutation result; it is a sensitivity
  analysis around that prespecified primary control.

Expected project paths are the same as the completed analysis.
"""

from pathlib import Path
import gc
import hashlib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from sklearn.metrics import roc_auc_score, average_precision_score

# ======================================================================================
# 1. PRE-SPECIFIED SETTINGS
# ======================================================================================
SEED = 2026
N_PERMUTATIONS = 10       # pre-specify now; do not change after inspecting results
AUGMENTATIONS_PER_ECG = 40
NOISE_SD = 0.03
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
RMS_EPSILON = 1e-8
PRED_BATCH_SIZE = 64
SAVE_WEIGHTS = True

ECG_ORDER = [
    "NSR", "AFib", "Hyperkalemia", "AFlutter", "VTach",
    "WPW", "VentPaced", "LBBB", "STEMI", "CompleteHeartBlock",
]

MODEL_CLASSES = [
    "AMI", "CLBBB", "CRBBB", "ILBBB", "IMI", "IRBBB", "ISCA", "ISCI",
    "ISC_", "IVCD", "LAFB/LPFB", "LAO/LAE", "LMI", "LVH", "NORM",
    "NST_", "PMI", "RAO/RAE", "RVH", "SEHYP", "STTC", "WPW", "_AVB",
]

POPULATIONS = ["consultant", "all_clinicians"]

# ======================================================================================
# 2. PATHS
# ======================================================================================
ROOT = Path("definitive_clean_gaze")
X_PATH = Path("final_digitisation_v9/X_gaze_final.npy")
TARGET_PATH = ROOT / "clean_gaze_targets.npz"
STEP1_MANIFEST_PATH = ROOT / "step1_manifest.json"
STEP2_MANIFEST_PATH = ROOT / "step2_cv" / "step2_manifest.json"

REVIEWER_DIR = ROOT / "reviewer_controls"
COMBINED_PRED_PATH = REVIEWER_DIR / "ptbxl_all_models_predictions.csv"
ORIGINAL_PRED_PATH = ROOT / "ptbxl_evaluation" / "ptbxl_fold10_predictions.csv"
PRIMARY_MAPPING_PATH = REVIEWER_DIR / "permuted_gaze_mapping.csv"

PTBXL_ROOT = Path("ptb-xl")
PTBXL_DATABASE_PATH = PTBXL_ROOT / "ptbxl_database.csv"
PTB_X_CACHE = REVIEWER_DIR / "ptbxl_fold10_X.npy"
PTB_ID_CACHE = REVIEWER_DIR / "ptbxl_fold10_ecg_ids.npy"

OUT = REVIEWER_DIR / "repeated_permutation_sensitivity"
MODEL_DIR = OUT / "models"
PRED_DIR = OUT / "predictions"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUT / "permutation_manifest.json"
METRICS_PATH = OUT / "permutation_metrics.csv"
SUMMARY_PATH = OUT / "permutation_sensitivity_summary.csv"

# ======================================================================================
# 3. REPRODUCIBILITY
# ======================================================================================
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
    print("TensorFlow deterministic operations enabled.")
except Exception as e:
    print("Could not explicitly enable deterministic operations:", repr(e))

# ======================================================================================
# 4. HELPERS
# ======================================================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    original = np.arange(n)
    for _ in range(100000):
        p = rng.permutation(original)
        if np.all(p != original):
            return p
    raise RuntimeError("Could not generate derangement.")


def attention_distribution(add3_tensor):
    rms = tf.sqrt(
        tf.reduce_mean(tf.square(add3_tensor), axis=[2, 3]) + RMS_EPSILON
    )
    return rms / (tf.reduce_sum(rms, axis=1, keepdims=True) + RMS_EPSILON)


def load_fresh_trainable_model(base_model_path: Path):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = tf.keras.models.load_model(base_model_path, compile=False)
    for layer in model.layers:
        layer.trainable = False
    conv10 = model.get_layer("conv2d_10")
    conv10.trainable = True
    rep_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("add_3").output,
    )
    train_vars = list(conv10.trainable_variables)
    if not train_vars:
        raise RuntimeError("conv2d_10 has no trainable variables.")
    return model, rep_model, train_vars


def train_control(base_model_path, X_aug, target_matrix, epoch_permutations, weight_path):
    model, rep_model, train_vars = load_fresh_trainable_model(base_model_path)
    Y_aug = np.repeat(target_matrix, AUGMENTATIONS_PER_ECG, axis=0).astype(np.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    history = []

    for epoch in range(EPOCHS):
        losses = []
        perm = epoch_permutations[epoch]
        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]
            xb = tf.convert_to_tensor(X_aug[idx], dtype=tf.float32)
            yb = tf.convert_to_tensor(Y_aug[idx], dtype=tf.float32)
            with tf.GradientTape() as tape:
                add3 = rep_model(xb, training=False)
                pred = attention_distribution(add3)
                loss = tf.reduce_mean(tf.square(pred - yb))
            grads = tape.gradient(loss, train_vars)
            if any(g is None for g in grads):
                raise RuntimeError("None gradient encountered.")
            optimizer.apply_gradients(zip(grads, train_vars))
            losses.append(float(loss.numpy()))
        history.append(float(np.mean(losses)))

    if SAVE_WEIGHTS:
        model.save_weights(weight_path)
    pd.DataFrame({"epoch": np.arange(1, EPOCHS+1), "loss": history}).to_csv(
        weight_path.with_suffix(".history.csv"), index=False
    )
    return model, history


def metrics(y_true, y_score):
    aucs, aps = [], []
    for j in range(y_true.shape[1]):
        aucs.append(roc_auc_score(y_true[:, j], y_score[:, j]))
        aps.append(average_precision_score(y_true[:, j], y_score[:, j]))
    return {
        "macro_AUROC": float(np.mean(aucs)),
        "macro_AP": float(np.mean(aps)),
        "micro_AUROC": float(roc_auc_score(y_true.ravel(), y_score.ravel())),
        "micro_AP": float(average_precision_score(y_true.ravel(), y_score.ravel())),
    }

# ======================================================================================
# 5. VERIFY ORIGINAL PIPELINE AND LOAD INPUTS
# ======================================================================================
for p in [X_PATH, TARGET_PATH, STEP1_MANIFEST_PATH, STEP2_MANIFEST_PATH]:
    if not p.exists():
        raise FileNotFoundError(p)

with open(STEP1_MANIFEST_PATH) as f:
    step1 = json.load(f)
with open(STEP2_MANIFEST_PATH) as f:
    step2 = json.load(f)

BASE_MODEL_PATH = Path(step1["base_model"])
if not BASE_MODEL_PATH.exists():
    raise FileNotFoundError(BASE_MODEL_PATH)

required = {
    "attention_layer": "add_3",
    "trainable_layer": "conv2d_10",
    "epochs": 25,
    "augmentations_per_training_ECG": 40,
    "gaussian_noise_sigma": 0.03,
    "batch_size": 16,
}
for k, v in required.items():
    if step2[k] != v:
        raise RuntimeError(f"Frozen pipeline mismatch: {k}={step2[k]} expected {v}")

X = np.load(X_PATH).astype(np.float32)
td = np.load(TARGET_PATH)
Y = {
    "consultant": td["consultant"].astype(np.float32),
    "all_clinicians": td["all_clinicians"].astype(np.float32),
}
if X.shape != (10, 12, 1000, 1):
    raise RuntimeError(f"Unexpected gaze ECG input shape: {X.shape}")

# ======================================================================================
# 6. PRE-SPECIFY / REUSE UNIQUE PERMUTATIONS BEFORE TRAINING
# ======================================================================================
primary_tuple = None
if PRIMARY_MAPPING_PATH.exists():
    primary = pd.read_csv(PRIMARY_MAPPING_PATH)
    name_to_idx = {n:i for i,n in enumerate(ECG_ORDER)}
    primary_tuple = tuple(name_to_idx[x] for x in primary["assigned_gaze_target"])

if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    if manifest["n_permutations"] != N_PERMUTATIONS:
        raise RuntimeError(
            "Existing manifest uses a different N_PERMUTATIONS. Do not change the count after "
            "the analysis has been prespecified."
        )
    permutations = [np.asarray(x, dtype=int) for x in manifest["permutations"]]
    print("Reusing pre-existing permutation manifest:", MANIFEST_PATH)
else:
    permutations = []
    seen = set()
    if primary_tuple is not None:
        seen.add(primary_tuple)
    for repeat in range(1, N_PERMUTATIONS + 1):
        rng = np.random.default_rng(SEED + 910000 + repeat)
        while True:
            p = derangement(10, rng)
            t = tuple(int(x) for x in p)
            if t not in seen:
                seen.add(t)
                permutations.append(p)
                break
    manifest = {
        "analysis": "Repeated permuted-gaze sensitivity analysis",
        "seed": SEED,
        "n_permutations": N_PERMUTATIONS,
        "generated_before_training": True,
        "primary_fixed_permutation_excluded_from_new_draws": primary_tuple is not None,
        "permutations": [p.tolist() for p in permutations],
        "mappings": [
            {ECG_ORDER[i]: ECG_ORDER[int(p[i])] for i in range(10)}
            for p in permutations
        ],
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Permutation manifest frozen BEFORE training:", MANIFEST_PATH)
    print("Manifest SHA256:", sha256_file(MANIFEST_PATH))

# ======================================================================================
# 7. RECREATE EXACT FULL-DATA AUGMENTATION + EPOCH ORDER
# ======================================================================================
aug_rng = np.random.default_rng(SEED + 500000)
X_base = np.repeat(X, AUGMENTATIONS_PER_ECG, axis=0).astype(np.float32)
noise = aug_rng.normal(0.0, NOISE_SD, size=X_base.shape).astype(np.float32)
X_aug = (X_base + noise).astype(np.float32)

perm_rng = np.random.default_rng(SEED + 600000)
epoch_permutations = [perm_rng.permutation(len(X_aug)) for _ in range(EPOCHS)]

# ======================================================================================
# 8. LOAD SAVED PTB-XL TARGETS / ORIGINAL MODEL PREDICTIONS
# ======================================================================================
if COMBINED_PRED_PATH.exists():
    saved = pd.read_csv(COMBINED_PRED_PATH)
elif ORIGINAL_PRED_PATH.exists():
    saved = pd.read_csv(ORIGINAL_PRED_PATH)
else:
    raise FileNotFoundError(
        "Need either reviewer_controls/ptbxl_all_models_predictions.csv or original fold-10 predictions."
    )

Y_TEST = np.column_stack([
    saved[f"true__{c}"].to_numpy(dtype=np.int8) for c in MODEL_CLASSES
])
SAVED_IDS = saved["ecg_id"].to_numpy()

# Load/calculate comparator metrics if columns exist
comparator = {}
name_prefixes = {
    "base": "base",
    "real_consultant_gaze": "real_consultant_gaze",
    "real_all_clinician_gaze": "real_all_clinician_gaze",
    "primary_permuted_consultant": "permuted_consultant_control",
    "primary_permuted_all_clinicians": "permuted_all_clinician_control",
}
for label, prefix in name_prefixes.items():
    cols = [f"{prefix}__{c}" for c in MODEL_CLASSES]
    if all(c in saved.columns for c in cols):
        P = np.column_stack([saved[c].to_numpy(float) for c in cols])
        comparator[label] = metrics(Y_TEST, P)

# ======================================================================================
# 9. LOAD / BUILD PTB-XL FOLD-10 CACHE IN SAVED-PREDICTION ORDER
# ======================================================================================
if PTB_X_CACHE.exists() and PTB_ID_CACHE.exists():
    cached_ids = np.load(PTB_ID_CACHE)
    if not np.array_equal(cached_ids, SAVED_IDS):
        raise RuntimeError("Cached PTB-XL ID order mismatch.")
    X_TEST = np.load(PTB_X_CACHE).astype(np.float32)
    print("Loaded cached PTB-XL fold-10 waveforms.")
else:
    if not PTBXL_DATABASE_PATH.exists():
        raise FileNotFoundError(PTBXL_DATABASE_PATH)
    meta = pd.read_csv(PTBXL_DATABASE_PATH, index_col="ecg_id")
    X_TEST = np.empty((len(SAVED_IDS), 12, 1000, 1), dtype=np.float32)
    expected_leads = ["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]
    for i, ecg_id in enumerate(SAVED_IDS):
        row = meta.loc[ecg_id]
        sig, fields = wfdb.rdsamp(str(PTBXL_ROOT / str(row["filename_lr"])))
        leads = [str(x).upper() for x in fields["sig_name"]]
        if sig.shape != (1000,12) or leads != expected_leads:
            raise RuntimeError(f"PTB-XL format mismatch for ECG {ecg_id}")
        X_TEST[i,:,:,0] = sig.T.astype(np.float32)
        if (i+1) % 250 == 0 or (i+1)==len(SAVED_IDS):
            print(f"Loaded PTB-XL {i+1}/{len(SAVED_IDS)}")
    np.save(PTB_X_CACHE, X_TEST)
    np.save(PTB_ID_CACHE, SAVED_IDS)

# ======================================================================================
# 10. RESUME-SAFE TRAINING / INFERENCE
# ======================================================================================
if METRICS_PATH.exists():
    metrics_df = pd.read_csv(METRICS_PATH)
else:
    metrics_df = pd.DataFrame()

for repeat, mapping in enumerate(permutations, start=1):
    print("\n" + "="*110)
    print(f"PERMUTATION {repeat}/{N_PERMUTATIONS}")
    print("Mapping:", {ECG_ORDER[i]: ECG_ORDER[int(mapping[i])] for i in range(10)})

    for population in POPULATIONS:
        already_done = False
        if len(metrics_df):
            already_done = np.any(
                (metrics_df["repeat"] == repeat) &
                (metrics_df["population"] == population)
            )
        if already_done:
            print(f"SKIP completed: repeat {repeat} / {population}")
            continue

        weight_path = MODEL_DIR / f"perm{repeat:02d}_{population}.weights.h5"
        pred_path = PRED_DIR / f"perm{repeat:02d}_{population}.npz"

        if pred_path.exists():
            print("Loading existing predictions:", pred_path)
            P = np.load(pred_path)["probabilities"].astype(np.float64)
        else:
            targets = Y[population][mapping]
            print(f"Training repeat {repeat} / {population} ...")
            model, hist = train_control(
                BASE_MODEL_PATH, X_aug, targets, epoch_permutations, weight_path
            )
            print(f"Loss: {hist[0]:.8f} -> {hist[-1]:.8f}")
            print("PTB-XL inference ...")
            P = model.predict(X_TEST, batch_size=PRED_BATCH_SIZE, verbose=1).astype(np.float64)
            np.savez_compressed(pred_path, probabilities=P)
            del model
            gc.collect()

        m = metrics(Y_TEST, P)
        row = {
            "repeat": repeat,
            "population": population,
            "mapping": json.dumps(mapping.tolist()),
            **m,
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        metrics_df.to_csv(METRICS_PATH, index=False)
        print("Saved metrics:", m)

# ======================================================================================
# 11. SENSITIVITY SUMMARY
# ======================================================================================
metrics_df = pd.read_csv(METRICS_PATH)
summary_rows = []
for population in POPULATIONS:
    temp = metrics_df[metrics_df.population == population].copy()
    real_key = "real_consultant_gaze" if population == "consultant" else "real_all_clinician_gaze"
    fixed_key = "primary_permuted_consultant" if population == "consultant" else "primary_permuted_all_clinicians"

    for metric_name in ["macro_AUROC", "macro_AP", "micro_AUROC", "micro_AP"]:
        vals = temp[metric_name].to_numpy(float)
        real_value = comparator.get(real_key, {}).get(metric_name, np.nan)
        fixed_value = comparator.get(fixed_key, {}).get(metric_name, np.nan)
        summary_rows.append({
            "population": population,
            "metric": metric_name,
            "n_permutations": len(vals),
            "permuted_mean": float(np.mean(vals)),
            "permuted_SD": float(np.std(vals, ddof=1)),
            "permuted_median": float(np.median(vals)),
            "permuted_min": float(np.min(vals)),
            "permuted_max": float(np.max(vals)),
            "real_gaze_value": float(real_value) if np.isfinite(real_value) else np.nan,
            "primary_fixed_permutation_value": float(fixed_value) if np.isfinite(fixed_value) else np.nan,
            "real_minus_permuted_mean": float(real_value - np.mean(vals)) if np.isfinite(real_value) else np.nan,
            "fraction_new_permutations_below_or_equal_real": float(np.mean(vals <= real_value)) if np.isfinite(real_value) else np.nan,
            "fraction_new_permutations_above_or_equal_real": float(np.mean(vals >= real_value)) if np.isfinite(real_value) else np.nan,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)

print("\n\n" + "="*120)
print("MANUSCRIPT SENSITIVITY RESULTS — REPEATED PERMUTED-GAZE CONTROLS")
print("="*120)
print(summary.round(6).to_string(index=False))
print("\nRaw metrics:", METRICS_PATH)
print("Summary:", SUMMARY_PATH)
print("Frozen permutation manifest:", MANIFEST_PATH)
print("\nDo NOT regenerate or selectively exclude permutations after inspecting these results.")
