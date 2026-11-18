"""
OPTIONAL construct-validity sanity check for the activation-derived lead profile
===============================================================================

Purpose
-------
Reviewers correctly noted that RMS activation magnitude at add_3 does not
necessarily equal functional diagnostic importance. This optional, no-training
analysis asks whether the base model's 12-lead activation-derived profile is
associated with a simple causal perturbation measure: the change in diagnostic
logits when each lead is zeroed in turn on PTB-XL fold 10.

This does NOT turn the activation profile into a validated attention mechanism.
It is only a functional sanity check that can be reported if useful.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr

SEED = 2026
EPS = 1e-7
PRED_BATCH_SIZE = 64

ROOT = Path("definitive_clean_gaze")
STEP1_MANIFEST_PATH = ROOT / "step1_manifest.json"
REVIEWER_DIR = ROOT / "reviewer_controls"
X_CACHE = REVIEWER_DIR / "ptbxl_fold10_X.npy"
ID_CACHE = REVIEWER_DIR / "ptbxl_fold10_ecg_ids.npy"
PRED_PATH = REVIEWER_DIR / "ptbxl_all_models_predictions.csv"
OUT_DIR = REVIEWER_DIR / "activation_profile_occlusion_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_ECG_PATH = OUT_DIR / "per_ecg_occlusion_concordance.csv"
SUMMARY_PATH = OUT_DIR / "occlusion_concordance_summary.csv"
INFLUENCE_PATH = OUT_DIR / "lead_profiles_and_occlusion_influence.npz"

MODEL_CLASSES = [
    "AMI", "CLBBB", "CRBBB", "ILBBB", "IMI", "IRBBB", "ISCA", "ISCI",
    "ISC_", "IVCD", "LAFB/LPFB", "LAO/LAE", "LMI", "LVH", "NORM",
    "NST_", "PMI", "RAO/RAE", "RVH", "SEHYP", "STTC", "WPW", "_AVB",
]

np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

if not X_CACHE.exists() or not ID_CACHE.exists() or not PRED_PATH.exists():
    raise FileNotFoundError(
        "This optional script expects the cached fold-10 waveforms and combined predictions "
        "created by the reviewer-control analysis. Run that analysis first."
    )

with open(STEP1_MANIFEST_PATH) as f:
    step1 = json.load(f)
BASE_MODEL_PATH = Path(step1["base_model"])

X = np.load(X_CACHE).astype(np.float32)
ecg_ids = np.load(ID_CACHE)
saved = pd.read_csv(PRED_PATH)
if not np.array_equal(ecg_ids, saved["ecg_id"].to_numpy()):
    raise RuntimeError("ECG order mismatch between cache and prediction table.")

Y = np.column_stack([saved[f"true__{c}"].to_numpy(np.int8) for c in MODEL_CLASSES])

model = tf.keras.models.load_model(BASE_MODEL_PATH, compile=False)
rep_model = tf.keras.Model(model.input, model.get_layer("add_3").output)

# 1) Activation-derived lead profile
print("Extracting add_3 activations ...")
A = rep_model.predict(X, batch_size=PRED_BATCH_SIZE, verbose=1)
# A shape: N x 12 x 55 x 128
rms = np.sqrt(np.mean(np.square(A), axis=(2,3)) + 1e-8)
profile = rms / (rms.sum(axis=1, keepdims=True) + 1e-8)
del A

# 2) Base model probabilities/logits
print("Base inference ...")
P0 = model.predict(X, batch_size=PRED_BATCH_SIZE, verbose=1).astype(np.float64)
P0 = np.clip(P0, EPS, 1-EPS)
L0 = np.log(P0 / (1-P0))

# 3) Lead-zeroing functional influence
N = len(X)
influence_all = np.zeros((N,12), dtype=np.float64)
influence_positive = np.full((N,12), np.nan, dtype=np.float64)

for lead in range(12):
    print(f"Occlusion lead {lead+1}/12")
    X_occ = X.copy()
    # Inputs are z-normalised; zero is the within-lead mean level.
    X_occ[:,lead,:,0] = 0.0
    P = model.predict(X_occ, batch_size=PRED_BATCH_SIZE, verbose=0).astype(np.float64)
    P = np.clip(P, EPS, 1-EPS)
    L = np.log(P / (1-P))
    delta = np.abs(L - L0)
    influence_all[:,lead] = delta.mean(axis=1)
    for i in range(N):
        pos = Y[i] == 1
        if pos.any():
            influence_positive[i,lead] = delta[i,pos].mean()
    del X_occ, P, L, delta

# normalise influence for distributional comparison
influence_all_norm = influence_all / (influence_all.sum(axis=1, keepdims=True) + 1e-12)
pos_sums = np.nansum(influence_positive, axis=1, keepdims=True)
influence_positive_norm = influence_positive / np.where(pos_sums > 0, pos_sums, np.nan)

rows=[]
for i in range(N):
    rho_all, p_all = spearmanr(profile[i], influence_all_norm[i])
    if np.all(np.isfinite(influence_positive_norm[i])):
        rho_pos, p_pos = spearmanr(profile[i], influence_positive_norm[i])
    else:
        rho_pos, p_pos = np.nan, np.nan
    rows.append({
        "ecg_id": int(ecg_ids[i]),
        "n_positive_labels": int(Y[i].sum()),
        "spearman_activation_vs_all_output_occlusion": float(rho_all),
        "p_all_output": float(p_all),
        "spearman_activation_vs_positive_label_occlusion": float(rho_pos) if np.isfinite(rho_pos) else np.nan,
        "p_positive_label": float(p_pos) if np.isfinite(p_pos) else np.nan,
        "top1_match_all_output": int(np.argmax(profile[i]) == np.argmax(influence_all_norm[i])),
        "top1_match_positive_label": int(np.argmax(profile[i]) == np.nanargmax(influence_positive_norm[i])) if np.all(np.isfinite(influence_positive_norm[i])) else np.nan,
    })

per_ecg = pd.DataFrame(rows)
per_ecg.to_csv(PER_ECG_PATH, index=False)
np.savez_compressed(
    INFLUENCE_PATH,
    activation_profile=profile,
    occlusion_influence_all=influence_all_norm,
    occlusion_influence_positive=influence_positive_norm,
    ecg_ids=ecg_ids,
)

summary_rows=[]
for col in [
    "spearman_activation_vs_all_output_occlusion",
    "spearman_activation_vs_positive_label_occlusion",
]:
    vals=per_ecg[col].dropna().to_numpy(float)
    summary_rows.append({
        "measure": col,
        "N": len(vals),
        "mean": float(np.mean(vals)),
        "SD": float(np.std(vals,ddof=1)),
        "median": float(np.median(vals)),
        "Q1": float(np.percentile(vals,25)),
        "Q3": float(np.percentile(vals,75)),
        "fraction_positive": float(np.mean(vals>0)),
    })
summary_rows += [
    {
        "measure": "top1_match_all_output",
        "N": int(per_ecg["top1_match_all_output"].notna().sum()),
        "mean": float(per_ecg["top1_match_all_output"].mean()),
        "SD": np.nan, "median": np.nan, "Q1": np.nan, "Q3": np.nan,
        "fraction_positive": np.nan,
    },
    {
        "measure": "top1_match_positive_label",
        "N": int(per_ecg["top1_match_positive_label"].notna().sum()),
        "mean": float(per_ecg["top1_match_positive_label"].mean()),
        "SD": np.nan, "median": np.nan, "Q1": np.nan, "Q3": np.nan,
        "fraction_positive": np.nan,
    },
]
summary=pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH,index=False)

print("\n" + "="*110)
print("OPTIONAL FUNCTIONAL SANITY CHECK — ACTIVATION PROFILE vs LEAD OCCLUSION")
print("="*110)
print(summary.round(4).to_string(index=False))
print("\nPer-ECG results:", PER_ECG_PATH)
print("Summary:", SUMMARY_PATH)
print("\nInterpret cautiously: correlation with lead occlusion is supportive construct evidence, not proof that the profile is an attention mechanism or a complete causal explanation.")
