# ==========================================================================================
# EHJ-DH REVIEWER ANALYSIS
# COMBINED DIAGNOSTIC + GAZE LOSS EXPERIMENT
#
# PURPOSE
# -------
# Test whether retaining diagnostic supervision changes the interpretation of the original
# gaze-only experiment.
#
# TRAINING DATA
# -------------
# Diagnostic objective:
#   PTB-XL stratified folds 1-9 only
#   19,601 ECGs
#   same 23 diagnostic subclasses used by the pretrained model
#
# Gaze objective:
#   same frozen 10 reconstructed gaze ECGs
#   same consultant / all-clinician 12-lead gaze targets
#   same 40 Gaussian-noise augmentations per ECG, SD 0.03
#
# TEST DATA
# ---------
# PTB-XL fold 10 ONLY
#   2,198 ECGs
#   NEVER used for training, lambda selection, hyperparameter selection, or early stopping
#
# ARMS
# ----
# 1. base                         : unchanged pretrained model
# 2. diagnostic_only             : BCE only
# 3. joint_consultant            : BCE + lambda * consultant-gaze MSE
# 4. joint_all_clinicians        : BCE + lambda * all-clinician-gaze MSE
# 5. joint_uniform               : BCE + lambda * uniform-target MSE
# 6. joint_perm_consultant       : BCE + lambda * mismatched consultant-gaze MSE
# 7. joint_perm_all_clinicians   : BCE + lambda * mismatched all-clinician-gaze MSE
#
# LAMBDA
# ------
# Frozen before training:
#
# lambda =
#     initial mean diagnostic BCE on PTB-XL folds 1-9
#     ------------------------------------------------
#     mean(initial consultant-gaze MSE,
#          initial all-clinician-gaze MSE)
#
# The SAME lambda is used for all joint/control arms.
#
# TRAINABLE PARAMETERS
# --------------------
# All layers from conv2d_10 onward may adapt EXCEPT BatchNormalization layers.
# BatchNorm parameters/statistics remain frozen.
# All forward passes use training=False.
#
# OPTIMISATION
# ------------
# Adam, LR 1e-5
# 4 PTB-XL epochs
# diagnostic batch size = 128
# gaze batch size = 16
#
# One diagnostic batch and one gaze batch are combined in each joint optimisation step.
# Four PTB epochs correspond to ~25 passes through the 400 augmented gaze samples.
#
# IMPORTANT
# ---------
# Do not change lambda, learning rate, epoch number, permutation, or arm definitions after
# seeing the results.
#
# The script is restart-safe. It writes a frozen manifest before training and saves each arm.
# ==========================================================================================


# ==========================================================================================
# 0. IMPORTS
# ==========================================================================================

import ast
import gc
import json
import math
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb

from sklearn.metrics import roc_auc_score, average_precision_score


# ==========================================================================================
# 1. FROZEN PARAMETERS
# ==========================================================================================

SEED = 2026

MODEL_CLASSES = [
    "AMI",
    "CLBBB",
    "CRBBB",
    "ILBBB",
    "IMI",
    "IRBBB",
    "ISCA",
    "ISCI",
    "ISC_",
    "IVCD",
    "LAFB/LPFB",
    "LAO/LAE",
    "LMI",
    "LVH",
    "NORM",
    "NST_",
    "PMI",
    "RAO/RAE",
    "RVH",
    "SEHYP",
    "STTC",
    "WPW",
    "_AVB",
]

EXPECTED_LEADS = [
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

ECG_ORDER = [
    "NSR",
    "AFib",
    "Hyperkalemia",
    "AFlutter",
    "VTach",
    "WPW",
    "VentPaced",
    "LBBB",
    "STEMI",
    "CompleteHeartBlock",
]

# Primary fixed derangement used in the previous matched-control analysis.
FIXED_PERMUTATION = {
    "NSR": "CompleteHeartBlock",
    "AFib": "STEMI",
    "Hyperkalemia": "AFib",
    "AFlutter": "NSR",
    "VTach": "VentPaced",
    "WPW": "VTach",
    "VentPaced": "WPW",
    "LBBB": "Hyperkalemia",
    "STEMI": "LBBB",
    "CompleteHeartBlock": "AFlutter",
}

# Gaze augmentation: unchanged from original experiment
AUGMENTATIONS_PER_ECG = 40
NOISE_SD = 0.03

# Joint-training definition
DIAG_BATCH_SIZE = 128
GAZE_BATCH_SIZE = 16
JOINT_EPOCHS = 4

# More parameters are trainable than in the original single-layer experiment,
# so use a conservative LR one order below the original 1e-4.
LEARNING_RATE = 1e-5

RMS_EPSILON = 1e-8

# Evaluation
PRED_BATCH_SIZE = 64
N_BOOTSTRAPS = 3000
BOOTSTRAP_SEED = 2026

EXPECTED_TRAIN_N = 19601
EXPECTED_TEST_N = 2198
EXPECTED_TEST_PATIENTS = 1904


# ==========================================================================================
# 2. REPRODUCIBILITY
# ==========================================================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("TensorFlow deterministic operations enabled.")
except Exception as e:
    print("Could not explicitly enable deterministic operations:", repr(e))


# ==========================================================================================
# 3. PATHS
# ==========================================================================================

ROOT = Path("definitive_clean_gaze")

STEP1_MANIFEST_PATH = ROOT / "step1_manifest.json"
TARGET_PATH = ROOT / "clean_gaze_targets.npz"
GAZE_X_PATH = Path("final_digitisation_v9/X_gaze_final.npy")

PTBXL_ROOT = Path("ptb-xl")
DATABASE_PATH = PTBXL_ROOT / "ptbxl_database.csv"
SCP_PATH = PTBXL_ROOT / "scp_statements.csv"

OUT_DIR = ROOT / "reviewer_controls" / "combined_loss_experiment"
CACHE_DIR = OUT_DIR / "cache"
MODEL_DIR = OUT_DIR / "models"
PRED_DIR = OUT_DIR / "predictions"
HISTORY_DIR = OUT_DIR / "history"

for d in [OUT_DIR, CACHE_DIR, MODEL_DIR, PRED_DIR, HISTORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUT_DIR / "combined_loss_manifest.json"
POINT_ESTIMATES_PATH = OUT_DIR / "combined_loss_point_estimates.csv"
BOOTSTRAP_PATH = OUT_DIR / "combined_loss_bootstrap_comparisons.csv"
GAZE_FIT_PATH = OUT_DIR / "combined_loss_training_gaze_fit.csv"

TRAIN_X_CACHE = CACHE_DIR / "ptbxl_folds1to9_X.npy"
TEST_X_CACHE = CACHE_DIR / "ptbxl_fold10_X.npy"

TRAIN_Y_CACHE = CACHE_DIR / "ptbxl_folds1to9_Y.npy"
TEST_Y_CACHE = CACHE_DIR / "ptbxl_fold10_Y.npy"

TRAIN_IDS_CACHE = CACHE_DIR / "ptbxl_folds1to9_ecg_ids.npy"
TEST_IDS_CACHE = CACHE_DIR / "ptbxl_fold10_ecg_ids.npy"
TEST_PATIENT_CACHE = CACHE_DIR / "ptbxl_fold10_patient_ids.npy"


for p in [
    STEP1_MANIFEST_PATH,
    TARGET_PATH,
    GAZE_X_PATH,
    DATABASE_PATH,
    SCP_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ==========================================================================================
# 4. HELPERS
# ==========================================================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def attention_distribution_from_add3(add3_tensor):
    rms = tf.sqrt(
        tf.reduce_mean(
            tf.square(add3_tensor),
            axis=[2, 3]
        )
        + RMS_EPSILON
    )

    return rms / (
        tf.reduce_sum(rms, axis=1, keepdims=True)
        + RMS_EPSILON
    )


def cosine_rows(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    num = np.sum(a * b, axis=1)
    den = (
        np.linalg.norm(a, axis=1)
        *
        np.linalg.norm(b, axis=1)
    )

    return num / np.maximum(den, eps)


def macro_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    aucs = []
    aps = []

    for c in range(y_true.shape[1]):
        yc = y_true[:, c]
        pc = y_prob[:, c]

        # Match manuscript bootstrap convention:
        # each class must have positives and negatives.
        if len(np.unique(yc)) < 2:
            return np.nan, np.nan

        aucs.append(
            roc_auc_score(yc, pc)
        )

        aps.append(
            average_precision_score(yc, pc)
        )

    return float(np.mean(aucs)), float(np.mean(aps))


def micro_metrics(y_true, y_prob):
    return (
        float(
            roc_auc_score(
                y_true.ravel(),
                y_prob.ravel()
            )
        ),
        float(
            average_precision_score(
                y_true.ravel(),
                y_prob.ravel()
            )
        ),
    )


# ==========================================================================================
# 5. LOAD AND VERIFY FROZEN BASE MODEL
# ==========================================================================================

with open(STEP1_MANIFEST_PATH, "r") as f:
    step1_manifest = json.load(f)

BASE_MODEL_PATH = Path(
    step1_manifest["base_model"]
)

if not BASE_MODEL_PATH.exists():
    raise FileNotFoundError(BASE_MODEL_PATH)

print("\n" + "=" * 110)
print("FROZEN INPUT VERIFICATION")
print("=" * 110)

print("Base model:", BASE_MODEL_PATH)
print("Base SHA256:", sha256_file(BASE_MODEL_PATH))
print("Gaze X SHA256:", sha256_file(GAZE_X_PATH))
print("Gaze target SHA256:", sha256_file(TARGET_PATH))

if sha256_file(BASE_MODEL_PATH) != step1_manifest["base_model_sha256"]:
    raise RuntimeError("Base model hash mismatch.")

if sha256_file(GAZE_X_PATH) != step1_manifest["frozen_ecg_sha256"]:
    raise RuntimeError("Frozen gaze ECG hash mismatch.")

if sha256_file(TARGET_PATH) != step1_manifest["clean_gaze_targets_sha256"]:
    raise RuntimeError("Frozen gaze-target hash mismatch.")

print("✓ Frozen base model, gaze inputs, and gaze targets verified.")


# ==========================================================================================
# 6. LOAD GAZE INPUTS AND TARGETS
# ==========================================================================================

X_gaze = np.load(
    GAZE_X_PATH
).astype(np.float32)

target_data = np.load(
    TARGET_PATH
)

Y_gaze = {
    "consultant":
        target_data["consultant"].astype(np.float32),

    "all_clinicians":
        target_data["all_clinicians"].astype(np.float32),
}

if X_gaze.shape != (10, 12, 1000, 1):
    raise RuntimeError(
        f"Unexpected gaze ECG shape: {X_gaze.shape}"
    )

for population in ["consultant", "all_clinicians"]:
    if Y_gaze[population].shape != (10, 12):
        raise RuntimeError(
            f"Unexpected {population} target shape: "
            f"{Y_gaze[population].shape}"
        )


# Uniform target
Y_uniform = np.full(
    (10, 12),
    1.0 / 12.0,
    dtype=np.float32
)


# Primary fixed permuted targets
ecg_to_index = {
    name: i
    for i, name in enumerate(ECG_ORDER)
}

permuted_indices = np.array(
    [
        ecg_to_index[
            FIXED_PERMUTATION[name]
        ]
        for name in ECG_ORDER
    ],
    dtype=int
)

Y_perm_consultant = (
    Y_gaze["consultant"][permuted_indices]
    .copy()
)

Y_perm_all = (
    Y_gaze["all_clinicians"][permuted_indices]
    .copy()
)

ARM_GAZE_TARGETS = {
    "joint_consultant":
        Y_gaze["consultant"],

    "joint_all_clinicians":
        Y_gaze["all_clinicians"],

    "joint_uniform":
        Y_uniform,

    "joint_perm_consultant":
        Y_perm_consultant,

    "joint_perm_all_clinicians":
        Y_perm_all,
}


# ==========================================================================================
# 7. BUILD EXACT 23-CLASS PTB-XL LABELS
# ==========================================================================================

metadata = pd.read_csv(
    DATABASE_PATH,
    index_col="ecg_id"
)

scp = pd.read_csv(
    SCP_PATH,
    index_col=0
)

metadata["scp_codes"] = (
    metadata["scp_codes"]
    .apply(ast.literal_eval)
)

diagnostic_scp = scp[
    scp["diagnostic"] == 1
].copy()

class_to_index = {
    class_name: i
    for i, class_name in enumerate(MODEL_CLASSES)
}


def aggregate_subclasses(code_dictionary):
    output = set()

    for code in code_dictionary.keys():

        if code not in diagnostic_scp.index:
            continue

        subclass = diagnostic_scp.loc[
            code,
            "diagnostic_subclass"
        ]

        if pd.isna(subclass):
            continue

        output.add(str(subclass))

    return output


def make_label_matrix(df):
    labels = (
        df["scp_codes"]
        .apply(aggregate_subclasses)
    )

    Y = np.zeros(
        (len(df), len(MODEL_CLASSES)),
        dtype=np.float32
    )

    for row_idx, row_labels in enumerate(labels):
        for label in row_labels:

            if label not in class_to_index:
                raise RuntimeError(
                    f"Unexpected diagnostic subclass: {label}"
                )

            Y[
                row_idx,
                class_to_index[label]
            ] = 1.0

    return Y


train_metadata = metadata[
    metadata["strat_fold"] != 10
].copy()

test_metadata = metadata[
    metadata["strat_fold"] == 10
].copy()

if len(train_metadata) != EXPECTED_TRAIN_N:
    raise RuntimeError(
        f"Expected {EXPECTED_TRAIN_N} folds-1-9 ECGs, "
        f"found {len(train_metadata)}"
    )

if len(test_metadata) != EXPECTED_TEST_N:
    raise RuntimeError(
        f"Expected {EXPECTED_TEST_N} fold-10 ECGs, "
        f"found {len(test_metadata)}"
    )

if test_metadata["patient_id"].nunique() != EXPECTED_TEST_PATIENTS:
    raise RuntimeError(
        "Unexpected number of unique fold-10 patients."
    )

train_patients = set(
    train_metadata["patient_id"].dropna()
)

test_patients = set(
    test_metadata["patient_id"].dropna()
)

if train_patients & test_patients:
    raise RuntimeError(
        "Patient overlap detected between folds 1-9 and fold 10."
    )

Y_train = make_label_matrix(
    train_metadata
)

Y_test = make_label_matrix(
    test_metadata
)

np.save(
    TRAIN_Y_CACHE,
    Y_train
)

np.save(
    TEST_Y_CACHE,
    Y_test
)

np.save(
    TRAIN_IDS_CACHE,
    train_metadata.index.to_numpy()
)

np.save(
    TEST_IDS_CACHE,
    test_metadata.index.to_numpy()
)

np.save(
    TEST_PATIENT_CACHE,
    test_metadata["patient_id"].to_numpy()
)

print("\n" + "=" * 110)
print("PTB-XL LABEL AUDIT")
print("=" * 110)

print("Folds 1-9:", len(train_metadata))
print("Fold 10:", len(test_metadata))
print("Fold-10 patients:", test_metadata["patient_id"].nunique())
print(
    "All-zero folds-1-9 rows:",
    int((Y_train.sum(axis=1) == 0).sum())
)
print(
    "All-zero fold-10 rows:",
    int((Y_test.sum(axis=1) == 0).sum())
)


# ==========================================================================================
# 8. BUILD / REUSE PTB-XL WAVEFORM CACHES
#
# Uses exactly the same 100-Hz filename_lr signals as the original fold-10 evaluation:
# raw shape      = (1000, 12)
# model shape    = (12, 1000, 1)
#
# No additional normalisation is applied to PTB-XL.
# ==========================================================================================

def build_waveform_cache(df, path):

    if path.exists():

        arr = np.load(
            path,
            mmap_mode="r"
        )

        expected = (
            len(df),
            12,
            1000,
            1
        )

        if arr.shape != expected:
            raise RuntimeError(
                f"Existing cache {path} has shape "
                f"{arr.shape}, expected {expected}"
            )

        print(
            f"Reusing waveform cache: {path}"
        )

        return arr

    print(
        f"\nBuilding waveform cache: {path}"
    )

    X_mem = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(df),
            12,
            1000,
            1
        )
    )

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        record_path = (
            PTBXL_ROOT
            /
            str(row["filename_lr"])
        )

        signal, fields = wfdb.rdsamp(
            str(record_path)
        )

        if signal.shape != (1000, 12):
            raise RuntimeError(
                f"Bad signal shape for row {i}: "
                f"{signal.shape}"
            )

        leads = [
            str(x).upper()
            for x in fields["sig_name"]
        ]

        if leads != EXPECTED_LEADS:
            raise RuntimeError(
                f"Lead-order mismatch at row {i}: "
                f"{leads}"
            )

        if float(fields["fs"]) != 100.0:
            raise RuntimeError(
                f"Sampling-rate mismatch at row {i}"
            )

        if not np.all(
            np.isfinite(signal)
        ):
            raise RuntimeError(
                f"Non-finite signal at row {i}"
            )

        X_mem[
            i,
            :,
            :,
            0
        ] = signal.T.astype(
            np.float32
        )

        if (
            (i + 1) % 500 == 0
            or
            (i + 1) == len(df)
        ):
            print(
                f"Loaded {i + 1}/{len(df)}"
            )

    X_mem.flush()

    del X_mem

    return np.load(
        path,
        mmap_mode="r"
    )


X_train = build_waveform_cache(
    train_metadata,
    TRAIN_X_CACHE
)

X_test = build_waveform_cache(
    test_metadata,
    TEST_X_CACHE
)

print(
    "\nTrain waveform shape:",
    X_train.shape
)

print(
    "Test waveform shape:",
    X_test.shape
)


# ==========================================================================================
# 9. BUILD FROZEN GAZE AUGMENTATION ONCE
#
# Identical augmented inputs are reused by every joint/control arm.
# ==========================================================================================

GAZE_AUG_X_PATH = CACHE_DIR / "gaze_augmented_X.npy"
GAZE_AUG_INDEX_PATH = CACHE_DIR / "gaze_augmented_source_index.npy"

if (
    GAZE_AUG_X_PATH.exists()
    and
    GAZE_AUG_INDEX_PATH.exists()
):

    X_gaze_aug = np.load(
        GAZE_AUG_X_PATH
    )

    gaze_source_index = np.load(
        GAZE_AUG_INDEX_PATH
    )

else:

    rng_aug = np.random.default_rng(
        SEED + 100
    )

    X_list = []
    index_list = []

    for ecg_idx in range(10):

        for _ in range(
            AUGMENTATIONS_PER_ECG
        ):

            noise = rng_aug.normal(
                loc=0.0,
                scale=NOISE_SD,
                size=X_gaze[ecg_idx].shape
            ).astype(np.float32)

            X_list.append(
                (
                    X_gaze[ecg_idx]
                    +
                    noise
                ).astype(np.float32)
            )

            index_list.append(
                ecg_idx
            )

    X_gaze_aug = np.stack(
        X_list
    ).astype(np.float32)

    gaze_source_index = np.asarray(
        index_list,
        dtype=np.int64
    )

    np.save(
        GAZE_AUG_X_PATH,
        X_gaze_aug
    )

    np.save(
        GAZE_AUG_INDEX_PATH,
        gaze_source_index
    )


if X_gaze_aug.shape != (
    400,
    12,
    1000,
    1
):
    raise RuntimeError(
        f"Unexpected augmented gaze shape: "
        f"{X_gaze_aug.shape}"
    )

print(
    "\nFrozen gaze augmentation:",
    X_gaze_aug.shape
)


# ==========================================================================================
# 10. FRESH MODEL WITH TRAINABLE TAIL
# ==========================================================================================

def load_fresh_tail_model():

    tf.keras.backend.clear_session()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )

    layer_names = [
        layer.name
        for layer in model.layers
    ]

    if "conv2d_10" not in layer_names:
        raise RuntimeError(
            "conv2d_10 not found."
        )

    conv10_index = layer_names.index(
        "conv2d_10"
    )

    # Freeze everything first.
    for layer in model.layers:
        layer.trainable = False

    # Allow conv2d_10 onward to adapt,
    # EXCEPT BatchNormalization.
    for layer in model.layers[
        conv10_index:
    ]:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False
        else:
            layer.trainable = True

    add3 = model.get_layer(
        "add_3"
    )

    joint_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.output,
            add3.output
        ]
    )

    train_vars = []

    for layer in model.layers:
        train_vars.extend(
            layer.trainable_variables
        )

    if len(train_vars) == 0:
        raise RuntimeError(
            "No trainable variables."
        )

    trainable_layers = [
        layer.name
        for layer in model.layers
        if len(layer.trainable_variables) > 0
    ]

    return (
        model,
        joint_model,
        train_vars,
        trainable_layers
    )


# Print trainable tail once.
_tmp_model, _tmp_joint, _tmp_vars, trainable_layer_names = (
    load_fresh_tail_model()
)

print("\n" + "=" * 110)
print("TRAINABLE TAIL")
print("=" * 110)

for name in trainable_layer_names:
    print(name)

print(
    "Trainable parameter tensors:",
    len(_tmp_vars)
)

del _tmp_model, _tmp_joint, _tmp_vars
gc.collect()


# ==========================================================================================
# 11. LOSS FUNCTIONS
# ==========================================================================================

bce_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    reduction=tf.keras.losses.Reduction.NONE
)


def diagnostic_bce(y_true, y_prob):

    # BinaryCrossentropy with reduction NONE returns
    # one value per ECG after averaging across 23 outputs.
    per_example = bce_object(
        y_true,
        y_prob
    )

    return tf.reduce_mean(
        per_example
    )


def gaze_mse_from_add3(
    add3,
    target
):

    profile = (
        attention_distribution_from_add3(
            add3
        )
    )

    return tf.reduce_mean(
        tf.square(
            profile
            -
            target
        )
    )


# ==========================================================================================
# 12. INITIAL DIAGNOSTIC LOSS
#
# Calculated across ALL folds 1-9, without training.
# ==========================================================================================

print("\n" + "=" * 110)
print("CALCULATING INITIAL LOSS SCALES")
print("=" * 110)

base_model = tf.keras.models.load_model(
    BASE_MODEL_PATH,
    compile=False
)

base_add3_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer(
        "add_3"
    ).output
)

diag_losses = []

for start in range(
    0,
    EXPECTED_TRAIN_N,
    PRED_BATCH_SIZE
):

    stop = min(
        start + PRED_BATCH_SIZE,
        EXPECTED_TRAIN_N
    )

    xb = tf.convert_to_tensor(
        np.asarray(
            X_train[start:stop]
        ),
        dtype=tf.float32
    )

    yb = tf.convert_to_tensor(
        Y_train[start:stop],
        dtype=tf.float32
    )

    pred = base_model(
        xb,
        training=False
    )

    loss = diagnostic_bce(
        yb,
        pred
    )

    diag_losses.append(
        (
            stop - start,
            float(loss.numpy())
        )
    )

INITIAL_DIAG_BCE = (
    sum(
        n * loss
        for n, loss in diag_losses
    )
    /
    sum(
        n
        for n, loss in diag_losses
    )
)


# ==========================================================================================
# 13. INITIAL REAL-GAZE LOSSES
# ==========================================================================================

base_add3 = base_add3_model(
    tf.convert_to_tensor(
        X_gaze,
        dtype=tf.float32
    ),
    training=False
)

base_profile = (
    attention_distribution_from_add3(
        base_add3
    ).numpy()
)

INITIAL_GAZE_MSE = {}

for population in [
    "consultant",
    "all_clinicians",
]:

    INITIAL_GAZE_MSE[
        population
    ] = float(
        np.mean(
            np.square(
                base_profile
                -
                Y_gaze[
                    population
                ]
            )
        )
    )


MEAN_INITIAL_REAL_GAZE_MSE = float(
    np.mean(
        [
            INITIAL_GAZE_MSE[
                "consultant"
            ],
            INITIAL_GAZE_MSE[
                "all_clinicians"
            ],
        ]
    )
)


LAMBDA_CALCULATED = float(
    INITIAL_DIAG_BCE
    /
    MEAN_INITIAL_REAL_GAZE_MSE
)


print(
    f"Initial diagnostic BCE: "
    f"{INITIAL_DIAG_BCE:.8f}"
)

print(
    f"Initial consultant gaze MSE: "
    f"{INITIAL_GAZE_MSE['consultant']:.8f}"
)

print(
    f"Initial all-clinician gaze MSE: "
    f"{INITIAL_GAZE_MSE['all_clinicians']:.8f}"
)

print(
    f"Mean initial real-gaze MSE: "
    f"{MEAN_INITIAL_REAL_GAZE_MSE:.8f}"
)

print(
    f"Calculated lambda: "
    f"{LAMBDA_CALCULATED:.8f}"
)


# ==========================================================================================
# 14. FREEZE MANIFEST BEFORE TRAINING
# ==========================================================================================

manifest_core = {
    "analysis":
        "combined diagnostic plus gaze loss reviewer sensitivity",

    "seed":
        SEED,

    "base_model":
        str(BASE_MODEL_PATH),

    "base_model_sha256":
        sha256_file(BASE_MODEL_PATH),

    "gaze_input":
        str(GAZE_X_PATH),

    "gaze_input_sha256":
        sha256_file(GAZE_X_PATH),

    "gaze_target":
        str(TARGET_PATH),

    "gaze_target_sha256":
        sha256_file(TARGET_PATH),

    "diagnostic_training_folds":
        "PTB-XL strat_fold 1-9",

    "diagnostic_test_fold":
        10,

    "n_diagnostic_train":
        EXPECTED_TRAIN_N,

    "n_diagnostic_test":
        EXPECTED_TEST_N,

    "trainable_from_layer":
        "conv2d_10",

    "batchnorm":
        "frozen; all forward passes training=False",

    "learning_rate":
        LEARNING_RATE,

    "joint_epochs":
        JOINT_EPOCHS,

    "diagnostic_batch_size":
        DIAG_BATCH_SIZE,

    "gaze_batch_size":
        GAZE_BATCH_SIZE,

    "gaze_augmentations_per_ecg":
        AUGMENTATIONS_PER_ECG,

    "gaze_noise_sd":
        NOISE_SD,

    "lambda_definition":
        "initial folds1-9 BCE divided by mean initial consultant/all-clinician gaze MSE",

    "initial_diagnostic_bce":
        INITIAL_DIAG_BCE,

    "initial_consultant_gaze_mse":
        INITIAL_GAZE_MSE[
            "consultant"
        ],

    "initial_all_clinician_gaze_mse":
        INITIAL_GAZE_MSE[
            "all_clinicians"
        ],

    "lambda":
        LAMBDA_CALCULATED,

    "fixed_permutation":
        FIXED_PERMUTATION,

    "trainable_layers":
        trainable_layer_names,

    "arms": [
        "diagnostic_only",
        "joint_consultant",
        "joint_all_clinicians",
        "joint_uniform",
        "joint_perm_consultant",
        "joint_perm_all_clinicians",
    ],
}


if MANIFEST_PATH.exists():

    with open(
        MANIFEST_PATH,
        "r"
    ) as f:
        frozen_manifest = json.load(f)

    # Do not silently overwrite an already-frozen design.
    required_checks = [
        "base_model_sha256",
        "gaze_input_sha256",
        "gaze_target_sha256",
        "learning_rate",
        "joint_epochs",
        "diagnostic_batch_size",
        "gaze_batch_size",
        "lambda",
    ]

    for key in required_checks:

        if key not in frozen_manifest:
            raise RuntimeError(
                f"Frozen manifest missing {key}"
            )

    if (
        frozen_manifest[
            "base_model_sha256"
        ]
        !=
        manifest_core[
            "base_model_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen combined-loss manifest "
            "does not match current base model."
        )

    # IMPORTANT: use the originally frozen lambda.
    LAMBDA = float(
        frozen_manifest[
            "lambda"
        ]
    )

    print(
        "\nReusing PRE-EXISTING frozen "
        "combined-loss manifest."
    )

else:

    with open(
        MANIFEST_PATH,
        "w"
    ) as f:
        json.dump(
            manifest_core,
            f,
            indent=2
        )

    LAMBDA = LAMBDA_CALCULATED

    print(
        "\nCombined-loss design frozen BEFORE training:"
    )

    print(
        MANIFEST_PATH
    )

    print(
        "Manifest SHA256:",
        sha256_file(
            MANIFEST_PATH
        )
    )


print(
    f"\nFROZEN lambda = {LAMBDA:.8f}"
)

print(
    "Do not change lambda or the training definition "
    "after viewing results."
)


del base_model, base_add3_model, base_add3
gc.collect()


# ==========================================================================================
# 15. PREPARE AUGMENTED TARGET ARRAYS
# ==========================================================================================

def augmented_targets(
    target_10x12
):
    return target_10x12[
        gaze_source_index
    ].astype(np.float32)


ARM_GAZE_TARGETS_AUG = {
    arm:
        augmented_targets(target)

    for arm, target
    in ARM_GAZE_TARGETS.items()
}


# ==========================================================================================
# 16. DETERMINISTIC EPOCH ORDERS
#
# Same diagnostic and gaze sample order is reused in every arm.
# ==========================================================================================

N_DIAG_BATCHES = int(
    math.ceil(
        EXPECTED_TRAIN_N
        /
        DIAG_BATCH_SIZE
    )
)

N_GAZE_SAMPLES_REQUIRED_PER_EPOCH = (
    N_DIAG_BATCHES
    *
    GAZE_BATCH_SIZE
)

print(
    "\nDiagnostic batches per epoch:",
    N_DIAG_BATCHES
)

print(
    "Gaze examples consumed per PTB epoch:",
    N_GAZE_SAMPLES_REQUIRED_PER_EPOCH
)

print(
    "Equivalent passes through 400 augmented gaze examples "
    "over all epochs:",
    (
        N_GAZE_SAMPLES_REQUIRED_PER_EPOCH
        *
        JOINT_EPOCHS
        /
        len(X_gaze_aug)
    )
)


def make_epoch_orders():

    diagnostic_orders = []
    gaze_orders = []

    for epoch in range(
        JOINT_EPOCHS
    ):

        rng_diag = np.random.default_rng(
            SEED + 1000 + epoch
        )

        diagnostic_orders.append(
            rng_diag.permutation(
                EXPECTED_TRAIN_N
            )
        )

        rng_gaze = np.random.default_rng(
            SEED + 2000 + epoch
        )

        pieces = []
        collected = 0

        while (
            collected
            <
            N_GAZE_SAMPLES_REQUIRED_PER_EPOCH
        ):

            p = rng_gaze.permutation(
                len(X_gaze_aug)
            )

            pieces.append(p)

            collected += len(p)

        g = np.concatenate(
            pieces
        )[
            :N_GAZE_SAMPLES_REQUIRED_PER_EPOCH
        ]

        gaze_orders.append(g)

    return (
        diagnostic_orders,
        gaze_orders
    )


DIAG_ORDERS, GAZE_ORDERS = (
    make_epoch_orders()
)


# ==========================================================================================
# 17. WEIGHT AUDIT HELPERS
# ==========================================================================================

def snapshot_weights(model):

    return {
        layer.name: [
            np.array(
                w,
                copy=True
            )
            for w in layer.get_weights()
        ]
        for layer in model.layers
    }


def changed_layers(
    before,
    model,
    tol=1e-12
):

    changed = []

    for layer in model.layers:

        old = before[
            layer.name
        ]

        new = layer.get_weights()

        max_change = 0.0

        for a, b in zip(
            old,
            new
        ):

            if a.size:
                max_change = max(
                    max_change,
                    float(
                        np.max(
                            np.abs(
                                a - b
                            )
                        )
                    )
                )

        if max_change > tol:

            changed.append(
                (
                    layer.name,
                    max_change
                )
            )

    return changed


# ==========================================================================================
# 18. TRAIN ONE ARM
# ==========================================================================================

TRAINING_ARMS = [
    "diagnostic_only",
    "joint_consultant",
    "joint_all_clinicians",
    "joint_uniform",
    "joint_perm_consultant",
    "joint_perm_all_clinicians",
]


def train_arm(arm_name):

    print("\n" + "=" * 110)
    print("TRAINING:", arm_name)
    print("=" * 110)

    weights_path = (
        MODEL_DIR
        /
        f"{arm_name}.weights.h5"
    )

    history_path = (
        HISTORY_DIR
        /
        f"{arm_name}_history.csv"
    )

    pred_path = (
        PRED_DIR
        /
        f"{arm_name}_fold10_predictions.npy"
    )

    (
        model,
        joint_model,
        train_vars,
        trainable_layers,
    ) = load_fresh_tail_model()

    # ----------------------------------------------------------
    # Resume completed arm if weights already exist.
    # ----------------------------------------------------------

    if weights_path.exists():

        print(
            "Existing completed weights found. "
            "Loading rather than retraining."
        )

        model.load_weights(
            weights_path
        )

        return (
            model,
            joint_model,
            weights_path,
            history_path,
            pred_path,
        )

    before = snapshot_weights(
        model
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    history_rows = []

    if arm_name == "diagnostic_only":
        gaze_targets_aug = None
    else:
        gaze_targets_aug = (
            ARM_GAZE_TARGETS_AUG[
                arm_name
            ]
        )

    for epoch in range(
        JOINT_EPOCHS
    ):

        diag_order = DIAG_ORDERS[
            epoch
        ]

        gaze_order = GAZE_ORDERS[
            epoch
        ]

        epoch_diag_losses = []
        epoch_gaze_losses = []
        epoch_total_losses = []

        for batch_number in range(
            N_DIAG_BATCHES
        ):

            diag_start = (
                batch_number
                *
                DIAG_BATCH_SIZE
            )

            diag_stop = min(
                diag_start
                +
                DIAG_BATCH_SIZE,
                EXPECTED_TRAIN_N
            )

            diag_idx = diag_order[
                diag_start:diag_stop
            ]

            xb_diag = tf.convert_to_tensor(
                np.asarray(
                    X_train[
                        diag_idx
                    ]
                ),
                dtype=tf.float32
            )

            yb_diag = tf.convert_to_tensor(
                Y_train[
                    diag_idx
                ],
                dtype=tf.float32
            )

            if (
                arm_name
                !=
                "diagnostic_only"
            ):

                gaze_start = (
                    batch_number
                    *
                    GAZE_BATCH_SIZE
                )

                gaze_stop = (
                    gaze_start
                    +
                    GAZE_BATCH_SIZE
                )

                gaze_idx = gaze_order[
                    gaze_start:gaze_stop
                ]

                xb_gaze = tf.convert_to_tensor(
                    X_gaze_aug[
                        gaze_idx
                    ],
                    dtype=tf.float32
                )

                yb_gaze = tf.convert_to_tensor(
                    gaze_targets_aug[
                        gaze_idx
                    ],
                    dtype=tf.float32
                )

            with tf.GradientTape() as tape:

                # Diagnostic objective
                pred_diag, _ = joint_model(
                    xb_diag,
                    training=False
                )

                diag_loss = diagnostic_bce(
                    yb_diag,
                    pred_diag
                )

                if (
                    arm_name
                    ==
                    "diagnostic_only"
                ):

                    gaze_loss = tf.constant(
                        0.0,
                        dtype=tf.float32
                    )

                    total_loss = diag_loss

                else:

                    # Gaze objective uses independent gaze-labelled ECG batch.
                    _, add3_gaze = joint_model(
                        xb_gaze,
                        training=False
                    )

                    gaze_loss = gaze_mse_from_add3(
                        add3_gaze,
                        yb_gaze
                    )

                    total_loss = (
                        diag_loss
                        +
                        (
                            tf.cast(
                                LAMBDA,
                                tf.float32
                            )
                            *
                            gaze_loss
                        )
                    )

            grads = tape.gradient(
                total_loss,
                train_vars
            )

            if any(
                g is None
                for g in grads
            ):
                bad = [
                    i
                    for i, g in enumerate(grads)
                    if g is None
                ]

                raise RuntimeError(
                    f"None gradients in "
                    f"{arm_name}: {bad}"
                )

            for g in grads:
                if not np.all(
                    np.isfinite(
                        g.numpy()
                    )
                ):
                    raise RuntimeError(
                        f"Non-finite gradient "
                        f"in {arm_name}"
                    )

            optimizer.apply_gradients(
                zip(
                    grads,
                    train_vars
                )
            )

            epoch_diag_losses.append(
                float(
                    diag_loss.numpy()
                )
            )

            epoch_gaze_losses.append(
                float(
                    gaze_loss.numpy()
                )
            )

            epoch_total_losses.append(
                float(
                    total_loss.numpy()
                )
            )

        row = {
            "arm":
                arm_name,

            "epoch":
                epoch + 1,

            "diagnostic_BCE":
                float(
                    np.mean(
                        epoch_diag_losses
                    )
                ),

            "gaze_MSE":
                float(
                    np.mean(
                        epoch_gaze_losses
                    )
                ),

            "total_loss":
                float(
                    np.mean(
                        epoch_total_losses
                    )
                ),
        }

        history_rows.append(row)

        print(
            f"Epoch {epoch + 1}/{JOINT_EPOCHS} | "
            f"BCE={row['diagnostic_BCE']:.6f} | "
            f"gaze MSE={row['gaze_MSE']:.8f} | "
            f"total={row['total_loss']:.6f}"
        )

    # ----------------------------------------------------------
    # Weight audit
    # ----------------------------------------------------------

    changed = changed_layers(
        before,
        model
    )

    changed_names = [
        name
        for name, _
        in changed
    ]

    allowed = set(
        trainable_layers
    )

    unexpected = [
        x
        for x in changed_names
        if x not in allowed
    ]

    if unexpected:
        raise RuntimeError(
            f"Unexpected changed layers: "
            f"{unexpected}"
        )

    print(
        "\nChanged layers:"
    )

    for name, delta in changed:
        print(
            f"  {name:<30s} "
            f"max|Δ|={delta:.8g}"
        )

    pd.DataFrame(
        history_rows
    ).to_csv(
        history_path,
        index=False
    )

    model.save_weights(
        weights_path
    )

    print(
        "Saved:",
        weights_path
    )

    return (
        model,
        joint_model,
        weights_path,
        history_path,
        pred_path,
    )


# ==========================================================================================
# 19. FOLD-10 INFERENCE
# ==========================================================================================

def predict_fold10(
    model,
    pred_path
):

    if pred_path.exists():

        pred = np.load(
            pred_path
        )

        if pred.shape != (
            EXPECTED_TEST_N,
            23
        ):
            raise RuntimeError(
                f"Bad saved prediction shape: "
                f"{pred.shape}"
            )

        return pred

    chunks = []

    for start in range(
        0,
        EXPECTED_TEST_N,
        PRED_BATCH_SIZE
    ):

        stop = min(
            start
            +
            PRED_BATCH_SIZE,
            EXPECTED_TEST_N
        )

        xb = tf.convert_to_tensor(
            np.asarray(
                X_test[
                    start:stop
                ]
            ),
            dtype=tf.float32
        )

        pred = model(
            xb,
            training=False
        ).numpy()

        chunks.append(
            pred
        )

    pred = np.concatenate(
        chunks,
        axis=0
    )

    np.save(
        pred_path,
        pred
    )

    return pred


# ==========================================================================================
# 20. BASE PREDICTIONS
# ==========================================================================================

BASE_PRED_PATH = (
    PRED_DIR
    /
    "base_fold10_predictions.npy"
)

if BASE_PRED_PATH.exists():

    PREDICTIONS = {
        "base":
            np.load(
                BASE_PRED_PATH
            )
    }

else:

    base = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )

    PREDICTIONS = {
        "base":
            predict_fold10(
                base,
                BASE_PRED_PATH
            )
    }

    del base
    gc.collect()


# ==========================================================================================
# 21. TRAIN / RESUME ALL ARMS
# ==========================================================================================

for arm in TRAINING_ARMS:

    (
        model,
        joint_model,
        weights_path,
        history_path,
        pred_path,
    ) = train_arm(
        arm
    )

    PREDICTIONS[
        arm
    ] = predict_fold10(
        model,
        pred_path
    )

    del model, joint_model
    gc.collect()

    print(
        f"Completed {arm}"
    )


# ==========================================================================================
# 22. POINT ESTIMATES
# ==========================================================================================

metric_rows = []

for arm in [
    "base"
] + TRAINING_ARMS:

    macro_auc, macro_ap = (
        macro_metrics(
            Y_test,
            PREDICTIONS[
                arm
            ]
        )
    )

    micro_auc, micro_ap = (
        micro_metrics(
            Y_test,
            PREDICTIONS[
                arm
            ]
        )
    )

    metric_rows.append(
        {
            "model":
                arm,

            "macro_AUROC":
                macro_auc,

            "macro_AP":
                macro_ap,

            "micro_AUROC":
                micro_auc,

            "micro_AP":
                micro_ap,
        }
    )

point_df = pd.DataFrame(
    metric_rows
)

point_df.to_csv(
    POINT_ESTIMATES_PATH,
    index=False
)


# ==========================================================================================
# 23. TRAINING-CASE GAZE FIT
#
# This is explicitly a TRAINING-SET target-engagement check.
# It is NOT evidence of gaze generalisation.
# ==========================================================================================

gaze_fit_rows = []

GAZE_EVAL_DEFINITIONS = [
    (
        "base_vs_consultant",
        "base",
        Y_gaze["consultant"],
    ),
    (
        "base_vs_all_clinicians",
        "base",
        Y_gaze["all_clinicians"],
    ),
    (
        "diagnostic_only_vs_consultant",
        "diagnostic_only",
        Y_gaze["consultant"],
    ),
    (
        "diagnostic_only_vs_all_clinicians",
        "diagnostic_only",
        Y_gaze["all_clinicians"],
    ),
    (
        "joint_consultant_vs_consultant",
        "joint_consultant",
        Y_gaze["consultant"],
    ),
    (
        "joint_all_clinicians_vs_all_clinicians",
        "joint_all_clinicians",
        Y_gaze["all_clinicians"],
    ),
    (
        "joint_uniform_vs_uniform",
        "joint_uniform",
        Y_uniform,
    ),
    (
        "joint_perm_consultant_vs_permuted_target",
        "joint_perm_consultant",
        Y_perm_consultant,
    ),
    (
        "joint_perm_all_vs_permuted_target",
        "joint_perm_all_clinicians",
        Y_perm_all,
    ),
]


def load_arm_model_for_eval(
    arm
):

    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )

    if arm != "base":

        weight_path = (
            MODEL_DIR
            /
            f"{arm}.weights.h5"
        )

        model.load_weights(
            weight_path
        )

    rep = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(
            "add_3"
        ).output
    )

    return model, rep


for (
    label,
    arm,
    target,
) in GAZE_EVAL_DEFINITIONS:

    model_eval, rep_eval = (
        load_arm_model_for_eval(
            arm
        )
    )

    add3 = rep_eval(
        tf.convert_to_tensor(
            X_gaze,
            dtype=tf.float32
        ),
        training=False
    )

    profile = (
        attention_distribution_from_add3(
            add3
        ).numpy()
    )

    mse = float(
        np.mean(
            np.square(
                profile
                -
                target
            )
        )
    )

    cos = cosine_rows(
        profile,
        target
    )

    gaze_fit_rows.append(
        {
            "comparison":
                label,

            "model":
                arm,

            "mean_cosine":
                float(
                    np.mean(cos)
                ),

            "mean_MSE":
                mse,

            "median_cosine":
                float(
                    np.median(cos)
                ),
        }
    )

    del model_eval, rep_eval
    gc.collect()


gaze_fit_df = pd.DataFrame(
    gaze_fit_rows
)

gaze_fit_df.to_csv(
    GAZE_FIT_PATH,
    index=False
)


# ==========================================================================================
# 24. PATIENT-CLUSTER BOOTSTRAP
# ==========================================================================================

patient_ids = (
    test_metadata[
        "patient_id"
    ].to_numpy()
)

unique_patients = np.unique(
    patient_ids
)

patient_to_rows = {
    patient:
        np.where(
            patient_ids
            ==
            patient
        )[0]

    for patient
    in unique_patients
}


BOOTSTRAP_COMPARISONS = [
    (
        "diagnostic_only_minus_base",
        "diagnostic_only",
        "base",
    ),

    (
        "joint_consultant_minus_diagnostic_only",
        "joint_consultant",
        "diagnostic_only",
    ),

    (
        "joint_all_clinicians_minus_diagnostic_only",
        "joint_all_clinicians",
        "diagnostic_only",
    ),

    (
        "joint_uniform_minus_diagnostic_only",
        "joint_uniform",
        "diagnostic_only",
    ),

    (
        "joint_perm_consultant_minus_diagnostic_only",
        "joint_perm_consultant",
        "diagnostic_only",
    ),

    (
        "joint_perm_all_minus_diagnostic_only",
        "joint_perm_all_clinicians",
        "diagnostic_only",
    ),

    (
        "joint_consultant_minus_joint_perm_consultant",
        "joint_consultant",
        "joint_perm_consultant",
    ),

    (
        "joint_all_minus_joint_perm_all",
        "joint_all_clinicians",
        "joint_perm_all_clinicians",
    ),
]


# Point estimates for differences
point_lookup = (
    point_df
    .set_index("model")
    .to_dict("index")
)

bootstrap_values = {
    comparison: {
        "macro_AUROC": [],
        "macro_AP": [],
    }

    for comparison, _, _
    in BOOTSTRAP_COMPARISONS
}


rng_boot = np.random.default_rng(
    BOOTSTRAP_SEED
)

valid_bootstraps = 0

print("\n" + "=" * 110)
print("PATIENT-CLUSTER BOOTSTRAP")
print("=" * 110)

for b in range(
    N_BOOTSTRAPS
):

    sampled_patients = rng_boot.choice(
        unique_patients,
        size=len(unique_patients),
        replace=True
    )

    sampled_rows = np.concatenate(
        [
            patient_to_rows[p]
            for p in sampled_patients
        ]
    )

    yb = Y_test[
        sampled_rows
    ]

    # Same validity convention as previous manuscript analysis:
    # all 23 classes must contain positives and negatives.
    positives = yb.sum(
        axis=0
    )

    if np.any(
        positives == 0
    ):
        continue

    if np.any(
        positives == len(yb)
    ):
        continue

    valid_bootstraps += 1

    bootstrap_arm_metrics = {}

    needed_arms = set()

    for _, a, c in BOOTSTRAP_COMPARISONS:
        needed_arms.add(a)
        needed_arms.add(c)

    for arm in needed_arms:

        auc, ap = macro_metrics(
            yb,
            PREDICTIONS[
                arm
            ][
                sampled_rows
            ]
        )

        bootstrap_arm_metrics[
            arm
        ] = {
            "macro_AUROC":
                auc,

            "macro_AP":
                ap,
        }

    for (
        comparison,
        arm_a,
        arm_b,
    ) in BOOTSTRAP_COMPARISONS:

        for metric in [
            "macro_AUROC",
            "macro_AP",
        ]:

            delta = (
                bootstrap_arm_metrics[
                    arm_a
                ][
                    metric
                ]
                -
                bootstrap_arm_metrics[
                    arm_b
                ][
                    metric
                ]
            )

            bootstrap_values[
                comparison
            ][
                metric
            ].append(
                delta
            )

    if (
        (b + 1) % 250 == 0
        or
        (b + 1) == N_BOOTSTRAPS
    ):

        print(
            f"Bootstrap draws: {b + 1}/{N_BOOTSTRAPS} | "
            f"valid: {valid_bootstraps}"
        )


bootstrap_rows = []

for (
    comparison,
    arm_a,
    arm_b,
) in BOOTSTRAP_COMPARISONS:

    for metric in [
        "macro_AUROC",
        "macro_AP",
    ]:

        values = np.asarray(
            bootstrap_values[
                comparison
            ][
                metric
            ],
            dtype=np.float64
        )

        point_difference = (
            point_lookup[
                arm_a
            ][
                metric
            ]
            -
            point_lookup[
                arm_b
            ][
                metric
            ]
        )

        bootstrap_rows.append(
            {
                "comparison":
                    comparison,

                "arm_A":
                    arm_a,

                "arm_B":
                    arm_b,

                "metric":
                    metric,

                "difference_A_minus_B":
                    point_difference,

                "CI_low":
                    float(
                        np.percentile(
                            values,
                            2.5
                        )
                    ),

                "CI_high":
                    float(
                        np.percentile(
                            values,
                            97.5
                        )
                    ),

                "valid_bootstraps":
                    len(values),
            }
        )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_df.to_csv(
    BOOTSTRAP_PATH,
    index=False
)


# ==========================================================================================
# 25. FINAL REPORT
# ==========================================================================================

print("\n\n")
print("=" * 120)
print("MANUSCRIPT JOINT-LOSS RESULTS")
print("=" * 120)

print("\nFROZEN DESIGN")
print("-" * 120)

print(
    f"PTB-XL diagnostic training ECGs: "
    f"{EXPECTED_TRAIN_N}"
)

print(
    f"PTB-XL fold-10 test ECGs: "
    f"{EXPECTED_TEST_N}"
)

print(
    f"Initial diagnostic BCE: "
    f"{INITIAL_DIAG_BCE:.8f}"
)

print(
    f"Initial consultant gaze MSE: "
    f"{INITIAL_GAZE_MSE['consultant']:.8f}"
)

print(
    f"Initial all-clinician gaze MSE: "
    f"{INITIAL_GAZE_MSE['all_clinicians']:.8f}"
)

print(
    f"Frozen lambda: "
    f"{LAMBDA:.8f}"
)

print(
    f"Learning rate: "
    f"{LEARNING_RATE}"
)

print(
    f"Epochs: "
    f"{JOINT_EPOCHS}"
)

print(
    f"Diagnostic batch size: "
    f"{DIAG_BATCH_SIZE}"
)

print(
    f"Gaze batch size: "
    f"{GAZE_BATCH_SIZE}"
)

print(
    f"Equivalent gaze passes: "
    f"{N_GAZE_SAMPLES_REQUIRED_PER_EPOCH * JOINT_EPOCHS / len(X_gaze_aug):.2f}"
)

print(
    f"Valid patient-cluster bootstrap replicates: "
    f"{valid_bootstraps}/{N_BOOTSTRAPS}"
)


print("\nPOINT ESTIMATES")
print("-" * 120)

print(
    point_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print("\nTRAINING-CASE GAZE FIT")
print("-" * 120)

print(
    gaze_fit_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print("\nPAIRED PATIENT-CLUSTER BOOTSTRAP COMPARISONS")
print("-" * 120)

print(
    bootstrap_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print("\nOUTPUT FILES")
print("-" * 120)

print(
    "Manifest:",
    MANIFEST_PATH
)

print(
    "Manifest SHA256:",
    sha256_file(
        MANIFEST_PATH
    )
)

print(
    "Point estimates:",
    POINT_ESTIMATES_PATH
)

print(
    "Training-case gaze fit:",
    GAZE_FIT_PATH
)

print(
    "Bootstrap comparisons:",
    BOOTSTRAP_PATH
)

print(
    "Model weights:",
    MODEL_DIR
)

print(
    "Predictions:",
    PRED_DIR
)

print(
    "Training histories:",
    HISTORY_DIR
)


print("\n" + "=" * 120)
print("IMPORTANT")
print("=" * 120)

print(
    "Do not change lambda, learning rate, epoch number, "
    "the fixed permutation, or selectively rerun arms after inspecting these results."
)

print(
    "Paste everything from 'MANUSCRIPT JOINT-LOSS RESULTS' onward back into ChatGPT."
)