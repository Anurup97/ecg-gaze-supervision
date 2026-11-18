# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: extra analysis.ipynb ; code cell 1
# Stage: original single-permutation matched controls / training-fold mean-gaze baseline
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.
# NOTE: As captured by researcher; verify appropriate output paths; no repeated-permutation analysis in this source.

# ==========================================================================================
# GAZE-ALIGNED ECG MODEL — REVIEWER-REQUESTED CONTROL REANALYSIS
#
# PURPOSE
# -------
# Addresses three specific methodological concerns:
#
#   1. UNIFORM-TARGET CONTROL
#      Does diagnostic performance deteriorate simply because conv2d_10 is
#      optimised for 25 epochs against ANY auxiliary attention objective?
#
#   2. PERMUTED-GAZE CONTROL
#      Does true ECG-specific gaze content matter, or does any structured
#      non-uniform target have the same effect?
#
#   3. TRAINING-FOLD MEAN-GAZE BASELINE
#      Does held-out alignment exceed what would be obtained simply by using
#      the average gaze distribution of the training ECGs?
#
# ALSO:
#   - case-specific direction-of-change analysis
#   - target concentration / entropy analysis
#   - PTB-XL fold-10 diagnostic evaluation of control models
#   - patient-cluster bootstrap
#   - LUDB frozen-representation control evaluation
#
# IMPORTANT
# ---------
# This code DOES NOT overwrite:
#   definitive_clean_gaze/step2_cv/
#   definitive_clean_gaze/final_models/
#   definitive_clean_gaze/ptbxl_evaluation/
#   definitive_clean_gaze/ludb_evaluation/
#
# All new results are written to:
#
#   definitive_clean_gaze/control_reanalysis/
#
# ==========================================================================================


# ==========================================================================================
# 0. IMPORTS
# ==========================================================================================

import ast
import gc
import json
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb

from scipy.signal import resample_poly
from scipy.stats import wilcoxon, spearmanr

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# ==========================================================================================
# 1. FIXED SETTINGS — MATCH ORIGINAL EXPERIMENT
# ==========================================================================================

SEED = 2026

N_SPLITS = 5

AUGMENTATIONS_PER_ECG = 40
NOISE_SD = 0.03

BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
RMS_EPSILON = 1e-8

PRED_BATCH_SIZE = 64

# Repeated permutations are used for the HELD-OUT alignment sensitivity analysis.
# The downstream diagnostic control uses one pre-specified deterministic derangement,
# matching the reviewer's requested single permuted-gaze experiment.
N_PERMUTED_CV_REPEATS = 10

# Bootstrap count matches your previous PTB-XL / LUDB analyses.
N_BOOT = 3000

RUN_PTBXL_CONTROLS = True
RUN_PTBXL_BOOTSTRAP = True

RUN_LUDB_CONTROLS = True
RUN_LUDB_BOOTSTRAP = True


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


LEAD_ORDER = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


POPULATIONS = [
    "consultant",
    "all_clinicians",
]


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


EXPECTED_LEADS_UPPER = [
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

X_PATH = Path(
    "final_digitisation_v9/X_gaze_final.npy"
)

GAZE_TARGET_PATH = (
    ROOT
    /
    "clean_gaze_targets.npz"
)

STEP1_MANIFEST_PATH = (
    ROOT
    /
    "step1_manifest.json"
)

STEP2_DIR = (
    ROOT
    /
    "step2_cv"
)

STEP2_METRICS_PATH = (
    STEP2_DIR
    /
    "heldout_metrics.csv"
)

STEP2_ATTENTION_PATH = (
    STEP2_DIR
    /
    "heldout_attention_distributions.csv"
)

STEP2_MANIFEST_PATH = (
    STEP2_DIR
    /
    "step2_manifest.json"
)

FINAL_MODEL_DIR = (
    ROOT
    /
    "final_models"
)

FINAL_MODEL_MANIFEST_PATH = (
    FINAL_MODEL_DIR
    /
    "final_model_manifest.json"
)

ORIGINAL_CONSULTANT_WEIGHTS = (
    FINAL_MODEL_DIR
    /
    "clean_consultant.weights.h5"
)

ORIGINAL_ALL_WEIGHTS = (
    FINAL_MODEL_DIR
    /
    "clean_all_clinicians.weights.h5"
)


CONTROL_DIR = (
    ROOT
    /
    "control_reanalysis"
)

CONTROL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CONTROL_CV_DIR = (
    CONTROL_DIR
    /
    "alignment_controls"
)

CONTROL_CV_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CONTROL_MODEL_DIR = (
    CONTROL_DIR
    /
    "control_models"
)

CONTROL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CONTROL_PTBXL_DIR = (
    CONTROL_DIR
    /
    "ptbxl_controls"
)

CONTROL_PTBXL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CONTROL_LUDB_DIR = (
    CONTROL_DIR
    /
    "ludb_controls"
)

CONTROL_LUDB_DIR.mkdir(
    parents=True,
    exist_ok=True
)


required_paths = [
    X_PATH,
    GAZE_TARGET_PATH,
    STEP1_MANIFEST_PATH,
    STEP2_METRICS_PATH,
    STEP2_ATTENTION_PATH,
    STEP2_MANIFEST_PATH,
    FINAL_MODEL_MANIFEST_PATH,
    ORIGINAL_CONSULTANT_WEIGHTS,
    ORIGINAL_ALL_WEIGHTS,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ==========================================================================================
# 4. HASH HELPER
# ==========================================================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


# ==========================================================================================
# 5. VERIFY FROZEN INPUTS
# ==========================================================================================

with open(
    STEP1_MANIFEST_PATH,
    "r"
) as f:

    step1_manifest = json.load(
        f
    )


with open(
    STEP2_MANIFEST_PATH,
    "r"
) as f:

    step2_manifest = json.load(
        f
    )


with open(
    FINAL_MODEL_MANIFEST_PATH,
    "r"
) as f:

    final_model_manifest = json.load(
        f
    )


BASE_MODEL_PATH = Path(
    step1_manifest[
        "base_model"
    ]
)


if not BASE_MODEL_PATH.exists():

    raise FileNotFoundError(
        BASE_MODEL_PATH
    )


X_HASH = sha256_file(
    X_PATH
)

GAZE_HASH = sha256_file(
    GAZE_TARGET_PATH
)

BASE_HASH = sha256_file(
    BASE_MODEL_PATH
)

ORIGINAL_CONSULTANT_HASH = sha256_file(
    ORIGINAL_CONSULTANT_WEIGHTS
)

ORIGINAL_ALL_HASH = sha256_file(
    ORIGINAL_ALL_WEIGHTS
)


print("=" * 110)
print("FROZEN INPUT / MODEL VERIFICATION")
print("=" * 110)

print("V9 gaze ECG:")
print(X_HASH)

print("\nGaze targets:")
print(GAZE_HASH)

print("\nBase model:")
print(BASE_HASH)

print("\nOriginal consultant model:")
print(ORIGINAL_CONSULTANT_HASH)

print("\nOriginal all-clinician model:")
print(ORIGINAL_ALL_HASH)


if X_HASH != step1_manifest["frozen_ecg_sha256"]:
    raise RuntimeError("Frozen V9 ECG hash mismatch.")

if GAZE_HASH != step1_manifest["clean_gaze_targets_sha256"]:
    raise RuntimeError("Frozen gaze-target hash mismatch.")

if BASE_HASH != step1_manifest["base_model_sha256"]:
    raise RuntimeError("Frozen base-model hash mismatch.")

if (
    ORIGINAL_CONSULTANT_HASH
    !=
    final_model_manifest[
        "consultant_weights_sha256"
    ]
):
    raise RuntimeError("Original consultant-model hash mismatch.")

if (
    ORIGINAL_ALL_HASH
    !=
    final_model_manifest[
        "all_clinician_weights_sha256"
    ]
):
    raise RuntimeError("Original all-clinician-model hash mismatch.")


print("\n✓ Original experiment is frozen and verified.")


# ==========================================================================================
# 6. VERIFY ORIGINAL TRAINING DEFINITION
# ==========================================================================================

required_step2 = {

    "attention_layer":
        "add_3",

    "trainable_layer":
        "conv2d_10",

    "epochs":
        25,

    "augmentations_per_training_ECG":
        40,

    "gaussian_noise_sigma":
        0.03,

    "batch_size":
        16,
}


for key, expected in required_step2.items():

    actual = step2_manifest[
        key
    ]

    if actual != expected:

        raise RuntimeError(
            f"Original Step-2 setting changed for {key}: "
            f"{actual} != {expected}"
        )


print(
    "✓ Control experiments will use the exact original training regime."
)


# ==========================================================================================
# 7. LOAD FROZEN GAZE DATA
# ==========================================================================================

X = np.load(
    X_PATH
).astype(
    np.float32
)


target_data = np.load(
    GAZE_TARGET_PATH
)


Y = {

    "consultant":
        target_data[
            "consultant"
        ].astype(
            np.float32
        ),

    "all_clinicians":
        target_data[
            "all_clinicians"
        ].astype(
            np.float32
        ),
}


saved_ecg_order = [
    str(x)
    for x
    in target_data[
        "ecg_order"
    ]
]


saved_lead_order = [
    str(x)
    for x
    in target_data[
        "lead_order"
    ]
]


if X.shape != (
    10,
    12,
    1000,
    1
):

    raise RuntimeError(
        f"Unexpected gaze ECG shape: {X.shape}"
    )


if saved_ecg_order != ECG_ORDER:

    raise RuntimeError(
        "ECG order mismatch."
    )


if saved_lead_order != LEAD_ORDER:

    raise RuntimeError(
        "Lead order mismatch."
    )


for population in POPULATIONS:

    if Y[
        population
    ].shape != (
        10,
        12
    ):

        raise RuntimeError(
            f"Bad target shape for {population}."
        )

    if not np.allclose(
        Y[
            population
        ].sum(
            axis=1
        ),
        1.0,
        atol=1e-5
    ):

        raise RuntimeError(
            f"{population}: gaze rows do not sum to 1."
        )


print(
    "✓ Frozen gaze arrays loaded."
)


# ==========================================================================================
# 8. ORIGINAL RESULTS
# ==========================================================================================

original_metrics = pd.read_csv(
    STEP2_METRICS_PATH
)


original_attention_long = pd.read_csv(
    STEP2_ATTENTION_PATH
)


if len(
    original_metrics
) != 20:

    raise RuntimeError(
        "Expected 20 original held-out result rows."
    )


# ==========================================================================================
# 9. FROZEN ATTENTION READOUT + METRICS
# ==========================================================================================

def attention_distribution_from_add3(
    add3_tensor
):

    rms = tf.sqrt(

        tf.reduce_mean(
            tf.square(
                add3_tensor
            ),
            axis=[
                2,
                3
            ]
        )

        +

        RMS_EPSILON
    )


    denominator = tf.reduce_sum(
        rms,
        axis=1,
        keepdims=True
    )


    return (
        rms
        /
        (
            denominator
            +
            RMS_EPSILON
        )
    )


def cosine_metric(
    p,
    q
):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )

    denom = (
        np.linalg.norm(p)
        *
        np.linalg.norm(q)
    )

    if denom <= 0:
        return np.nan

    return float(
        np.dot(
            p,
            q
        )
        /
        denom
    )


def centered_cosine_metric(
    p,
    q
):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )

    uniform = np.ones(
        12,
        dtype=np.float64
    ) / 12.0

    p = p - uniform
    q = q - uniform

    denom = (
        np.linalg.norm(p)
        *
        np.linalg.norm(q)
    )

    if denom <= 1e-15:
        return np.nan

    return float(
        np.dot(
            p,
            q
        )
        /
        denom
    )


def mse_metric(
    p,
    q
):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )

    return float(
        np.mean(
            np.square(
                p - q
            )
        )
    )


def jsd_metric(
    p,
    q,
    eps=1e-12
):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )

    p = np.clip(
        p,
        eps,
        None
    )

    q = np.clip(
        q,
        eps,
        None
    )

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (
        p + q
    )

    kl_pm = np.sum(
        p
        *
        np.log(
            p / m
        )
    )

    kl_qm = np.sum(
        q
        *
        np.log(
            q / m
        )
    )

    return float(
        0.5
        *
        (
            kl_pm
            +
            kl_qm
        )
    )


def tv_metric(
    p,
    q
):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )

    return float(
        0.5
        *
        np.sum(
            np.abs(
                p - q
            )
        )
    )


def metric_bundle(
    prediction,
    target
):

    return {

        "cosine":
            cosine_metric(
                prediction,
                target
            ),

        "centered_cosine":
            centered_cosine_metric(
                prediction,
                target
            ),

        "MSE":
            mse_metric(
                prediction,
                target
            ),

        "JSD":
            jsd_metric(
                prediction,
                target
            ),

        "TV":
            tv_metric(
                prediction,
                target
            ),
    }


def safe_wilcoxon(
    a,
    b
):

    a = np.asarray(
        a,
        dtype=np.float64
    )

    b = np.asarray(
        b,
        dtype=np.float64
    )

    valid = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    a = a[
        valid
    ]

    b = b[
        valid
    ]

    if len(a) == 0:
        return np.nan, np.nan

    if np.allclose(
        a,
        b
    ):
        return 0.0, 1.0

    result = wilcoxon(
        a,
        b,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        )
    )


# ==========================================================================================
# 10. FRESH MODEL / WEIGHT-AUDIT HELPERS
# ==========================================================================================

def load_fresh_trainable_model():

    tf.keras.backend.clear_session()

    tf.random.set_seed(
        SEED
    )

    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )

    for layer in model.layers:
        layer.trainable = False

    conv10 = model.get_layer(
        "conv2d_10"
    )

    conv10.trainable = True

    rep_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(
            "add_3"
        ).output
    )

    train_vars = list(
        conv10.trainable_variables
    )

    if len(train_vars) == 0:

        raise RuntimeError(
            "conv2d_10 has no trainable variables."
        )

    return (
        model,
        rep_model,
        train_vars
    )


def predict_attention(
    rep_model,
    x
):

    a = attention_distribution_from_add3(
        rep_model(
            x,
            training=False
        )
    ).numpy()

    if not np.all(
        np.isfinite(
            a
        )
    ):

        raise RuntimeError(
            "Non-finite attention."
        )

    if not np.allclose(
        a.sum(
            axis=1
        ),
        1.0,
        atol=1e-5
    ):

        raise RuntimeError(
            "Attention rows do not sum to 1."
        )

    return a


def snapshot_weights(
    model
):

    return {

        layer.name: [
            np.array(
                w,
                copy=True
            )
            for w
            in layer.get_weights()
        ]

        for layer
        in model.layers
    }


def weight_change_audit(
    before,
    model
):

    rows = []

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

            if a.shape != b.shape:

                raise RuntimeError(
                    f"Weight-shape mismatch: {layer.name}"
                )

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

        rows.append(
            {
                "layer":
                    layer.name,

                "max_abs_change":
                    max_change,
            }
        )

    audit = pd.DataFrame(
        rows
    )

    changed = set(
        audit.loc[
            audit[
                "max_abs_change"
            ]
            >
            1e-12,
            "layer"
        ]
    )

    if changed != {
        "conv2d_10"
    }:

        raise RuntimeError(
            f"Unexpected changed layers: {changed}"
        )

    return audit


# ==========================================================================================
# 11. GENERIC CONTROL-TRAINING FUNCTION
#
# Y_original_targets is one 12-vector per ORIGINAL ECG in X_train ordering.
# ==========================================================================================

def train_control_model(
    X_train_augmented,
    target_matrix_for_original_ECGs,
    epoch_permutations,
    save_path=None,
    verbose_label=""
):

    (
        model,
        rep_model,
        train_vars
    ) = load_fresh_trainable_model()

    before = snapshot_weights(
        model
    )

    target_matrix_for_original_ECGs = np.asarray(
        target_matrix_for_original_ECGs,
        dtype=np.float32
    )

    Y_augmented = np.repeat(
        target_matrix_for_original_ECGs,
        AUGMENTATIONS_PER_ECG,
        axis=0
    ).astype(
        np.float32
    )

    if len(
        Y_augmented
    ) != len(
        X_train_augmented
    ):

        raise RuntimeError(
            "Control X/Y augmentation mismatch."
        )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    history = []

    for epoch in range(
        EPOCHS
    ):

        permutation = epoch_permutations[
            epoch
        ]

        losses = []

        for start in range(
            0,
            len(
                permutation
            ),
            BATCH_SIZE
        ):

            idx = permutation[
                start:
                start + BATCH_SIZE
            ]

            xb = tf.convert_to_tensor(
                X_train_augmented[
                    idx
                ],
                dtype=tf.float32
            )

            yb = tf.convert_to_tensor(
                Y_augmented[
                    idx
                ],
                dtype=tf.float32
            )

            with tf.GradientTape() as tape:

                add3 = rep_model(
                    xb,
                    training=False
                )

                predicted_attention = (
                    attention_distribution_from_add3(
                        add3
                    )
                )

                loss = tf.reduce_mean(
                    tf.square(
                        predicted_attention
                        -
                        yb
                    )
                )

            gradients = tape.gradient(
                loss,
                train_vars
            )

            if any(
                gradient is None
                for gradient
                in gradients
            ):

                raise RuntimeError(
                    "None gradient detected."
                )

            for gradient in gradients:

                if not np.all(
                    np.isfinite(
                        gradient.numpy()
                    )
                ):

                    raise RuntimeError(
                        "Non-finite gradient."
                    )

            optimizer.apply_gradients(
                zip(
                    gradients,
                    train_vars
                )
            )

            losses.append(
                float(
                    loss.numpy()
                )
            )

        history.append(
            float(
                np.mean(
                    losses
                )
            )
        )

    audit = weight_change_audit(
        before,
        model
    )

    if save_path is not None:

        model.save_weights(
            save_path
        )

    if verbose_label:

        print(
            f"{verbose_label}: "
            f"loss {history[0]:.8f} -> {history[-1]:.8f}"
        )

    return (
        model,
        rep_model,
        history,
        audit
    )


# ==========================================================================================
# 12. RECONSTRUCT EXACT ORIGINAL 5-FOLD SPLIT
# ==========================================================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)


folds = []

for fold_number, (
    train_indices,
    test_indices
) in enumerate(
    kf.split(
        np.arange(
            10
        )
    ),
    start=1
):

    folds.append(
        {
            "fold":
                fold_number,

            "train_indices":
                np.asarray(
                    train_indices,
                    dtype=int
                ),

            "test_indices":
                np.asarray(
                    test_indices,
                    dtype=int
                ),
        }
    )


print("\n" + "=" * 110)
print("CONTROL REANALYSIS — FROZEN FOLDS")
print("=" * 110)

for fold in folds:

    print(
        f"Fold {fold['fold']} held out:",
        [
            ECG_ORDER[i]
            for i
            in fold[
                "test_indices"
            ]
        ]
    )


# Verify against original result rows.

for fold in folds:

    expected = set(
        ECG_ORDER[i]
        for i
        in fold[
            "test_indices"
        ]
    )

    observed = set(
        original_metrics.loc[
            original_metrics[
                "fold"
            ]
            ==
            fold[
                "fold"
            ],
            "ECG"
        ].unique()
    )

    if expected != observed:

        raise RuntimeError(
            f"Fold {fold['fold']} does not reproduce original split."
        )


print(
    "✓ Exact original held-out folds reproduced."
)


# ==========================================================================================
# 13. BASE ATTENTION ON ALL TEN ECGs
# ==========================================================================================

(
    temp_base_model,
    temp_base_rep,
    _
) = load_fresh_trainable_model()


BASE_ATTENTION_ALL = predict_attention(
    temp_base_rep,
    X
)


del temp_base_rep
del temp_base_model

gc.collect()


# ==========================================================================================
# 14. HELPER: RECONSTRUCT ORIGINAL TRUE / BASE / TRAINED ATTENTION VECTORS
# ==========================================================================================

lead_position = {
    lead: i
    for i, lead
    in enumerate(
        LEAD_ORDER
    )
}


def get_original_attention_vectors(
    population,
    ecg
):

    temp = original_attention_long[
        (
            original_attention_long[
                "population"
            ]
            ==
            population
        )
        &
        (
            original_attention_long[
                "ECG"
            ]
            ==
            ecg
        )
    ].copy()

    if len(
        temp
    ) != 12:

        raise RuntimeError(
            f"{population}/{ecg}: expected 12 lead rows."
        )

    temp[
        "_order"
    ] = temp[
        "lead"
    ].map(
        lead_position
    )

    temp = temp.sort_values(
        "_order"
    )

    return (
        temp[
            "target"
        ].to_numpy(
            dtype=np.float64
        ),
        temp[
            "baseline"
        ].to_numpy(
            dtype=np.float64
        ),
        temp[
            "trained"
        ].to_numpy(
            dtype=np.float64
        ),
    )


# ==========================================================================================
# 15. EXPERIMENT 1 — TRAINING-FOLD MEAN-GAZE BASELINE
#
# NO MODEL TRAINING.
# ==========================================================================================

mean_gaze_rows = []


for fold in folds:

    fold_number = fold[
        "fold"
    ]

    train_indices = fold[
        "train_indices"
    ]

    test_indices = fold[
        "test_indices"
    ]

    for population in POPULATIONS:

        fold_mean_gaze = np.mean(
            Y[
                population
            ][
                train_indices
            ],
            axis=0
        )

        fold_mean_gaze = (
            fold_mean_gaze
            /
            fold_mean_gaze.sum()
        )

        for ecg_idx in test_indices:

            ecg = ECG_ORDER[
                ecg_idx
            ]

            true_target = Y[
                population
            ][
                ecg_idx
            ]

            base_attention = BASE_ATTENTION_ALL[
                ecg_idx
            ]

            (
                original_target,
                original_base,
                original_trained
            ) = get_original_attention_vectors(
                population,
                ecg
            )

            if not np.allclose(
                true_target,
                original_target,
                atol=1e-6
            ):

                raise RuntimeError(
                    f"{population}/{ecg}: target mismatch."
                )

            if not np.allclose(
                base_attention,
                original_base,
                atol=1e-5
            ):

                raise RuntimeError(
                    f"{population}/{ecg}: base-attention mismatch."
                )

            base_metrics = metric_bundle(
                base_attention,
                true_target
            )

            mean_metrics = metric_bundle(
                fold_mean_gaze,
                true_target
            )

            real_metrics = metric_bundle(
                original_trained,
                true_target
            )

            row = {

                "population":
                    population,

                "fold":
                    fold_number,

                "ECG":
                    ecg,
            }

            for name, values in [
                (
                    "base",
                    base_metrics
                ),
                (
                    "mean_gaze",
                    mean_metrics
                ),
                (
                    "real_gaze_model",
                    real_metrics
                ),
            ]:

                for metric_name, value in values.items():

                    row[
                        f"{name}_{metric_name}"
                    ] = value

            mean_gaze_rows.append(
                row
            )


mean_gaze_df = pd.DataFrame(
    mean_gaze_rows
)


mean_gaze_df.to_csv(
    CONTROL_CV_DIR
    /
    "mean_gaze_baseline_per_ECG.csv",
    index=False
)


# ==========================================================================================
# 16. MEAN-GAZE vs REAL GAZE MODEL SUMMARY
# ==========================================================================================

mean_gaze_summary_rows = []


for population in POPULATIONS:

    temp = mean_gaze_df[
        mean_gaze_df[
            "population"
        ]
        ==
        population
    ].copy()

    comparisons = [

        (
            "cosine",
            "higher"
        ),

        (
            "centered_cosine",
            "higher"
        ),

        (
            "MSE",
            "lower"
        ),

        (
            "JSD",
            "lower"
        ),

        (
            "TV",
            "lower"
        ),
    ]

    for metric, direction in comparisons:

        mean_values = temp[
            f"mean_gaze_{metric}"
        ].to_numpy()

        real_values = temp[
            f"real_gaze_model_{metric}"
        ].to_numpy()

        statistic, p_value = safe_wilcoxon(
            real_values,
            mean_values
        )

        if direction == "higher":

            superiority = (
                real_values
                -
                mean_values
            )

        else:

            superiority = (
                mean_values
                -
                real_values
            )

        mean_gaze_summary_rows.append(
            {

                "population":
                    population,

                "metric":
                    metric,

                "direction_better":
                    direction,

                "training_fold_mean_gaze":
                    float(
                        np.nanmean(
                            mean_values
                        )
                    ),

                "real_gaze_model":
                    float(
                        np.nanmean(
                            real_values
                        )
                    ),

                "real_model_mean_advantage":
                    float(
                        np.nanmean(
                            superiority
                        )
                    ),

                "real_model_median_advantage":
                    float(
                        np.nanmedian(
                            superiority
                        )
                    ),

                "ECGs_real_model_better":
                    int(
                        np.nansum(
                            superiority > 0
                        )
                    ),

                "wilcoxon_statistic":
                    statistic,

                "wilcoxon_p":
                    p_value,
            }
        )


mean_gaze_summary_df = pd.DataFrame(
    mean_gaze_summary_rows
)


mean_gaze_summary_df.to_csv(
    CONTROL_CV_DIR
    /
    "mean_gaze_vs_real_model_summary.csv",
    index=False
)


print("\n\n" + "=" * 110)
print("EXPERIMENT 1 — REAL MODEL vs TRAINING-FOLD MEAN-GAZE BASELINE")
print("=" * 110)

print(
    mean_gaze_summary_df
    .round(
        7
    )
    .to_string(
        index=False
    )
)


# ==========================================================================================
# 17. CASE-SPECIFIC SHIFT / GAP-CLOSURE ANALYSIS
# ==========================================================================================

def target_entropy_normalised(
    p
):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    p = np.clip(
        p,
        1e-12,
        None
    )

    p = p / p.sum()

    entropy = -np.sum(
        p
        *
        np.log(
            p
        )
    )

    return float(
        entropy
        /
        np.log(
            len(p)
        )
    )


def direction_cosine(
    actual_shift,
    required_shift
):

    actual_shift = np.asarray(
        actual_shift,
        dtype=np.float64
    )

    required_shift = np.asarray(
        required_shift,
        dtype=np.float64
    )

    denominator = (
        np.linalg.norm(
            actual_shift
        )
        *
        np.linalg.norm(
            required_shift
        )
    )

    if denominator <= 1e-15:
        return np.nan

    return float(
        np.dot(
            actual_shift,
            required_shift
        )
        /
        denominator
    )


shift_rows = []


for population in POPULATIONS:

    for ecg in ECG_ORDER:

        ecg_idx = ECG_ORDER.index(
            ecg
        )

        (
            target,
            baseline,
            trained
        ) = get_original_attention_vectors(
            population,
            ecg
        )

        required_shift = (
            target
            -
            baseline
        )

        actual_shift = (
            trained
            -
            baseline
        )

        target_max_idx = int(
            np.argmax(
                target
            )
        )

        base_mse = mse_metric(
            baseline,
            target
        )

        trained_mse = mse_metric(
            trained,
            target
        )

        base_jsd = jsd_metric(
            baseline,
            target
        )

        trained_jsd = jsd_metric(
            trained,
            target
        )

        required_at_max = float(
            required_shift[
                target_max_idx
            ]
        )

        actual_at_max = float(
            actual_shift[
                target_max_idx
            ]
        )

        if abs(
            required_at_max
        ) > 1e-12:

            fraction_required_shift_at_max = (
                actual_at_max
                /
                required_at_max
            )

        else:

            fraction_required_shift_at_max = np.nan

        row = {

            "population":
                population,

            "ECG":
                ecg,

            "target_entropy_normalised":
                target_entropy_normalised(
                    target
                ),

            "target_max_lead":
                LEAD_ORDER[
                    target_max_idx
                ],

            "target_max_weight":
                float(
                    target[
                        target_max_idx
                    ]
                ),

            "baseline_weight_at_target_max_lead":
                float(
                    baseline[
                        target_max_idx
                    ]
                ),

            "trained_weight_at_target_max_lead":
                float(
                    trained[
                        target_max_idx
                    ]
                ),

            "required_shift_at_target_max_lead":
                required_at_max,

            "actual_shift_at_target_max_lead":
                actual_at_max,

            "fraction_required_shift_achieved_at_target_max_lead":
                fraction_required_shift_at_max,

            "direction_cosine":
                direction_cosine(
                    actual_shift,
                    required_shift
                ),

            "baseline_cosine":
                cosine_metric(
                    baseline,
                    target
                ),

            "trained_cosine":
                cosine_metric(
                    trained,
                    target
                ),

            "delta_cosine":
                cosine_metric(
                    trained,
                    target
                )
                -
                cosine_metric(
                    baseline,
                    target
                ),

            "baseline_centered_cosine":
                centered_cosine_metric(
                    baseline,
                    target
                ),

            "trained_centered_cosine":
                centered_cosine_metric(
                    trained,
                    target
                ),

            "delta_centered_cosine":
                centered_cosine_metric(
                    trained,
                    target
                )
                -
                centered_cosine_metric(
                    baseline,
                    target
                ),

            "MSE_gap_closure":
                (
                    1.0
                    -
                    (
                        trained_mse
                        /
                        base_mse
                    )
                )
                if base_mse > 0
                else np.nan,

            "JSD_gap_closure":
                (
                    1.0
                    -
                    (
                        trained_jsd
                        /
                        base_jsd
                    )
                )
                if base_jsd > 0
                else np.nan,
        }

        shift_rows.append(
            row
        )


shift_df = pd.DataFrame(
    shift_rows
)


shift_df.to_csv(
    CONTROL_CV_DIR
    /
    "case_specific_shift_analysis.csv",
    index=False
)


correlation_rows = []


for population in POPULATIONS:

    temp = shift_df[
        shift_df[
            "population"
        ]
        ==
        population
    ]

    for outcome in [
        "delta_cosine",
        "delta_centered_cosine",
        "MSE_gap_closure",
        "JSD_gap_closure",
        "direction_cosine",
    ]:

        rho, p_value = spearmanr(
            temp[
                "target_entropy_normalised"
            ],
            temp[
                outcome
            ],
            nan_policy="omit"
        )

        correlation_rows.append(
            {

                "population":
                    population,

                "x":
                    "target_entropy_normalised",

                "y":
                    outcome,

                "spearman_rho":
                    float(
                        rho
                    ),

                "p":
                    float(
                        p_value
                    ),
            }
        )


entropy_correlation_df = pd.DataFrame(
    correlation_rows
)


entropy_correlation_df.to_csv(
    CONTROL_CV_DIR
    /
    "target_entropy_vs_alignment_gain.csv",
    index=False
)


print("\n\n" + "=" * 110)
print("CASE-SPECIFIC SHIFT ANALYSIS")
print("=" * 110)

print(
    shift_df[
        [
            "population",
            "ECG",
            "target_entropy_normalised",
            "target_max_lead",
            "target_max_weight",
            "baseline_weight_at_target_max_lead",
            "trained_weight_at_target_max_lead",
            "fraction_required_shift_achieved_at_target_max_lead",
            "direction_cosine",
            "delta_cosine",
            "delta_centered_cosine",
            "MSE_gap_closure",
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ==========================================================================================
# 18. DERANGEMENT HELPER
# ==========================================================================================

def deterministic_derangement(
    indices,
    seed
):

    indices = np.asarray(
        indices,
        dtype=int
    )

    rng = np.random.default_rng(
        seed
    )

    for _ in range(
        100000
    ):

        permuted = rng.permutation(
            indices
        )

        if np.all(
            permuted != indices
        ):

            return permuted

    raise RuntimeError(
        "Could not generate derangement."
    )


# ==========================================================================================
# 19. EXPERIMENT 2 + 3 — UNIFORM AND PERMUTED CONTROLS IN HELD-OUT CV
# ==========================================================================================

control_cv_rows = []
control_cv_history_rows = []
control_cv_weight_rows = []
permutation_manifest_rows = []


UNIFORM_VECTOR = np.ones(
    12,
    dtype=np.float32
) / 12.0


for fold in folds:

    fold_number = fold[
        "fold"
    ]

    train_indices = fold[
        "train_indices"
    ]

    test_indices = fold[
        "test_indices"
    ]

    print("\n\n" + "#" * 110)
    print(
        f"CONTROL CV — FOLD {fold_number}"
    )
    print("#" * 110)

    # --------------------------------------------------------------------------------------
    # Reproduce EXACT noisy training examples from original Step 2.
    # --------------------------------------------------------------------------------------

    augmentation_rng = np.random.default_rng(
        SEED
        +
        10000
        *
        fold_number
    )

    X_train_base = np.repeat(
        X[
            train_indices
        ],
        AUGMENTATIONS_PER_ECG,
        axis=0
    ).astype(
        np.float32
    )

    noise = augmentation_rng.normal(
        loc=0.0,
        scale=NOISE_SD,
        size=X_train_base.shape
    ).astype(
        np.float32
    )

    X_train_augmented = (
        X_train_base
        +
        noise
    ).astype(
        np.float32
    )

    n_training = len(
        X_train_augmented
    )

    # --------------------------------------------------------------------------------------
    # Reproduce EXACT epoch batch order from original Step 2.
    # --------------------------------------------------------------------------------------

    permutation_rng = np.random.default_rng(
        SEED
        +
        20000
        *
        fold_number
    )

    epoch_permutations = [

        permutation_rng.permutation(
            n_training
        )

        for _
        in range(
            EPOCHS
        )
    ]

    # ======================================================================================
    # 19A. UNIFORM TARGET CONTROL
    #
    # Training is population-independent because target = 1/12 for every ECG.
    # Train ONE model per fold.
    # ======================================================================================

    uniform_targets_train = np.repeat(
        UNIFORM_VECTOR[
            None,
            :
        ],
        len(
            train_indices
        ),
        axis=0
    ).astype(
        np.float32
    )

    uniform_weight_path = (
        CONTROL_CV_DIR
        /
        f"uniform_fold{fold_number}.weights.h5"
    )

    (
        uniform_model,
        uniform_rep,
        uniform_history,
        uniform_audit
    ) = train_control_model(

        X_train_augmented=
            X_train_augmented,

        target_matrix_for_original_ECGs=
            uniform_targets_train,

        epoch_permutations=
            epoch_permutations,

        save_path=
            uniform_weight_path,

        verbose_label=
            f"Uniform fold {fold_number}"
    )

    uniform_test_attention = predict_attention(
        uniform_rep,
        X[
            test_indices
        ]
    )

    for epoch_idx, loss_value in enumerate(
        uniform_history,
        start=1
    ):

        control_cv_history_rows.append(
            {

                "condition":
                    "uniform",

                "population":
                    "not_applicable",

                "repeat":
                    0,

                "fold":
                    fold_number,

                "epoch":
                    epoch_idx,

                "loss":
                    loss_value,
            }
        )

    changed_uniform = uniform_audit[
        uniform_audit[
            "max_abs_change"
        ]
        >
        1e-12
    ].copy()

    changed_uniform[
        "condition"
    ] = "uniform"

    changed_uniform[
        "population"
    ] = "not_applicable"

    changed_uniform[
        "repeat"
    ] = 0

    changed_uniform[
        "fold"
    ] = fold_number

    control_cv_weight_rows.extend(
        changed_uniform.to_dict(
            "records"
        )
    )

    for local_idx, ecg_idx in enumerate(
        test_indices
    ):

        ecg = ECG_ORDER[
            ecg_idx
        ]

        pred = uniform_test_attention[
            local_idx
        ]

        for population in POPULATIONS:

            target = Y[
                population
            ][
                ecg_idx
            ]

            metrics = metric_bundle(
                pred,
                target
            )

            control_cv_rows.append(
                {

                    "condition":
                        "uniform",

                    "population":
                        population,

                    "repeat":
                        0,

                    "fold":
                        fold_number,

                    "ECG":
                        ecg,

                    **metrics,
                }
            )

    del uniform_rep
    del uniform_model

    gc.collect()

    # ======================================================================================
    # 19B. REPEATED PERMUTED-GAZE CONTROLS
    #
    # Derangement occurs ONLY within the 8 training ECGs.
    # Held-out gaze never enters training.
    #
    # Same ECG permutation is used for consultant and all-clinician arms.
    # ======================================================================================

    for repeat in range(
        1,
        N_PERMUTED_CV_REPEATS
        +
        1
    ):

        permuted_global_indices = deterministic_derangement(
            train_indices,
            seed=(
                SEED
                +
                300000
                +
                1000
                *
                fold_number
                +
                repeat
            )
        )

        for original_idx, target_idx in zip(
            train_indices,
            permuted_global_indices
        ):

            permutation_manifest_rows.append(
                {

                    "scope":
                        "heldout_CV",

                    "repeat":
                        repeat,

                    "fold":
                        fold_number,

                    "training_ECG":
                        ECG_ORDER[
                            original_idx
                        ],

                    "assigned_gaze_target":
                        ECG_ORDER[
                            target_idx
                        ],
                }
            )

        for population in POPULATIONS:

            permuted_targets_train = Y[
                population
            ][
                permuted_global_indices
            ]

            weight_path = (
                CONTROL_CV_DIR
                /
                (
                    f"permuted_{population}_"
                    f"repeat{repeat:02d}_fold{fold_number}.weights.h5"
                )
            )

            (
                perm_model,
                perm_rep,
                perm_history,
                perm_audit
            ) = train_control_model(

                X_train_augmented=
                    X_train_augmented,

                target_matrix_for_original_ECGs=
                    permuted_targets_train,

                epoch_permutations=
                    epoch_permutations,

                save_path=
                    weight_path,

                verbose_label=(
                    f"Permuted {population} "
                    f"repeat {repeat}/{N_PERMUTED_CV_REPEATS} "
                    f"fold {fold_number}"
                )
            )

            perm_test_attention = predict_attention(
                perm_rep,
                X[
                    test_indices
                ]
            )

            for epoch_idx, loss_value in enumerate(
                perm_history,
                start=1
            ):

                control_cv_history_rows.append(
                    {

                        "condition":
                            "permuted",

                        "population":
                            population,

                        "repeat":
                            repeat,

                        "fold":
                            fold_number,

                        "epoch":
                            epoch_idx,

                        "loss":
                            loss_value,
                    }
                )

            changed_perm = perm_audit[
                perm_audit[
                    "max_abs_change"
                ]
                >
                1e-12
            ].copy()

            changed_perm[
                "condition"
            ] = "permuted"

            changed_perm[
                "population"
            ] = population

            changed_perm[
                "repeat"
            ] = repeat

            changed_perm[
                "fold"
            ] = fold_number

            control_cv_weight_rows.extend(
                changed_perm.to_dict(
                    "records"
                )
            )

            for local_idx, ecg_idx in enumerate(
                test_indices
            ):

                ecg = ECG_ORDER[
                    ecg_idx
                ]

                target = Y[
                    population
                ][
                    ecg_idx
                ]

                pred = perm_test_attention[
                    local_idx
                ]

                metrics = metric_bundle(
                    pred,
                    target
                )

                control_cv_rows.append(
                    {

                        "condition":
                            "permuted",

                        "population":
                            population,

                        "repeat":
                            repeat,

                        "fold":
                            fold_number,

                        "ECG":
                            ecg,

                        **metrics,
                    }
                )

            del perm_rep
            del perm_model

            gc.collect()


# ==========================================================================================
# 20. SAVE CONTROL-CV RAW RESULTS
# ==========================================================================================

control_cv_df = pd.DataFrame(
    control_cv_rows
)

control_cv_history_df = pd.DataFrame(
    control_cv_history_rows
)

control_cv_weight_df = pd.DataFrame(
    control_cv_weight_rows
)

permutation_manifest_df = pd.DataFrame(
    permutation_manifest_rows
)


control_cv_df.to_csv(
    CONTROL_CV_DIR
    /
    "control_heldout_alignment.csv",
    index=False
)


control_cv_history_df.to_csv(
    CONTROL_CV_DIR
    /
    "control_training_histories.csv",
    index=False
)


control_cv_weight_df.to_csv(
    CONTROL_CV_DIR
    /
    "control_weight_audit.csv",
    index=False
)


permutation_manifest_df.to_csv(
    CONTROL_CV_DIR
    /
    "permutation_manifest.csv",
    index=False
)


# ==========================================================================================
# 21. CONTROL HELD-OUT ALIGNMENT SUMMARY
# ==========================================================================================

control_summary_rows = []


for population in POPULATIONS:

    # Uniform has one result per ECG.
    uniform_temp = control_cv_df[
        (
            control_cv_df[
                "population"
            ]
            ==
            population
        )
        &
        (
            control_cv_df[
                "condition"
            ]
            ==
            "uniform"
        )
    ]

    # Permuted: summarise each repeat, then overall repeat distribution.
    perm_temp = control_cv_df[
        (
            control_cv_df[
                "population"
            ]
            ==
            population
        )
        &
        (
            control_cv_df[
                "condition"
            ]
            ==
            "permuted"
        )
    ]

    real_temp = original_metrics[
        original_metrics[
            "population"
        ]
        ==
        population
    ]

    for metric, higher_better, old_real_column in [

        (
            "cosine",
            True,
            "trained_cosine"
        ),

        (
            "MSE",
            False,
            "trained_MSE"
        ),

        (
            "JSD",
            False,
            "trained_JSD"
        ),
    ]:

        real_mean = float(
            real_temp[
                old_real_column
            ].mean()
        )

        uniform_mean = float(
            uniform_temp[
                metric
            ].mean()
        )

        repeat_means = (
            perm_temp
            .groupby(
                "repeat"
            )[
                metric
            ]
            .mean()
        )

        control_summary_rows.append(
            {

                "population":
                    population,

                "metric":
                    metric,

                "real_gaze_model_mean":
                    real_mean,

                "uniform_control_mean":
                    uniform_mean,

                "permuted_control_mean_across_repeats":
                    float(
                        repeat_means.mean()
                    ),

                "permuted_control_SD_across_repeats":
                    float(
                        repeat_means.std(
                            ddof=1
                        )
                    ),

                "permuted_control_min_repeat_mean":
                    float(
                        repeat_means.min()
                    ),

                "permuted_control_max_repeat_mean":
                    float(
                        repeat_means.max()
                    ),
            }
        )


control_alignment_summary_df = pd.DataFrame(
    control_summary_rows
)


control_alignment_summary_df.to_csv(
    CONTROL_CV_DIR
    /
    "control_alignment_summary.csv",
    index=False
)


print("\n\n" + "=" * 110)
print("EXPERIMENTS 2–3 — HELD-OUT CONTROL ALIGNMENT SUMMARY")
print("=" * 110)

print(
    control_alignment_summary_df
    .round(
        7
    )
    .to_string(
        index=False
    )
)


# ==========================================================================================
# 22. DIRECT PER-ECG REAL-vs-MEAN/PERMUTED COMPARISON
# ==========================================================================================

# Average the repeated permuted-control result at ECG level.
permuted_mean_by_ecg = (

    control_cv_df[
        control_cv_df[
            "condition"
        ]
        ==
        "permuted"
    ]

    .groupby(
        [
            "population",
            "fold",
            "ECG",
        ]
    )[
        [
            "cosine",
            "centered_cosine",
            "MSE",
            "JSD",
            "TV",
        ]
    ]

    .mean()

    .reset_index()
)


uniform_by_ecg = control_cv_df[
    control_cv_df[
        "condition"
    ]
    ==
    "uniform"
].copy()


real_extended_rows = []


for _, row in original_metrics.iterrows():

    population = row[
        "population"
    ]

    ecg = row[
        "ECG"
    ]

    (
        target,
        baseline,
        trained
    ) = get_original_attention_vectors(
        population,
        ecg
    )

    extra = metric_bundle(
        trained,
        target
    )

    real_extended_rows.append(
        {

            "population":
                population,

            "fold":
                int(
                    row[
                        "fold"
                    ]
                ),

            "ECG":
                ecg,

            **extra,
        }
    )


real_extended_df = pd.DataFrame(
    real_extended_rows
)


direct_comparison = (

    real_extended_df

    .merge(
        mean_gaze_df[
            [
                "population",
                "fold",
                "ECG",
                "mean_gaze_cosine",
                "mean_gaze_centered_cosine",
                "mean_gaze_MSE",
                "mean_gaze_JSD",
                "mean_gaze_TV",
            ]
        ],
        on=[
            "population",
            "fold",
            "ECG",
        ],
        how="left"
    )

    .merge(
        uniform_by_ecg[
            [
                "population",
                "fold",
                "ECG",
                "cosine",
                "centered_cosine",
                "MSE",
                "JSD",
                "TV",
            ]
        ].rename(
            columns={
                "cosine":
                    "uniform_cosine",

                "centered_cosine":
                    "uniform_centered_cosine",

                "MSE":
                    "uniform_MSE",

                "JSD":
                    "uniform_JSD",

                "TV":
                    "uniform_TV",
            }
        ),
        on=[
            "population",
            "fold",
            "ECG",
        ],
        how="left"
    )

    .merge(
        permuted_mean_by_ecg[
            [
                "population",
                "fold",
                "ECG",
                "cosine",
                "centered_cosine",
                "MSE",
                "JSD",
                "TV",
            ]
        ].rename(
            columns={
                "cosine":
                    "permuted_mean_cosine",

                "centered_cosine":
                    "permuted_mean_centered_cosine",

                "MSE":
                    "permuted_mean_MSE",

                "JSD":
                    "permuted_mean_JSD",

                "TV":
                    "permuted_mean_TV",
            }
        ),
        on=[
            "population",
            "fold",
            "ECG",
        ],
        how="left"
    )
)


direct_comparison.to_csv(
    CONTROL_CV_DIR
    /
    "real_vs_control_alignment_by_ECG.csv",
    index=False
)


# ==========================================================================================
# 23. TRAIN FINAL FULL-DATA CONTROL MODELS
#
# EXACT full-data augmentation/batch schedule from original Step 3.
# ==========================================================================================

print("\n\n" + "=" * 110)
print("TRAINING FULL-DATA CONTROL MODELS")
print("=" * 110)


augmentation_rng = np.random.default_rng(
    SEED
    +
    500000
)


X_full_base = np.repeat(
    X,
    AUGMENTATIONS_PER_ECG,
    axis=0
).astype(
    np.float32
)


full_noise = augmentation_rng.normal(
    loc=0.0,
    scale=NOISE_SD,
    size=X_full_base.shape
).astype(
    np.float32
)


X_full_augmented = (
    X_full_base
    +
    full_noise
).astype(
    np.float32
)


if X_full_augmented.shape != (
    400,
    12,
    1000,
    1
):

    raise RuntimeError(
        "Unexpected full control-training shape."
    )


full_permutation_rng = np.random.default_rng(
    SEED
    +
    600000
)


full_epoch_permutations = [

    full_permutation_rng.permutation(
        400
    )

    for _
    in range(
        EPOCHS
    )
]


# ------------------------------------------------------------------------------------------
# Uniform full-data model
# ------------------------------------------------------------------------------------------

uniform_full_targets = np.repeat(
    UNIFORM_VECTOR[
        None,
        :
    ],
    10,
    axis=0
).astype(
    np.float32
)


UNIFORM_FULL_WEIGHTS = (
    CONTROL_MODEL_DIR
    /
    "uniform_control.weights.h5"
)


(
    uniform_full_model,
    uniform_full_rep,
    uniform_full_history,
    uniform_full_audit
) = train_control_model(

    X_train_augmented=
        X_full_augmented,

    target_matrix_for_original_ECGs=
        uniform_full_targets,

    epoch_permutations=
        full_epoch_permutations,

    save_path=
        UNIFORM_FULL_WEIGHTS,

    verbose_label=
        "FULL-DATA UNIFORM CONTROL"
)


UNIFORM_FULL_HASH = sha256_file(
    UNIFORM_FULL_WEIGHTS
)


del uniform_full_rep
del uniform_full_model

gc.collect()


# ------------------------------------------------------------------------------------------
# ONE PRE-SPECIFIED FULL-DATA GAZE DERANGEMENT
#
# Same ECG-to-wrong-ECG mapping for both gaze populations.
# ------------------------------------------------------------------------------------------

FULL_PERMUTED_INDICES = deterministic_derangement(
    np.arange(
        10
    ),
    seed=(
        SEED
        +
        700000
    )
)


print("\nPre-specified full-data gaze permutation:")

for source_idx, target_idx in zip(
    np.arange(
        10
    ),
    FULL_PERMUTED_INDICES
):

    print(
        f"  {ECG_ORDER[source_idx]:>20s}  ->  "
        f"{ECG_ORDER[target_idx]}"
    )

    permutation_manifest_rows.append(
        {

            "scope":
                "full_data_primary_permuted_control",

            "repeat":
                1,

            "fold":
                0,

            "training_ECG":
                ECG_ORDER[
                    source_idx
                ],

            "assigned_gaze_target":
                ECG_ORDER[
                    target_idx
                ],
        }
    )


FULL_PERMUTED_WEIGHT_PATHS = {}
FULL_PERMUTED_HASHES = {}


for population in POPULATIONS:

    if population == "consultant":

        weight_path = (
            CONTROL_MODEL_DIR
            /
            "permuted_consultant_control.weights.h5"
        )

    else:

        weight_path = (
            CONTROL_MODEL_DIR
            /
            "permuted_all_clinicians_control.weights.h5"
        )

    permuted_full_targets = Y[
        population
    ][
        FULL_PERMUTED_INDICES
    ]

    (
        perm_full_model,
        perm_full_rep,
        perm_full_history,
        perm_full_audit
    ) = train_control_model(

        X_train_augmented=
            X_full_augmented,

        target_matrix_for_original_ECGs=
            permuted_full_targets,

        epoch_permutations=
            full_epoch_permutations,

        save_path=
            weight_path,

        verbose_label=
            f"FULL-DATA PERMUTED {population.upper()}"
    )

    FULL_PERMUTED_WEIGHT_PATHS[
        population
    ] = weight_path

    FULL_PERMUTED_HASHES[
        population
    ] = sha256_file(
        weight_path
    )

    del perm_full_rep
    del perm_full_model

    gc.collect()


permutation_manifest_df = pd.DataFrame(
    permutation_manifest_rows
)


permutation_manifest_df.to_csv(
    CONTROL_DIR
    /
    "complete_permutation_manifest.csv",
    index=False
)


# ==========================================================================================
# 24. FULL-DATA CONTROL MODEL MANIFEST
# ==========================================================================================

control_model_manifest = {

    "stage":
        "Reviewer-requested auxiliary-target controls",

    "base_model":
        str(
            BASE_MODEL_PATH
        ),

    "base_model_sha256":
        BASE_HASH,

    "gaze_ECG_sha256":
        X_HASH,

    "gaze_target_sha256":
        GAZE_HASH,

    "attention_layer":
        "add_3",

    "attention_readout":
        (
            "RMS over time and channels, normalised across 12 leads"
        ),

    "trainable_layer":
        "conv2d_10",

    "classification_loss":
        None,

    "auxiliary_loss":
        "mean squared error",

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "augmentations_per_ECG":
        AUGMENTATIONS_PER_ECG,

    "noise_SD":
        NOISE_SD,

    "uniform_control_weights":
        str(
            UNIFORM_FULL_WEIGHTS
        ),

    "uniform_control_sha256":
        UNIFORM_FULL_HASH,

    "permuted_consultant_weights":
        str(
            FULL_PERMUTED_WEIGHT_PATHS[
                "consultant"
            ]
        ),

    "permuted_consultant_sha256":
        FULL_PERMUTED_HASHES[
            "consultant"
        ],

    "permuted_all_clinicians_weights":
        str(
            FULL_PERMUTED_WEIGHT_PATHS[
                "all_clinicians"
            ]
        ),

    "permuted_all_clinicians_sha256":
        FULL_PERMUTED_HASHES[
            "all_clinicians"
        ],

    "full_data_permutation_indices":
        FULL_PERMUTED_INDICES.tolist(),

    "full_data_permutation_mapping": {

        ECG_ORDER[
            i
        ]:
            ECG_ORDER[
                int(
                    FULL_PERMUTED_INDICES[
                        i
                    ]
                )
            ]

        for i
        in range(
            10
        )
    },
}


with open(
    CONTROL_MODEL_DIR
    /
    "control_model_manifest.json",
    "w"
) as f:

    json.dump(
        control_model_manifest,
        f,
        indent=2
    )


print(
    "\n✓ Full-data uniform and permuted control models trained and frozen."
)


# ==========================================================================================
# 25. PTB-XL CONTROL EVALUATION
# ==========================================================================================

if RUN_PTBXL_CONTROLS:

    print("\n\n" + "#" * 110)
    print("PTB-XL CONTROL EVALUATION")
    print("#" * 110)

    PTBXL_ROOT = Path(
        "ptb-xl"
    )

    PTBXL_DATABASE_PATH = (
        PTBXL_ROOT
        /
        "ptbxl_database.csv"
    )

    ORIGINAL_PTBXL_PRED_PATH = (
        ROOT
        /
        "ptbxl_evaluation"
        /
        "ptbxl_fold10_predictions.csv"
    )

    for path in [
        PTBXL_DATABASE_PATH,
        ORIGINAL_PTBXL_PRED_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )


    # ======================================================================================
    # 25A. LOAD ORIGINAL SAVED FOLD-10 RESULTS
    # ======================================================================================

    ptb_saved = pd.read_csv(
        ORIGINAL_PTBXL_PRED_PATH
    )


    if len(
        ptb_saved
    ) != 2198:

        raise RuntimeError(
            "Expected 2198 saved PTB-XL predictions."
        )


    Y_ptb = np.column_stack(
        [

            ptb_saved[
                f"true__{class_name}"
            ].to_numpy(
                dtype=np.int8
            )

            for class_name
            in MODEL_CLASSES
        ]
    )


    P_ptb = {

        "BASE":
            np.column_stack(
                [

                    ptb_saved[
                        f"base__{class_name}"
                    ].to_numpy(
                        dtype=np.float64
                    )

                    for class_name
                    in MODEL_CLASSES
                ]
            ),

        "GAZE_CONSULTANT":
            np.column_stack(
                [

                    ptb_saved[
                        f"consultant__{class_name}"
                    ].to_numpy(
                        dtype=np.float64
                    )

                    for class_name
                    in MODEL_CLASSES
                ]
            ),

        "GAZE_ALL_CLINICIANS":
            np.column_stack(
                [

                    ptb_saved[
                        f"all_clinicians__{class_name}"
                    ].to_numpy(
                        dtype=np.float64
                    )

                    for class_name
                    in MODEL_CLASSES
                ]
            ),
    }


    # ======================================================================================
    # 25B. LOAD/CACHE EXACT FOLD-10 WAVEFORMS IN SAVED-PREDICTION ORDER
    # ======================================================================================

    PTB_X_CACHE = (
        CONTROL_PTBXL_DIR
        /
        "ptbxl_fold10_X.npy"
    )

    PTB_ID_CACHE = (
        CONTROL_PTBXL_DIR
        /
        "ptbxl_fold10_ecg_ids.npy"
    )


    saved_ecg_ids = ptb_saved[
        "ecg_id"
    ].to_numpy()


    if (
        PTB_X_CACHE.exists()
        and
        PTB_ID_CACHE.exists()
    ):

        cached_ids = np.load(
            PTB_ID_CACHE
        )

        if not np.array_equal(
            cached_ids,
            saved_ecg_ids
        ):

            raise RuntimeError(
                "Cached PTB-XL IDs do not match saved prediction order."
            )

        X_ptb = np.load(
            PTB_X_CACHE
        ).astype(
            np.float32
        )

        print(
            "Loaded cached PTB-XL fold-10 waveforms."
        )

    else:

        metadata = pd.read_csv(
            PTBXL_DATABASE_PATH,
            index_col="ecg_id"
        )

        missing_ids = set(
            saved_ecg_ids
        ) - set(
            metadata.index
        )

        if missing_ids:

            raise RuntimeError(
                "Saved PTB-XL ECG IDs missing from metadata."
            )

        X_ptb = np.empty(
            (
                2198,
                12,
                1000,
                1
            ),
            dtype=np.float32
        )

        for i, ecg_id in enumerate(
            saved_ecg_ids
        ):

            row = metadata.loc[
                ecg_id
            ]

            record_path = (
                PTBXL_ROOT
                /
                str(
                    row[
                        "filename_lr"
                    ]
                )
            )

            signal, fields = wfdb.rdsamp(
                str(
                    record_path
                )
            )

            if signal.shape != (
                1000,
                12
            ):

                raise RuntimeError(
                    f"PTB-XL {ecg_id}: bad shape {signal.shape}."
                )

            leads = [
                str(x).upper()
                for x
                in fields[
                    "sig_name"
                ]
            ]

            if leads != EXPECTED_LEADS_UPPER:

                raise RuntimeError(
                    f"PTB-XL {ecg_id}: lead-order mismatch."
                )

            X_ptb[
                i,
                :,
                :,
                0
            ] = signal.T.astype(
                np.float32
            )

            if (
                (i + 1) % 250 == 0
                or
                (i + 1) == 2198
            ):

                print(
                    f"Loaded PTB-XL {i + 1}/2198"
                )

        np.save(
            PTB_X_CACHE,
            X_ptb
        )

        np.save(
            PTB_ID_CACHE,
            saved_ecg_ids
        )


    if X_ptb.shape != (
        2198,
        12,
        1000,
        1
    ):

        raise RuntimeError(
            f"Bad PTB-XL control X shape: {X_ptb.shape}"
        )


    # ======================================================================================
    # 25C. LOAD CONTROL MODEL + PREDICT
    # ======================================================================================

    def load_model_with_control_weights(
        weight_path
    ):

        tf.keras.backend.clear_session()

        model = tf.keras.models.load_model(
            BASE_MODEL_PATH,
            compile=False
        )

        model.load_weights(
            weight_path
        )

        if model.output_shape != (
            None,
            23
        ):

            raise RuntimeError(
                "Unexpected diagnostic output shape."
            )

        return model


    def predict_diagnostic_control(
        weight_path,
        label
    ):

        model = load_model_with_control_weights(
            weight_path
        )

        print(
            f"\nPTB-XL inference: {label}"
        )

        prediction = model.predict(
            X_ptb,
            batch_size=PRED_BATCH_SIZE,
            verbose=1
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float64
        )

        if prediction.shape != (
            2198,
            23
        ):

            raise RuntimeError(
                f"{label}: bad prediction shape."
            )

        del model
        gc.collect()

        return prediction


    P_ptb[
        "UNIFORM"
    ] = predict_diagnostic_control(
        UNIFORM_FULL_WEIGHTS,
        "UNIFORM"
    )


    P_ptb[
        "PERMUTED_CONSULTANT"
    ] = predict_diagnostic_control(
        FULL_PERMUTED_WEIGHT_PATHS[
            "consultant"
        ],
        "PERMUTED CONSULTANT"
    )


    P_ptb[
        "PERMUTED_ALL_CLINICIANS"
    ] = predict_diagnostic_control(
        FULL_PERMUTED_WEIGHT_PATHS[
            "all_clinicians"
        ],
        "PERMUTED ALL CLINICIANS"
    )


    # ======================================================================================
    # 25D. PTB-XL METRICS
    # ======================================================================================

    def ptb_metrics(
        y_true,
        y_score
    ):

        per_auc = []
        per_ap = []

        for j in range(
            y_true.shape[
                1
            ]
        ):

            per_auc.append(
                roc_auc_score(
                    y_true[
                        :,
                        j
                    ],
                    y_score[
                        :,
                        j
                    ]
                )
            )

            per_ap.append(
                average_precision_score(
                    y_true[
                        :,
                        j
                    ],
                    y_score[
                        :,
                        j
                    ]
                )
            )

        return {

            "macro_AUROC":
                float(
                    np.mean(
                        per_auc
                    )
                ),

            "macro_AP":
                float(
                    np.mean(
                        per_ap
                    )
                ),

            "micro_AUROC":
                float(
                    roc_auc_score(
                        y_true.ravel(),
                        y_score.ravel()
                    )
                ),

            "micro_AP":
                float(
                    average_precision_score(
                        y_true.ravel(),
                        y_score.ravel()
                    )
                ),

            "per_class_AUROC":
                np.asarray(
                    per_auc
                ),

            "per_class_AP":
                np.asarray(
                    per_ap
                ),
        }


    ptb_point = {

        name:
            ptb_metrics(
                Y_ptb,
                predictions
            )

        for name, predictions
        in P_ptb.items()
    }


    ptb_macro_rows = []


    for model_name, m in ptb_point.items():

        ptb_macro_rows.append(
            {

                "model":
                    model_name,

                "macro_AUROC":
                    m[
                        "macro_AUROC"
                    ],

                "macro_AP":
                    m[
                        "macro_AP"
                    ],

                "micro_AUROC":
                    m[
                        "micro_AUROC"
                    ],

                "micro_AP":
                    m[
                        "micro_AP"
                    ],
            }
        )


    ptb_macro_df = pd.DataFrame(
        ptb_macro_rows
    )


    base_point = ptb_point[
        "BASE"
    ]


    for metric in [
        "macro_AUROC",
        "macro_AP",
        "micro_AUROC",
        "micro_AP",
    ]:

        ptb_macro_df[
            f"delta_{metric}_vs_BASE"
        ] = (
            ptb_macro_df[
                metric
            ]
            -
            base_point[
                metric
            ]
        )


    ptb_macro_df.to_csv(
        CONTROL_PTBXL_DIR
        /
        "ptbxl_control_macro_summary.csv",
        index=False
    )


    print("\n\n" + "=" * 110)
    print("PTB-XL — GAZE vs UNIFORM vs PERMUTED CONTROLS")
    print("=" * 110)

    print(
        ptb_macro_df
        .round(
            6
        )
        .to_string(
            index=False
        )
    )


    # ======================================================================================
    # 25E. SAVE COMPLETE PTB CONTROL PREDICTIONS
    # ======================================================================================

    ptb_control_predictions = ptb_saved[
        [
            "ecg_id",
            "patient_id",
            "all_zero_target",
        ]
    ].copy()


    for j, class_name in enumerate(
        MODEL_CLASSES
    ):

        ptb_control_predictions[
            f"true__{class_name}"
        ] = Y_ptb[
            :,
            j
        ]

        for model_name, probabilities in P_ptb.items():

            ptb_control_predictions[
                f"{model_name}__{class_name}"
            ] = probabilities[
                :,
                j
            ]


    ptb_control_predictions.to_csv(
        CONTROL_PTBXL_DIR
        /
        "ptbxl_control_predictions.csv",
        index=False
    )


    # ======================================================================================
    # 26. PATIENT-CLUSTER BOOTSTRAP OF CONTROL COMPARISONS
    # ======================================================================================

    if RUN_PTBXL_BOOTSTRAP:

        print("\n" + "=" * 110)
        print("PTB-XL CONTROL PATIENT-CLUSTER BOOTSTRAP")
        print("=" * 110)

        patient_groups = (
            ptb_saved
            .groupby(
                "patient_id",
                sort=False
            )
            .indices
        )

        patient_ids = np.asarray(
            list(
                patient_groups.keys()
            ),
            dtype=object
        )

        comparisons = [

            (
                "GAZE_CONSULTANT_vs_BASE",
                "GAZE_CONSULTANT",
                "BASE"
            ),

            (
                "GAZE_ALL_CLINICIANS_vs_BASE",
                "GAZE_ALL_CLINICIANS",
                "BASE"
            ),

            (
                "UNIFORM_vs_BASE",
                "UNIFORM",
                "BASE"
            ),

            (
                "PERMUTED_CONSULTANT_vs_BASE",
                "PERMUTED_CONSULTANT",
                "BASE"
            ),

            (
                "PERMUTED_ALL_CLINICIANS_vs_BASE",
                "PERMUTED_ALL_CLINICIANS",
                "BASE"
            ),

            (
                "GAZE_CONSULTANT_vs_UNIFORM",
                "GAZE_CONSULTANT",
                "UNIFORM"
            ),

            (
                "GAZE_ALL_CLINICIANS_vs_UNIFORM",
                "GAZE_ALL_CLINICIANS",
                "UNIFORM"
            ),

            (
                "GAZE_CONSULTANT_vs_PERMUTED_CONSULTANT",
                "GAZE_CONSULTANT",
                "PERMUTED_CONSULTANT"
            ),

            (
                "GAZE_ALL_CLINICIANS_vs_PERMUTED_ALL_CLINICIANS",
                "GAZE_ALL_CLINICIANS",
                "PERMUTED_ALL_CLINICIANS"
            ),
        ]


        bootstrap_metrics = [
            "macro_AUROC",
            "macro_AP",
            "micro_AUROC",
            "micro_AP",
        ]


        boot_store = {

            comparison_name: {

                metric:
                    np.full(
                        N_BOOT,
                        np.nan,
                        dtype=np.float64
                    )

                for metric
                in bootstrap_metrics
            }

            for comparison_name, _, _
            in comparisons
        }


        def ptb_boot_metrics(
            y,
            p
        ):

            per_auc = []
            per_ap = []

            for j in range(
                y.shape[
                    1
                ]
            ):

                yj = y[
                    :,
                    j
                ]

                if (
                    np.sum(
                        yj == 1
                    )
                    ==
                    0
                    or
                    np.sum(
                        yj == 0
                    )
                    ==
                    0
                ):

                    return {
                        "macro_AUROC":
                            np.nan,

                        "macro_AP":
                            np.nan,

                        "micro_AUROC":
                            float(
                                roc_auc_score(
                                    y.ravel(),
                                    p.ravel()
                                )
                            ),

                        "micro_AP":
                            float(
                                average_precision_score(
                                    y.ravel(),
                                    p.ravel()
                                )
                            ),
                    }

                per_auc.append(
                    roc_auc_score(
                        yj,
                        p[
                            :,
                            j
                        ]
                    )
                )

                per_ap.append(
                    average_precision_score(
                        yj,
                        p[
                            :,
                            j
                        ]
                    )
                )

            return {

                "macro_AUROC":
                    float(
                        np.mean(
                            per_auc
                        )
                    ),

                "macro_AP":
                    float(
                        np.mean(
                            per_ap
                        )
                    ),

                "micro_AUROC":
                    float(
                        roc_auc_score(
                            y.ravel(),
                            p.ravel()
                        )
                    ),

                "micro_AP":
                    float(
                        average_precision_score(
                            y.ravel(),
                            p.ravel()
                        )
                    ),
            }


        rng = np.random.default_rng(
            SEED
            +
            800000
        )


        for b in range(
            N_BOOT
        ):

            sampled_positions = rng.integers(
                0,
                len(
                    patient_ids
                ),
                size=len(
                    patient_ids
                )
            )

            sampled_patients = patient_ids[
                sampled_positions
            ]

            idx = np.concatenate(
                [

                    np.asarray(
                        patient_groups[
                            patient_id
                        ],
                        dtype=int
                    )

                    for patient_id
                    in sampled_patients
                ]
            )

            y_boot = Y_ptb[
                idx
            ]

            metric_cache = {}

            for model_name, probabilities in P_ptb.items():

                metric_cache[
                    model_name
                ] = ptb_boot_metrics(
                    y_boot,
                    probabilities[
                        idx
                    ]
                )

            for comparison_name, model_a, model_b in comparisons:

                for metric in bootstrap_metrics:

                    a = metric_cache[
                        model_a
                    ][
                        metric
                    ]

                    b_value = metric_cache[
                        model_b
                    ][
                        metric
                    ]

                    if (
                        np.isfinite(
                            a
                        )
                        and
                        np.isfinite(
                            b_value
                        )
                    ):

                        boot_store[
                            comparison_name
                        ][
                            metric
                        ][
                            b
                        ] = (
                            a
                            -
                            b_value
                        )

            if (
                (b + 1) % 250 == 0
                or
                (b + 1) == N_BOOT
            ):

                print(
                    f"Completed {b + 1}/{N_BOOT}"
                )


        ptb_boot_rows = []


        for comparison_name, model_a, model_b in comparisons:

            for metric in bootstrap_metrics:

                values = boot_store[
                    comparison_name
                ][
                    metric
                ]

                values = values[
                    np.isfinite(
                        values
                    )
                ]

                point_delta = (
                    ptb_point[
                        model_a
                    ][
                        metric
                    ]
                    -
                    ptb_point[
                        model_b
                    ][
                        metric
                    ]
                )

                ci_low, ci_high = np.percentile(
                    values,
                    [
                        2.5,
                        97.5
                    ]
                )

                ptb_boot_rows.append(
                    {

                        "comparison":
                            comparison_name,

                        "metric":
                            metric,

                        "point_delta":
                            point_delta,

                        "bootstrap_mean_delta":
                            float(
                                np.mean(
                                    values
                                )
                            ),

                        "CI_low":
                            float(
                                ci_low
                            ),

                        "CI_high":
                            float(
                                ci_high
                            ),

                        "valid_bootstraps":
                            int(
                                len(
                                    values
                                )
                            ),

                        "fraction_below_zero":
                            float(
                                np.mean(
                                    values < 0
                                )
                            ),

                        "fraction_above_zero":
                            float(
                                np.mean(
                                    values > 0
                                )
                            ),
                    }
                )


        ptb_boot_df = pd.DataFrame(
            ptb_boot_rows
        )


        ptb_boot_df.to_csv(
            CONTROL_PTBXL_DIR
            /
            "ptbxl_control_patient_cluster_bootstrap.csv",
            index=False
        )


        print("\n\n" + "=" * 110)
        print("PTB-XL CONTROL BOOTSTRAP — KEY COMPARISONS")
        print("=" * 110)

        print(
            ptb_boot_df[
                ptb_boot_df[
                    "metric"
                ].isin(
                    [
                        "macro_AUROC",
                        "macro_AP",
                    ]
                )
            ]
            .round(
                6
            )
            .to_string(
                index=False
            )
        )


# ==========================================================================================
# 27. LUDB CONTROL REPRESENTATION-TRANSFER ANALYSIS
# ==========================================================================================

if RUN_LUDB_CONTROLS:

    print("\n\n" + "#" * 110)
    print("LUDB CONTROL REPRESENTATION-TRANSFER ANALYSIS")
    print("#" * 110)

    ORIGINAL_LUDB_DIR = (
        ROOT
        /
        "ludb_evaluation"
    )

    LUDB_X_PATH = (
        ORIGINAL_LUDB_DIR
        /
        "ludb_preprocessed_100hz.npy"
    )

    LUDB_TARGET_PATH = (
        ORIGINAL_LUDB_DIR
        /
        "ludb_frozen_targets.csv"
    )

    LUDB_ORIGINAL_PRED_PATH = (
        ORIGINAL_LUDB_DIR
        /
        "ludb_step5c_oof_predictions.csv"
    )

    LUDB_ORIGINAL_SUMMARY_PATH = (
        ORIGINAL_LUDB_DIR
        /
        "ludb_step5c_oof_summary.csv"
    )


    for path in [
        LUDB_X_PATH,
        LUDB_TARGET_PATH,
        LUDB_ORIGINAL_PRED_PATH,
        LUDB_ORIGINAL_SUMMARY_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )


    X_ludb = np.load(
        LUDB_X_PATH
    ).astype(
        np.float32
    )


    ludb_targets = pd.read_csv(
        LUDB_TARGET_PATH
    )


    ludb_original_predictions = pd.read_csv(
        LUDB_ORIGINAL_PRED_PATH
    )


    ludb_original_summary = pd.read_csv(
        LUDB_ORIGINAL_SUMMARY_PATH
    )


    if X_ludb.shape != (
        200,
        12,
        1000,
        1
    ):

        raise RuntimeError(
            f"Bad frozen LUDB X shape: {X_ludb.shape}"
        )


    if len(
        ludb_targets
    ) != 200:

        raise RuntimeError(
            "Expected 200 frozen LUDB targets."
        )


    # ======================================================================================
    # 27A. EXTRACT dense_1 FEATURES FROM CONTROL MODELS
    # ======================================================================================

    FEATURE_LAYER = "dense_1"
    FEATURE_DIM = 64


    def extract_control_features(
        weight_path,
        model_name
    ):

        tf.keras.backend.clear_session()

        model = tf.keras.models.load_model(
            BASE_MODEL_PATH,
            compile=False
        )

        model.load_weights(
            weight_path
        )

        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(
                FEATURE_LAYER
            ).output
        )

        Z = feature_model.predict(
            X_ludb,
            batch_size=PRED_BATCH_SIZE,
            verbose=1
        )

        Z = np.asarray(
            Z,
            dtype=np.float32
        )

        if Z.shape != (
            200,
            FEATURE_DIM
        ):

            raise RuntimeError(
                f"{model_name}: bad LUDB feature shape {Z.shape}."
            )

        if not np.all(
            np.isfinite(
                Z
            )
        ):

            raise RuntimeError(
                f"{model_name}: non-finite LUDB features."
            )

        del feature_model
        del model

        gc.collect()

        return Z


    Z_uniform = extract_control_features(
        UNIFORM_FULL_WEIGHTS,
        "UNIFORM"
    )


    Z_perm_consultant = extract_control_features(
        FULL_PERMUTED_WEIGHT_PATHS[
            "consultant"
        ],
        "PERMUTED_CONSULTANT"
    )


    Z_perm_all = extract_control_features(
        FULL_PERMUTED_WEIGHT_PATHS[
            "all_clinicians"
        ],
        "PERMUTED_ALL_CLINICIANS"
    )


    CONTROL_FEATURES = {

        "UNIFORM":
            Z_uniform,

        "PERMUTED_CONSULTANT":
            Z_perm_consultant,

        "PERMUTED_ALL_CLINICIANS":
            Z_perm_all,
    }


    for model_name, Z in CONTROL_FEATURES.items():

        np.save(
            CONTROL_LUDB_DIR
            /
            f"ludb_features_{model_name.lower()}.npy",
            Z
        )


    # ======================================================================================
    # 27B. USE EXACT SAVED STEP-5C FOLD ASSIGNMENTS
    #
    # No new endpoint definition.
    # No new CV split.
    # ======================================================================================

    id_to_feature_row = {

        int(
            record_id
        ):
            i

        for i, record_id
        in enumerate(
            ludb_targets[
                "ID"
            ].astype(
                int
            )
        )
    }


    def make_ludb_probe():

        return Pipeline(
            [

                (
                    "scale",
                    StandardScaler()
                ),

                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=SEED,
                    )
                ),
            ]
        )


    def binary_metrics_from_probabilities(
        y_true,
        y_prob
    ):

        y_pred = (
            y_prob
            >=
            0.5
        ).astype(
            int
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1
            ]
        ).ravel()

        sensitivity = (
            tp
            /
            (
                tp + fn
            )
            if (
                tp + fn
            )
            else np.nan
        )

        specificity = (
            tn
            /
            (
                tn + fp
            )
            if (
                tn + fp
            )
            else np.nan
        )

        return {

            "AUROC":
                float(
                    roc_auc_score(
                        y_true,
                        y_prob
                    )
                ),

            "AP":
                float(
                    average_precision_score(
                        y_true,
                        y_prob
                    )
                ),

            "balanced_accuracy":
                float(
                    balanced_accuracy_score(
                        y_true,
                        y_pred
                    )
                ),

            "accuracy":
                float(
                    accuracy_score(
                        y_true,
                        y_pred
                    )
                ),

            "F1":
                float(
                    f1_score(
                        y_true,
                        y_pred,
                        zero_division=0
                    )
                ),

            "sensitivity":
                float(
                    sensitivity
                ),

            "specificity":
                float(
                    specificity
                ),
        }


    ludb_control_prediction_rows = []
    ludb_control_summary_rows = []


    tasks = [
        "Conduction",
        "LVH",
        "Ischaemia_related",
        "AFib",
        "Paced",
    ]


    for task in tasks:

        original_base_task = ludb_original_predictions[
            (
                ludb_original_predictions[
                    "task"
                ]
                ==
                task
            )
            &
            (
                ludb_original_predictions[
                    "model"
                ]
                ==
                "BASE"
            )
        ].copy()


        if len(
            original_base_task
        ) == 0:

            raise RuntimeError(
                f"No original BASE predictions found for LUDB task {task}."
            )


        original_base_task = (
            original_base_task
            .sort_values(
                "record_id"
            )
            .reset_index(
                drop=True
            )
        )


        record_ids = original_base_task[
            "record_id"
        ].astype(
            int
        ).to_numpy()


        y = original_base_task[
            "true_label"
        ].astype(
            int
        ).to_numpy()


        fold_assignment = original_base_task[
            "fold"
        ].astype(
            int
        ).to_numpy()


        feature_rows = np.asarray(
            [

                id_to_feature_row[
                    int(
                        record_id
                    )
                ]

                for record_id
                in record_ids
            ],
            dtype=int
        )


        role = (
            "primary"
            if task in [
                "Conduction",
                "LVH",
                "Ischaemia_related",
            ]
            else
            "exploratory"
        )


        for model_name, Z_all in CONTROL_FEATURES.items():

            Z = Z_all[
                feature_rows
            ]

            oof_prob = np.full(
                len(
                    y
                ),
                np.nan,
                dtype=np.float64
            )


            for fold_number in sorted(
                np.unique(
                    fold_assignment
                )
            ):

                test_mask = (
                    fold_assignment
                    ==
                    fold_number
                )

                train_mask = ~test_mask

                if len(
                    np.unique(
                        y[
                            train_mask
                        ]
                    )
                ) < 2:

                    raise RuntimeError(
                        f"{task}/{model_name}/fold{fold_number}: one-class train fold."
                    )

                probe = make_ludb_probe()

                probe.fit(
                    Z[
                        train_mask
                    ],
                    y[
                        train_mask
                    ]
                )

                oof_prob[
                    test_mask
                ] = probe.predict_proba(
                    Z[
                        test_mask
                    ]
                )[
                    :,
                    1
                ]


            if not np.all(
                np.isfinite(
                    oof_prob
                )
            ):

                raise RuntimeError(
                    f"{task}/{model_name}: missing OOF probabilities."
                )


            m = binary_metrics_from_probabilities(
                y,
                oof_prob
            )


            ludb_control_summary_rows.append(
                {

                    "task":
                        task,

                    "role":
                        role,

                    "model":
                        model_name,

                    "N":
                        int(
                            len(
                                y
                            )
                        ),

                    "positive":
                        int(
                            y.sum()
                        ),

                    **m,
                }
            )


            for i in range(
                len(
                    y
                )
            ):

                ludb_control_prediction_rows.append(
                    {

                        "task":
                            task,

                        "role":
                            role,

                        "model":
                            model_name,

                        "record_id":
                            int(
                                record_ids[
                                    i
                                ]
                            ),

                        "fold":
                            int(
                                fold_assignment[
                                    i
                                ]
                            ),

                        "true_label":
                            int(
                                y[
                                    i
                                ]
                            ),

                        "probability":
                            float(
                                oof_prob[
                                    i
                                ]
                            ),

                        "predicted_label":
                            int(
                                oof_prob[
                                    i
                                ]
                                >=
                                0.5
                            ),
                    }
                )


    ludb_control_summary_df = pd.DataFrame(
        ludb_control_summary_rows
    )


    ludb_control_predictions_df = pd.DataFrame(
        ludb_control_prediction_rows
    )


    ludb_control_summary_df.to_csv(
        CONTROL_LUDB_DIR
        /
        "ludb_control_oof_summary.csv",
        index=False
    )


    ludb_control_predictions_df.to_csv(
        CONTROL_LUDB_DIR
        /
        "ludb_control_oof_predictions.csv",
        index=False
    )


    # Combine with original frozen results.
    ludb_combined_summary = pd.concat(
        [
            ludb_original_summary,
            ludb_control_summary_df,
        ],
        ignore_index=True
    )


    ludb_combined_summary.to_csv(
        CONTROL_LUDB_DIR
        /
        "ludb_original_plus_control_summary.csv",
        index=False
    )


    print("\n\n" + "=" * 110)
    print("LUDB — ORIGINAL GAZE MODELS + CONTROL MODELS")
    print("=" * 110)

    print(
        ludb_combined_summary[
            [
                "task",
                "model",
                "N",
                "positive",
                "AUROC",
                "AP",
                "balanced_accuracy",
            ]
        ]
        .round(
            6
        )
        .to_string(
            index=False
        )
    )


    # ======================================================================================
    # 28. LUDB PAIRED BOOTSTRAP — CONTROLS
    # ======================================================================================

    if RUN_LUDB_BOOTSTRAP:

        print("\n" + "=" * 110)
        print("LUDB CONTROL PAIRED BOOTSTRAP")
        print("=" * 110)

        original_for_boot = ludb_original_predictions[
            ludb_original_predictions[
                "model"
            ].isin(
                [
                    "BASE",
                    "CONSULTANT",
                    "ALL_CLINICIANS",
                ]
            )
        ].copy()


        all_ludb_predictions = pd.concat(
            [
                original_for_boot,
                ludb_control_predictions_df,
            ],
            ignore_index=True
        )


        ludb_boot_comparisons = [

            (
                "UNIFORM_vs_BASE",
                "UNIFORM",
                "BASE"
            ),

            (
                "PERMUTED_CONSULTANT_vs_BASE",
                "PERMUTED_CONSULTANT",
                "BASE"
            ),

            (
                "PERMUTED_ALL_CLINICIANS_vs_BASE",
                "PERMUTED_ALL_CLINICIANS",
                "BASE"
            ),

            (
                "CONSULTANT_vs_UNIFORM",
                "CONSULTANT",
                "UNIFORM"
            ),

            (
                "ALL_CLINICIANS_vs_UNIFORM",
                "ALL_CLINICIANS",
                "UNIFORM"
            ),

            (
                "CONSULTANT_vs_PERMUTED_CONSULTANT",
                "CONSULTANT",
                "PERMUTED_CONSULTANT"
            ),

            (
                "ALL_CLINICIANS_vs_PERMUTED_ALL_CLINICIANS",
                "ALL_CLINICIANS",
                "PERMUTED_ALL_CLINICIANS"
            ),
        ]


        ludb_boot_rows = []


        for task in tasks:

            task_frame = all_ludb_predictions[
                all_ludb_predictions[
                    "task"
                ]
                ==
                task
            ].copy()


            model_frames = {}


            for model_name in [
                "BASE",
                "CONSULTANT",
                "ALL_CLINICIANS",
                "UNIFORM",
                "PERMUTED_CONSULTANT",
                "PERMUTED_ALL_CLINICIANS",
            ]:

                frame = (
                    task_frame[
                        task_frame[
                            "model"
                        ]
                        ==
                        model_name
                    ]
                    .sort_values(
                        "record_id"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                if len(
                    frame
                ) == 0:

                    raise RuntimeError(
                        f"{task}/{model_name}: no predictions."
                    )

                model_frames[
                    model_name
                ] = frame


            reference = model_frames[
                "BASE"
            ]


            ids = reference[
                "record_id"
            ].to_numpy()


            y = reference[
                "true_label"
            ].astype(
                int
            ).to_numpy()


            probabilities = {}


            for model_name, frame in model_frames.items():

                if not np.array_equal(
                    frame[
                        "record_id"
                    ].to_numpy(),
                    ids
                ):

                    raise RuntimeError(
                        f"{task}/{model_name}: record mismatch."
                    )

                if not np.array_equal(
                    frame[
                        "true_label"
                    ].astype(
                        int
                    ).to_numpy(),
                    y
                ):

                    raise RuntimeError(
                        f"{task}/{model_name}: target mismatch."
                    )

                probabilities[
                    model_name
                ] = frame[
                    "probability"
                ].to_numpy(
                    dtype=np.float64
                )


            point_metrics = {

                model_name:
                    binary_metrics_from_probabilities(
                        y,
                        probability
                    )

                for model_name, probability
                in probabilities.items()
            }


            rng = np.random.default_rng(
                SEED
                +
                900000
                +
                tasks.index(
                    task
                )
            )


            boot_values = {

                comparison_name: {

                    "AUROC":
                        [],

                    "AP":
                        [],

                    "balanced_accuracy":
                        [],
                }

                for comparison_name, _, _
                in ludb_boot_comparisons
            }


            for b in range(
                N_BOOT
            ):

                idx = rng.integers(
                    0,
                    len(
                        y
                    ),
                    size=len(
                        y
                    )
                )

                y_boot = y[
                    idx
                ]

                if len(
                    np.unique(
                        y_boot
                    )
                ) < 2:

                    continue

                m_boot = {}

                for model_name, p in probabilities.items():

                    m_boot[
                        model_name
                    ] = binary_metrics_from_probabilities(
                        y_boot,
                        p[
                            idx
                        ]
                    )

                for comparison_name, model_a, model_b in ludb_boot_comparisons:

                    for metric in [
                        "AUROC",
                        "AP",
                        "balanced_accuracy",
                    ]:

                        boot_values[
                            comparison_name
                        ][
                            metric
                        ].append(
                            (
                                m_boot[
                                    model_a
                                ][
                                    metric
                                ]
                                -
                                m_boot[
                                    model_b
                                ][
                                    metric
                                ]
                            )
                        )


            role = (
                "primary"
                if task in [
                    "Conduction",
                    "LVH",
                    "Ischaemia_related",
                ]
                else
                "exploratory"
            )


            for comparison_name, model_a, model_b in ludb_boot_comparisons:

                for metric in [
                    "AUROC",
                    "AP",
                    "balanced_accuracy",
                ]:

                    values = np.asarray(
                        boot_values[
                            comparison_name
                        ][
                            metric
                        ],
                        dtype=np.float64
                    )

                    point_delta = (
                        point_metrics[
                            model_a
                        ][
                            metric
                        ]
                        -
                        point_metrics[
                            model_b
                        ][
                            metric
                        ]
                    )

                    ci_low, ci_high = np.percentile(
                        values,
                        [
                            2.5,
                            97.5
                        ]
                    )

                    ludb_boot_rows.append(
                        {

                            "task":
                                task,

                            "role":
                                role,

                            "comparison":
                                comparison_name,

                            "metric":
                                metric,

                            "point_delta":
                                float(
                                    point_delta
                                ),

                            "CI_low":
                                float(
                                    ci_low
                                ),

                            "CI_high":
                                float(
                                    ci_high
                                ),

                            "valid_bootstraps":
                                int(
                                    len(
                                        values
                                    )
                                ),
                        }
                    )


        ludb_boot_df = pd.DataFrame(
            ludb_boot_rows
        )


        ludb_boot_df.to_csv(
            CONTROL_LUDB_DIR
            /
            "ludb_control_paired_bootstrap.csv",
            index=False
        )


        print("\n\n" + "=" * 110)
        print("LUDB CONTROL BOOTSTRAP — PRIMARY TASKS")
        print("=" * 110)

        print(
            ludb_boot_df[
                (
                    ludb_boot_df[
                        "role"
                    ]
                    ==
                    "primary"
                )
                &
                (
                    ludb_boot_df[
                        "metric"
                    ].isin(
                        [
                            "AUROC",
                            "AP",
                        ]
                    )
                )
            ]
            .round(
                6
            )
            .to_string(
                index=False
            )
        )


# ==========================================================================================
# 29. MASTER MANIFEST
# ==========================================================================================

master_manifest = {

    "analysis":
        "Reviewer-requested control reanalysis",

    "seed":
        SEED,

    "original_base_model_sha256":
        BASE_HASH,

    "original_consultant_model_sha256":
        ORIGINAL_CONSULTANT_HASH,

    "original_all_clinician_model_sha256":
        ORIGINAL_ALL_HASH,

    "frozen_gaze_ECG_sha256":
        X_HASH,

    "frozen_gaze_target_sha256":
        GAZE_HASH,

    "three_requested_controls": [

        "training-fold mean-gaze baseline",

        "uniform 1/12 lead target",

        "permuted gaze target",
    ],

    "permuted_CV_repetitions":
        N_PERMUTED_CV_REPEATS,

    "full_data_primary_permuted_control":
        (
            "one deterministic derangement shared between "
            "consultant and all-clinician target sets"
        ),

    "classification_loss_during_control_training":
        None,

    "training_regime_matched_to_original":
        True,

    "attention_layer":
        "add_3",

    "attention_readout":
        (
            "RMS activation energy over temporal/channel axes, "
            "normalised across 12 leads"
        ),

    "only_trainable_layer":
        "conv2d_10",

    "epochs":
        EPOCHS,

    "augmentations_per_ECG":
        AUGMENTATIONS_PER_ECG,

    "noise_SD":
        NOISE_SD,

    "learning_rate":
        LEARNING_RATE,

    "batch_size":
        BATCH_SIZE,

    "PTBXL_control_evaluation":
        RUN_PTBXL_CONTROLS,

    "PTBXL_patient_cluster_bootstrap":
        RUN_PTBXL_BOOTSTRAP,

    "LUDB_control_evaluation":
        RUN_LUDB_CONTROLS,

    "LUDB_control_bootstrap":
        RUN_LUDB_BOOTSTRAP,

    "output_directory":
        str(
            CONTROL_DIR
        ),
}


with open(
    CONTROL_DIR
    /
    "control_reanalysis_manifest.json",
    "w"
) as f:

    json.dump(
        master_manifest,
        f,
        indent=2
    )


# ==========================================================================================
# 30. FINAL OUTPUT
# ==========================================================================================

print("\n\n" + "=" * 120)
print("REVIEWER-REQUESTED CONTROL REANALYSIS COMPLETE")
print("=" * 120)

print(
    f"""
All new files were saved under:

    {CONTROL_DIR}

Please paste back the following outputs:

1. EXPERIMENT 1 — REAL MODEL vs TRAINING-FOLD MEAN-GAZE BASELINE

2. CASE-SPECIFIC SHIFT ANALYSIS

3. EXPERIMENTS 2–3 — HELD-OUT CONTROL ALIGNMENT SUMMARY

4. PTB-XL — GAZE vs UNIFORM vs PERMUTED CONTROLS

5. PTB-XL CONTROL BOOTSTRAP — KEY COMPARISONS

6. LUDB — ORIGINAL GAZE MODELS + CONTROL MODELS

7. LUDB CONTROL BOOTSTRAP — PRIMARY TASKS

If there is an error before completion, paste the complete traceback and stop there.
Do not modify any parameters in response to intermediate results.
"""
)
