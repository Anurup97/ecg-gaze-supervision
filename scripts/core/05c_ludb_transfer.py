# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 12
# Stage: LUDB transfer from frozen representations
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 5C — LUDB EXTERNAL FROZEN-REPRESENTATION TRANSFER
#
# NO GAZE-MODEL TRAINING.
# NO ENDPOINT CHANGES.
# NO MODEL SELECTION.
#
# Models:
#   BASE
#   CONSULTANT
#   ALL_CLINICIANS
#
# Representation:
#   dense_1 (64 dimensions)
#
# External dataset:
#   LUDB
#
# Evaluation:
#   5-fold stratified out-of-fold linear probes.
#
# PRIMARY TASKS:
#   Conduction
#   LVH
#   Ischaemia_related
#
# EXPLORATORY:
#   AFib
#   Paced
#
# Preprocessing frozen in Step 5B:
#   500 -> 100 Hz using scipy.signal.resample_poly
#   no additional amplitude normalisation
#
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

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

from sklearn.model_selection import StratifiedKFold
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


# ======================================================================================
# 1. FIXED SETTINGS
# ======================================================================================

SEED = 2026
N_FOLDS = 5

FEATURE_LAYER = "dense_1"
FEATURE_DIM = 64

PRED_BATCH_SIZE = 64


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


PRIMARY_TASKS = [
    "Conduction",
    "LVH",
    "Ischaemia_related",
]


EXPLORATORY_TASKS = [
    "AFib",
    "Paced",
]


TASK_COLUMNS = {
    "Conduction":
        "target_Conduction",

    "LVH":
        "target_LVH",

    "Ischaemia_related":
        "target_Ischaemia_related",

    "AFib":
        "target_AFib",

    "Paced":
        "target_Paced",
}


EXPECTED_COUNTS = {

    "Conduction": {
        "N": 200,
        "positive": 66,
    },

    "LVH": {
        "N": 200,
        "positive": 108,
    },

    "Ischaemia_related": {
        "N": 188,
        "positive": 39,
    },

    "AFib": {
        "N": 200,
        "positive": 15,
    },

    "Paced": {
        "N": 200,
        "positive": 10,
    },
}


# ======================================================================================
# 2. PATHS
# ======================================================================================

ROOT = Path(
    "definitive_clean_gaze"
)


LUDB_ROOT = Path(
    "Test ECG"
)


LUDB_DATA = (
    LUDB_ROOT
    /
    "data"
)


FINAL_DIR = (
    ROOT
    /
    "final_models"
)


LUDB_DIR = (
    ROOT
    /
    "ludb_evaluation"
)


TARGET_PATH = (
    LUDB_DIR
    /
    "ludb_frozen_targets.csv"
)


STEP5B_MANIFEST_PATH = (
    LUDB_DIR
    /
    "ludb_step5b_manifest.json"
)


STEP1_MANIFEST_PATH = (
    ROOT
    /
    "step1_manifest.json"
)


FINAL_MODEL_MANIFEST_PATH = (
    FINAL_DIR
    /
    "final_model_manifest.json"
)


CONSULTANT_WEIGHTS = (
    FINAL_DIR
    /
    "clean_consultant.weights.h5"
)


ALL_WEIGHTS = (
    FINAL_DIR
    /
    "clean_all_clinicians.weights.h5"
)


for path in [
    TARGET_PATH,
    STEP5B_MANIFEST_PATH,
    STEP1_MANIFEST_PATH,
    FINAL_MODEL_MANIFEST_PATH,
    CONSULTANT_WEIGHTS,
    ALL_WEIGHTS,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ======================================================================================
# 3. HASH HELPER
# ======================================================================================

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


# ======================================================================================
# 4. VERIFY FROZEN MODEL FILES
# ======================================================================================

with open(
    STEP1_MANIFEST_PATH,
    "r"
) as f:

    step1_manifest = json.load(
        f
    )


with open(
    FINAL_MODEL_MANIFEST_PATH,
    "r"
) as f:

    final_model_manifest = json.load(
        f
    )


with open(
    STEP5B_MANIFEST_PATH,
    "r"
) as f:

    step5b_manifest = json.load(
        f
    )


BASE_MODEL_PATH = Path(
    step1_manifest[
        "base_model"
    ]
)


BASE_HASH = sha256_file(
    BASE_MODEL_PATH
)


CONSULTANT_HASH = sha256_file(
    CONSULTANT_WEIGHTS
)


ALL_HASH = sha256_file(
    ALL_WEIGHTS
)


print("=" * 100)
print("FROZEN MODEL VERIFICATION")
print("=" * 100)


print(
    "Base:",
    BASE_HASH
)


print(
    "Consultant:",
    CONSULTANT_HASH
)


print(
    "All clinicians:",
    ALL_HASH
)


if (
    BASE_HASH
    !=
    step1_manifest[
        "base_model_sha256"
    ]
):

    raise RuntimeError(
        "Base model hash mismatch."
    )


if (
    CONSULTANT_HASH
    !=
    final_model_manifest[
        "consultant_weights_sha256"
    ]
):

    raise RuntimeError(
        "Consultant weight hash mismatch."
    )


if (
    ALL_HASH
    !=
    final_model_manifest[
        "all_clinician_weights_sha256"
    ]
):

    raise RuntimeError(
        "All-clinician weight hash mismatch."
    )


print(
    "\n✓ Frozen model hashes verified."
)


# ======================================================================================
# 5. VERIFY STEP-5B FROZEN REPRESENTATION/PREPROCESSING
# ======================================================================================

if step5b_manifest[
    "feature_layer"
] != FEATURE_LAYER:

    raise RuntimeError(
        "Feature-layer definition changed."
    )


if step5b_manifest[
    "feature_dimension"
] != FEATURE_DIM:

    raise RuntimeError(
        "Feature dimension changed."
    )


if step5b_manifest[
    "native_sampling_frequency_Hz"
] != 500:

    raise RuntimeError(
        "Unexpected LUDB native sampling frequency."
    )


if step5b_manifest[
    "model_sampling_frequency_Hz"
] != 100:

    raise RuntimeError(
        "Unexpected model sampling frequency."
    )


print(
    "✓ Step-5B representation and preprocessing definition verified."
)


# ======================================================================================
# 6. LOAD FROZEN TARGETS
#
# IMPORTANT:
# We do NOT reconstruct clinical labels here.
# We use the target file frozen before model inference.
# ======================================================================================

targets = pd.read_csv(
    TARGET_PATH
)


if len(
    targets
) != 200:

    raise RuntimeError(
        f"Expected 200 LUDB target rows; found {len(targets)}."
    )


if targets[
    "ID"
].nunique() != 200:

    raise RuntimeError(
        "LUDB IDs are not unique."
    )


print("\n" + "=" * 100)
print("FROZEN LUDB TARGET AUDIT")
print("=" * 100)


target_audit_rows = []


for task_name, column in TASK_COLUMNS.items():

    valid = targets[
        column
    ].notna()


    y = targets.loc[
        valid,
        column
    ].astype(
        int
    )


    actual_n = int(
        len(
            y
        )
    )


    actual_positive = int(
        y.sum()
    )


    expected_n = EXPECTED_COUNTS[
        task_name
    ][
        "N"
    ]


    expected_positive = EXPECTED_COUNTS[
        task_name
    ][
        "positive"
    ]


    if (
        actual_n != expected_n
        or
        actual_positive != expected_positive
    ):

        raise RuntimeError(
            f"{task_name}: frozen-target count mismatch."
        )


    target_audit_rows.append(
        {
            "task":
                task_name,

            "role":
                (
                    "primary"
                    if task_name in PRIMARY_TASKS
                    else "exploratory"
                ),

            "N":
                actual_n,

            "positive":
                actual_positive,

            "negative":
                actual_n
                -
                actual_positive,

            "excluded":
                200
                -
                actual_n,
        }
    )


target_audit_df = pd.DataFrame(
    target_audit_rows
)


print(
    target_audit_df.to_string(
        index=False
    )
)


print(
    "\n✓ Frozen LUDB endpoints verified."
)


# ======================================================================================
# 7. FROZEN LUDB PREPROCESSING
# ======================================================================================

def preprocess_ludb_record(
    record_id
):

    path = (
        LUDB_DATA
        /
        str(
            int(
                record_id
            )
        )
    )


    signal, fields = wfdb.rdsamp(
        str(
            path
        )
    )


    if signal.shape != (
        5000,
        12
    ):

        raise RuntimeError(
            f"Record {record_id}: native shape {signal.shape}"
        )


    if float(
        fields[
            "fs"
        ]
    ) != 500.0:

        raise RuntimeError(
            f"Record {record_id}: unexpected fs."
        )


    leads = [
        str(
            lead
        ).upper()
        for lead
        in fields[
            "sig_name"
        ]
    ]


    if leads != EXPECTED_LEADS:

        raise RuntimeError(
            f"Record {record_id}: lead-order mismatch."
        )


    if not np.all(
        np.isfinite(
            signal
        )
    ):

        raise RuntimeError(
            f"Record {record_id}: non-finite raw signal."
        )


    downsampled = resample_poly(
        signal,
        up=1,
        down=5,
        axis=0
    )


    if downsampled.shape != (
        1000,
        12
    ):

        raise RuntimeError(
            f"Record {record_id}: bad resampled shape."
        )


    if not np.all(
        np.isfinite(
            downsampled
        )
    ):

        raise RuntimeError(
            f"Record {record_id}: non-finite resampled signal."
        )


    # NO amplitude normalisation.
    #
    # Frozen model convention:
    #     (lead, time, channel)

    X = (
        downsampled.T[
            :,
            :,
            None
        ]
    ).astype(
        np.float32
    )


    return X


# ======================================================================================
# 8. PREPROCESS ALL 200 LUDB ECGs
# ======================================================================================

print("\n" + "=" * 100)
print("PREPROCESSING ALL LUDB ECGs")
print("=" * 100)


record_ids = targets[
    "ID"
].astype(
    int
).to_numpy()


X_ludb = np.empty(
    (
        200,
        12,
        1000,
        1
    ),
    dtype=np.float32
)


for i, record_id in enumerate(
    record_ids
):

    X_ludb[
        i
    ] = preprocess_ludb_record(
        record_id
    )


    if (
        (i + 1) % 50 == 0
        or
        (i + 1) == 200
    ):

        print(
            f"Prepared {i + 1}/200"
        )


if not np.all(
    np.isfinite(
        X_ludb
    )
):

    raise RuntimeError(
        "Non-finite prepared LUDB array."
    )


print(
    "\nX_ludb:",
    X_ludb.shape
)


print(
    "Minimum:",
    float(
        X_ludb.min()
    )
)


print(
    "Maximum:",
    float(
        X_ludb.max()
    )
)


# Save the exact frozen preprocessed external dataset.
X_SAVE_PATH = (
    LUDB_DIR
    /
    "ludb_preprocessed_100hz.npy"
)


np.save(
    X_SAVE_PATH,
    X_ludb
)


X_HASH = sha256_file(
    X_SAVE_PATH
)


print(
    "\nPreprocessed LUDB SHA256:"
)

print(
    X_HASH
)


# ======================================================================================
# 9. MODEL LOADER
# ======================================================================================

def load_variant(
    weight_path=None
):

    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )


    if weight_path is not None:

        model.load_weights(
            weight_path
        )


    if model.output_shape != (
        None,
        23
    ):

        raise RuntimeError(
            f"Unexpected model output: {model.output_shape}"
        )


    feature_layer = model.get_layer(
        FEATURE_LAYER
    )


    if feature_layer.output.shape[-1] != FEATURE_DIM:

        raise RuntimeError(
            "Unexpected dense_1 feature dimension."
        )


    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=feature_layer.output
    )


    return (
        model,
        feature_model
    )


# ======================================================================================
# 10. LOAD THREE FROZEN MODELS
# ======================================================================================

base_model, base_feature_model = load_variant()


consultant_model, consultant_feature_model = load_variant(
    CONSULTANT_WEIGHTS
)


all_model, all_feature_model = load_variant(
    ALL_WEIGHTS
)


# ======================================================================================
# 11. EXTRACT REPRESENTATIONS
# ======================================================================================

print("\n" + "=" * 100)
print("EXTRACTING FROZEN LUDB REPRESENTATIONS")
print("=" * 100)


FEATURE_MODELS = {

    "BASE":
        base_feature_model,

    "CONSULTANT":
        consultant_feature_model,

    "ALL_CLINICIANS":
        all_feature_model,
}


features = {}


for model_name, feature_model in FEATURE_MODELS.items():

    print(
        "\n",
        model_name
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
        64
    ):

        raise RuntimeError(
            f"{model_name}: bad feature shape {Z.shape}"
        )


    if not np.all(
        np.isfinite(
            Z
        )
    ):

        raise RuntimeError(
            f"{model_name}: non-finite features."
        )


    features[
        model_name
    ] = Z


    np.save(
        LUDB_DIR
        /
        f"ludb_features_{model_name.lower()}.npy",

        Z
    )


    print(
        "shape:",
        Z.shape
    )


# ======================================================================================
# 12. REPRESENTATION CHANGE AUDIT
# ======================================================================================

feature_change_rows = []


for model_name in [
    "CONSULTANT",
    "ALL_CLINICIANS",
]:

    absolute_change = np.abs(
        features[
            model_name
        ]
        -
        features[
            "BASE"
        ]
    )


    feature_change_rows.append(
        {
            "model":
                model_name,

            "mean_absolute_feature_change":
                float(
                    absolute_change.mean()
                ),

            "median_absolute_feature_change":
                float(
                    np.median(
                        absolute_change
                    )
                ),

            "max_absolute_feature_change":
                float(
                    absolute_change.max()
                ),
        }
    )


feature_change_df = pd.DataFrame(
    feature_change_rows
)


print("\n" + "=" * 100)
print("LUDB FEATURE-CHANGE AUDIT")
print("=" * 100)


print(
    feature_change_df
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ======================================================================================
# 13. FREEZE PROBE CONFIGURATION
#
# StandardScaler is fitted ONLY on the training fold.
#
# This removes trivial feature-scale differences between model variants while
# preserving differences in linear separability.
# ======================================================================================

def make_probe():

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


# ======================================================================================
# 14. METRIC HELPER
# ======================================================================================

def binary_metrics(
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

        "TP":
            int(
                tp
            ),

        "TN":
            int(
                tn
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            ),
    }


# ======================================================================================
# 15. OOF EVALUATION
# ======================================================================================

summary_rows = []

prediction_rows = []

fold_rows = []


for task_name, target_column in TASK_COLUMNS.items():

    print(
        "\n\n"
        +
        "#" * 100
    )

    print(
        f"TASK — {task_name}"
    )

    print(
        "#" * 100
    )


    valid_mask = targets[
        target_column
    ].notna().to_numpy()


    valid_indices = np.where(
        valid_mask
    )[0]


    y = targets.loc[
        valid_mask,
        target_column
    ].astype(
        int
    ).to_numpy()


    ids = record_ids[
        valid_indices
    ]


    print(
        "Role:",
        (
            "PRIMARY"
            if task_name in PRIMARY_TASKS
            else "EXPLORATORY"
        )
    )


    print(
        "N:",
        len(
            y
        )
    )


    print(
        "Positive:",
        int(
            y.sum()
        )
    )


    print(
        "Negative:",
        int(
            len(
                y
            )
            -
            y.sum()
        )
    )


    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    )


    splits = list(
        splitter.split(
            np.zeros(
                len(
                    y
                )
            ),
            y
        )
    )


    # --------------------------------------------------------------
    # Save exact folds BEFORE model-specific probing.
    # --------------------------------------------------------------

    fold_assignment = np.zeros(
        len(
            y
        ),
        dtype=int
    )


    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        splits,
        start=1
    ):

        fold_assignment[
            test_idx
        ] = fold


        fold_rows.append(
            {
                "task":
                    task_name,

                "fold":
                    fold,

                "train_N":
                    int(
                        len(
                            train_idx
                        )
                    ),

                "test_N":
                    int(
                        len(
                            test_idx
                        )
                    ),

                "train_positive":
                    int(
                        y[
                            train_idx
                        ].sum()
                    ),

                "test_positive":
                    int(
                        y[
                            test_idx
                        ].sum()
                    ),
            }
        )


        print(
            f"Fold {fold}: "
            f"train={len(train_idx)} "
            f"(pos={int(y[train_idx].sum())}), "
            f"test={len(test_idx)} "
            f"(pos={int(y[test_idx].sum())})"
        )


    # --------------------------------------------------------------
    # OOF prediction for each representation.
    # --------------------------------------------------------------

    for model_name in [
        "BASE",
        "CONSULTANT",
        "ALL_CLINICIANS",
    ]:

        Z = features[
            model_name
        ][
            valid_indices
        ]


        probabilities = np.full(
            len(
                y
            ),
            np.nan,
            dtype=np.float64
        )


        for fold, (
            train_idx,
            test_idx
        ) in enumerate(
            splits,
            start=1
        ):

            probe = make_probe()


            with warnings.catch_warnings():

                warnings.simplefilter(
                    "error"
                )


                probe.fit(
                    Z[
                        train_idx
                    ],
                    y[
                        train_idx
                    ]
                )


            probabilities[
                test_idx
            ] = probe.predict_proba(
                Z[
                    test_idx
                ]
            )[
                :,
                1
            ]


        if not np.all(
            np.isfinite(
                probabilities
            )
        ):

            raise RuntimeError(
                f"{task_name}/{model_name}: incomplete OOF predictions."
            )


        metrics = binary_metrics(
            y,
            probabilities
        )


        summary_rows.append(
            {
                "task":
                    task_name,

                "role":
                    (
                        "primary"
                        if task_name in PRIMARY_TASKS
                        else "exploratory"
                    ),

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

                "negative":
                    int(
                        len(
                            y
                        )
                        -
                        y.sum()
                    ),

                **metrics,
            }
        )


        predicted_labels = (
            probabilities
            >=
            0.5
        ).astype(
            int
        )


        for i in range(
            len(
                y
            )
        ):

            prediction_rows.append(
                {
                    "task":
                        task_name,

                    "role":
                        (
                            "primary"
                            if task_name in PRIMARY_TASKS
                            else "exploratory"
                        ),

                    "model":
                        model_name,

                    "record_id":
                        int(
                            ids[
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
                            probabilities[
                                i
                            ]
                        ),

                    "predicted_label":
                        int(
                            predicted_labels[
                                i
                            ]
                        ),
                }
            )


# ======================================================================================
# 16. TABLES
# ======================================================================================

summary_df = pd.DataFrame(
    summary_rows
)


predictions_df = pd.DataFrame(
    prediction_rows
)


fold_df = pd.DataFrame(
    fold_rows
)


# ======================================================================================
# 17. PAIRED POINT DIFFERENCES VS BASE
# ======================================================================================

comparison_rows = []


for task_name in TASK_COLUMNS:

    task_summary = summary_df[
        summary_df[
            "task"
        ]
        ==
        task_name
    ].set_index(
        "model"
    )


    base = task_summary.loc[
        "BASE"
    ]


    for model_name in [
        "CONSULTANT",
        "ALL_CLINICIANS",
    ]:

        row = task_summary.loc[
            model_name
        ]


        comparison_rows.append(
            {
                "task":
                    task_name,

                "role":
                    row[
                        "role"
                    ],

                "comparison":
                    (
                        model_name
                        +
                        "_vs_BASE"
                    ),

                "delta_AUROC":
                    float(
                        row[
                            "AUROC"
                        ]
                        -
                        base[
                            "AUROC"
                        ]
                    ),

                "delta_AP":
                    float(
                        row[
                            "AP"
                        ]
                        -
                        base[
                            "AP"
                        ]
                    ),

                "delta_balanced_accuracy":
                    float(
                        row[
                            "balanced_accuracy"
                        ]
                        -
                        base[
                            "balanced_accuracy"
                        ]
                    ),
            }
        )


comparison_df = pd.DataFrame(
    comparison_rows
)


# ======================================================================================
# 18. SAVE
# ======================================================================================

summary_df.to_csv(
    LUDB_DIR
    /
    "ludb_step5c_oof_summary.csv",
    index=False
)


predictions_df.to_csv(
    LUDB_DIR
    /
    "ludb_step5c_oof_predictions.csv",
    index=False
)


comparison_df.to_csv(
    LUDB_DIR
    /
    "ludb_step5c_point_differences.csv",
    index=False
)


fold_df.to_csv(
    LUDB_DIR
    /
    "ludb_step5c_fold_manifest.csv",
    index=False
)


feature_change_df.to_csv(
    LUDB_DIR
    /
    "ludb_step5c_feature_change_audit.csv",
    index=False
)


# ======================================================================================
# 19. PRINT PRIMARY RESULTS
# ======================================================================================

print(
    "\n\n"
    +
    "=" * 110
)

print(
    "LUDB PRIMARY EXTERNAL REPRESENTATION-TRANSFER RESULTS"
)

print(
    "=" * 110
)


primary_summary = summary_df[
    summary_df[
        "role"
    ]
    ==
    "primary"
]


print(
    primary_summary[
        [
            "task",
            "model",
            "N",
            "positive",
            "AUROC",
            "AP",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\nPRIMARY POINT DIFFERENCES VS BASE"
)


print(
    comparison_df[
        comparison_df[
            "role"
        ]
        ==
        "primary"
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ======================================================================================
# 20. PRINT EXPLORATORY RESULTS
# ======================================================================================

print(
    "\n\n"
    +
    "=" * 110
)

print(
    "LUDB EXPLORATORY REPRESENTATION-TRANSFER RESULTS"
)

print(
    "=" * 110
)


exploratory_summary = summary_df[
    summary_df[
        "role"
    ]
    ==
    "exploratory"
]


print(
    exploratory_summary[
        [
            "task",
            "model",
            "N",
            "positive",
            "AUROC",
            "AP",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\nEXPLORATORY POINT DIFFERENCES VS BASE"
)


print(
    comparison_df[
        comparison_df[
            "role"
        ]
        ==
        "exploratory"
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ======================================================================================
# 21. EVALUATION MANIFEST
# ======================================================================================

manifest = {

    "stage":
        "Step 5C LUDB frozen external representation-transfer evaluation",

    "interpretation":
        (
            "External linear-probe evaluation of frozen ECG representations; "
            "NOT direct validation of the PTB-XL 23-class diagnostic head."
        ),

    "LUDB_records":
        200,

    "preprocessed_LUDB_sha256":
        X_HASH,

    "native_sampling_frequency_Hz":
        500,

    "model_sampling_frequency_Hz":
        100,

    "resampling":
        "scipy.signal.resample_poly(up=1, down=5, axis=time)",

    "additional_amplitude_normalisation":
        False,

    "feature_layer":
        FEATURE_LAYER,

    "feature_dimension":
        FEATURE_DIM,

    "primary_tasks":
        PRIMARY_TASKS,

    "exploratory_tasks":
        EXPLORATORY_TASKS,

    "ischaemia_endpoint":
        (
            "Positive: definite 'Ischemia:', 'Scar formation:' or 'STEMI:' "
            "annotation. Negative: empty Ischemia field. "
            "Undefined ischemia/scar/supp.NSTEMI-only records excluded."
        ),

    "ischaemia_N":
        188,

    "ischaemia_positive":
        39,

    "cross_validation":
        "5-fold StratifiedKFold out-of-fold",

    "CV_seed":
        SEED,

    "probe":
        (
            "Training-fold StandardScaler followed by logistic regression "
            "(C=1.0, lbfgs, class_weight=balanced)."
        ),

    "threshold_for_secondary_classification_metrics":
        0.5,

    "primary_transfer_metrics":
        [
            "AUROC",
            "Average Precision",
        ],

    "gaze_model_training_performed":
        False,

    "endpoint_tuning_after_model_inference":
        False,

    "base_model_sha256":
        BASE_HASH,

    "consultant_weights_sha256":
        CONSULTANT_HASH,

    "all_clinician_weights_sha256":
        ALL_HASH,
}


with open(
    LUDB_DIR
    /
    "ludb_step5c_manifest.json",
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ======================================================================================
# 22. FINISH
# ======================================================================================

print(
    "\n\n"
    +
    "=" * 100
)

print(
    "STEP 5C COMPLETE"
)

print(
    "=" * 100
)


print(
    """
Please paste back:

1. LUDB FEATURE-CHANGE AUDIT

2. LUDB PRIMARY EXTERNAL REPRESENTATION-TRANSFER RESULTS

3. PRIMARY POINT DIFFERENCES VS BASE

4. LUDB EXPLORATORY REPRESENTATION-TRANSFER RESULTS

5. EXPLORATORY POINT DIFFERENCES VS BASE

Do NOT change the models, tasks, folds, feature layer or preprocessing
after seeing these results.

Afterwards Step 5D will compute paired bootstrap confidence intervals
using ONLY the saved OOF predictions.
"""
)
