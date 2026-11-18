# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 6
# Stage: PTB-XL fold-10 predictions
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 4B — PTB-XL FOLD-10 DIAGNOSTIC EVALUATION
#
# INFERENCE ONLY.
# NO TRAINING.
# NO THRESHOLD TUNING.
#
# Models:
#   1. Base pretrained model
#   2. Final consultant-gaze model
#   3. Final all-clinician-gaze model
#
# Test set:
#   PTB-XL strat_fold == 10
#   N = 2198
#
# Metrics:
#   - per-class AUROC
#   - per-class Average Precision (AP)
#   - macro AUROC
#   - macro AP
#
# All 2198 records are retained, including records with zero positive
# diagnostic-subclass labels.
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import ast
import gc
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ======================================================================================
# 1. FROZEN DEFINITIONS
# ======================================================================================

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


EXPECTED_TEST_N = 2198

PRED_BATCH_SIZE = 64


# ======================================================================================
# 2. PATHS
# ======================================================================================

PTBXL_ROOT = Path(
    "ptb-xl"
)


DATABASE_PATH = (
    PTBXL_ROOT
    /
    "ptbxl_database.csv"
)


SCP_PATH = (
    PTBXL_ROOT
    /
    "scp_statements.csv"
)


ROOT = Path(
    "definitive_clean_gaze"
)


STEP1_MANIFEST_PATH = (
    ROOT
    /
    "step1_manifest.json"
)


FINAL_MANIFEST_PATH = (
    ROOT
    /
    "final_models"
    /
    "final_model_manifest.json"
)


CONSULTANT_WEIGHTS = (
    ROOT
    /
    "final_models"
    /
    "clean_consultant.weights.h5"
)


ALL_CLINICIAN_WEIGHTS = (
    ROOT
    /
    "final_models"
    /
    "clean_all_clinicians.weights.h5"
)


OUT_DIR = (
    ROOT
    /
    "ptbxl_evaluation"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for path in [
    DATABASE_PATH,
    SCP_PATH,
    STEP1_MANIFEST_PATH,
    FINAL_MANIFEST_PATH,
    CONSULTANT_WEIGHTS,
    ALL_CLINICIAN_WEIGHTS,
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
# 4. VERIFY FROZEN MODELS
# ======================================================================================

with open(
    STEP1_MANIFEST_PATH,
    "r"
) as f:

    step1_manifest = json.load(
        f
    )


with open(
    FINAL_MANIFEST_PATH,
    "r"
) as f:

    final_manifest = json.load(
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
    ALL_CLINICIAN_WEIGHTS
)


print("=" * 100)
print("FROZEN MODEL VERIFICATION")
print("=" * 100)


print(
    "Base:"
)

print(
    BASE_HASH
)


print(
    "\nConsultant:"
)

print(
    CONSULTANT_HASH
)


print(
    "\nAll clinicians:"
)

print(
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
    final_manifest[
        "consultant_weights_sha256"
    ]
):

    raise RuntimeError(
        "Consultant model hash mismatch."
    )


if (
    ALL_HASH
    !=
    final_manifest[
        "all_clinician_weights_sha256"
    ]
):

    raise RuntimeError(
        "All-clinician model hash mismatch."
    )


print(
    "\n✓ All three diagnostic models are frozen and verified."
)


# ======================================================================================
# 5. LOAD PTB-XL METADATA
# ======================================================================================

metadata = pd.read_csv(
    DATABASE_PATH,
    index_col="ecg_id"
)


scp = pd.read_csv(
    SCP_PATH,
    index_col=0
)


metadata[
    "scp_codes"
] = metadata[
    "scp_codes"
].apply(
    ast.literal_eval
)


test_metadata = metadata[
    metadata[
        "strat_fold"
    ]
    ==
    10
].copy()


if len(
    test_metadata
) != EXPECTED_TEST_N:

    raise RuntimeError(
        f"Expected {EXPECTED_TEST_N} fold-10 ECGs, "
        f"found {len(test_metadata)}."
    )


print("\n" + "=" * 100)
print("PTB-XL TEST SET")
print("=" * 100)


print(
    "Fold-10 ECGs:",
    len(
        test_metadata
    )
)


print(
    "Unique patients:",
    test_metadata[
        "patient_id"
    ].nunique()
)


# ======================================================================================
# 6. RECONSTRUCT EXACT 23-CLASS TARGETS
# ======================================================================================

diagnostic_scp = scp[
    scp[
        "diagnostic"
    ]
    ==
    1
].copy()


def aggregate_subclasses(
    code_dictionary
):

    output = set()


    for code in code_dictionary.keys():

        if code not in diagnostic_scp.index:
            continue


        subclass = diagnostic_scp.loc[
            code,
            "diagnostic_subclass"
        ]


        if pd.isna(
            subclass
        ):
            continue


        output.add(
            str(
                subclass
            )
        )


    return output


test_labels = test_metadata[
    "scp_codes"
].apply(
    aggregate_subclasses
)


class_to_index = {
    class_name: i
    for i, class_name in enumerate(
        MODEL_CLASSES
    )
}


Y_test = np.zeros(
    (
        len(
            test_metadata
        ),
        len(
            MODEL_CLASSES
        )
    ),
    dtype=np.int8
)


for row_idx, labels in enumerate(
    test_labels
):

    for label in labels:

        if label not in class_to_index:

            raise RuntimeError(
                f"Unexpected diagnostic subclass: {label}"
            )


        Y_test[
            row_idx,
            class_to_index[
                label
            ]
        ] = 1


if Y_test.shape != (
    2198,
    23
):

    raise RuntimeError(
        f"Unexpected Y_test shape: {Y_test.shape}"
    )


class_positive_counts = Y_test.sum(
    axis=0
)


if np.any(
    class_positive_counts == 0
):

    raise RuntimeError(
        "At least one diagnostic class has zero positives."
    )


if np.any(
    class_positive_counts == len(
        Y_test
    )
):

    raise RuntimeError(
        "At least one diagnostic class has zero negatives."
    )


all_zero_rows = (
    Y_test.sum(
        axis=1
    )
    ==
    0
)


print(
    "Target shape:",
    Y_test.shape
)


print(
    "All-zero 23-class target rows:",
    int(
        all_zero_rows.sum()
    )
)


if int(
    all_zero_rows.sum()
) != 40:

    raise RuntimeError(
        "Expected exactly 40 all-zero fold-10 targets."
    )


print(
    "\n✓ Exact fold-10 targets reconstructed."
)


# ======================================================================================
# 7. LOAD ALL 2198 RAW ECGs
#
# Source waveform:
#     (1000, 12)
#
# Model input:
#     (12, 1000, 1)
# ======================================================================================

print("\n" + "=" * 100)
print("LOADING PTB-XL FOLD-10 WAVEFORMS")
print("=" * 100)


X_test = np.empty(
    (
        EXPECTED_TEST_N,
        12,
        1000,
        1
    ),
    dtype=np.float32
)


for i, (_, row) in enumerate(
    test_metadata.iterrows()
):

    record_path = (
        PTBXL_ROOT
        /
        str(
            row[
                "filename_lr"
            ]
        )
    )


    signal, signal_metadata = wfdb.rdsamp(
        str(
            record_path
        )
    )


    if signal.shape != (
        1000,
        12
    ):

        raise RuntimeError(
            f"Bad signal shape at row {i}: {signal.shape}"
        )


    leads = [
        str(x).upper()
        for x in signal_metadata[
            "sig_name"
        ]
    ]


    if leads != EXPECTED_LEADS:

        raise RuntimeError(
            f"Lead-order mismatch at row {i}: {leads}"
        )


    if float(
        signal_metadata[
            "fs"
        ]
    ) != 100.0:

        raise RuntimeError(
            f"Sampling-rate mismatch at row {i}."
        )


    if not np.all(
        np.isfinite(
            signal
        )
    ):

        raise RuntimeError(
            f"Non-finite ECG values at row {i}."
        )


    X_test[
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
        (i + 1) == EXPECTED_TEST_N
    ):

        print(
            f"Loaded {i + 1}/{EXPECTED_TEST_N}"
        )


print(
    "\nFinal X_test shape:",
    X_test.shape
)


print(
    "Finite:",
    np.all(
        np.isfinite(
            X_test
        )
    )
)


print(
    "Global minimum:",
    float(
        X_test.min()
    )
)


print(
    "Global maximum:",
    float(
        X_test.max()
    )
)


print(
    "\n✓ All fold-10 waveforms loaded."
)


# ======================================================================================
# 8. LOAD MODEL VARIANTS
# ======================================================================================

def load_base_model():

    tf.keras.backend.clear_session()


    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )


    if model.output_shape != (
        None,
        23
    ):

        raise RuntimeError(
            f"Unexpected output shape: {model.output_shape}"
        )


    final_activation = (
        model.layers[
            -1
        ]
        .activation
        .__name__
    )


    if final_activation != "sigmoid":

        raise RuntimeError(
            f"Expected sigmoid output, found {final_activation}."
        )


    return model


# ======================================================================================
# 9. WEIGHT-DIFFERENCE AUDIT AFTER LOADING SAVED FINAL WEIGHTS
# ======================================================================================

def model_weight_snapshot(
    model
):

    return {
        layer.name: [
            np.array(
                weight,
                copy=True
            )
            for weight in layer.get_weights()
        ]
        for layer in model.layers
    }


def compare_to_snapshot(
    base_snapshot,
    model
):

    rows = []


    for layer in model.layers:

        old_weights = base_snapshot[
            layer.name
        ]


        new_weights = layer.get_weights()


        max_change = 0.0


        for old, new in zip(
            old_weights,
            new_weights
        ):

            if old.shape != new.shape:

                raise RuntimeError(
                    f"Weight shape mismatch: {layer.name}"
                )


            if old.size:

                max_change = max(
                    max_change,
                    float(
                        np.max(
                            np.abs(
                                old - new
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


    return pd.DataFrame(
        rows
    )


# Base reference
base_model = load_base_model()


base_snapshot = model_weight_snapshot(
    base_model
)


# Consultant
consultant_model = load_base_model()


consultant_model.load_weights(
    CONSULTANT_WEIGHTS
)


consultant_weight_audit = compare_to_snapshot(
    base_snapshot,
    consultant_model
)


# All clinicians
all_model = load_base_model()


all_model.load_weights(
    ALL_CLINICIAN_WEIGHTS
)


all_weight_audit = compare_to_snapshot(
    base_snapshot,
    all_model
)


for name, audit in [
    (
        "consultant",
        consultant_weight_audit
    ),
    (
        "all_clinicians",
        all_weight_audit
    ),
]:

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
            f"{name}: unexpected changed layers {changed}"
        )


print("\n" + "=" * 100)
print("DOWNSTREAM MODEL WEIGHT AUDIT")
print("=" * 100)


print(
    "Consultant vs base changed layers:"
)

print(
    consultant_weight_audit.loc[
        consultant_weight_audit[
            "max_abs_change"
        ]
        >
        1e-12
    ].to_string(
        index=False
    )
)


print(
    "\nAll-clinician vs base changed layers:"
)

print(
    all_weight_audit.loc[
        all_weight_audit[
            "max_abs_change"
        ]
        >
        1e-12
    ].to_string(
        index=False
    )
)


print(
    "\n✓ Saved final gaze models differ from base ONLY at conv2d_10."
)


# ======================================================================================
# 10. INFERENCE
# ======================================================================================

def run_prediction(
    model,
    X,
    name
):

    print(
        f"\nRunning inference: {name}"
    )


    prediction = model.predict(
        X,
        batch_size=PRED_BATCH_SIZE,
        verbose=1
    )


    prediction = np.asarray(
        prediction,
        dtype=np.float32
    )


    if prediction.shape != (
        EXPECTED_TEST_N,
        23
    ):

        raise RuntimeError(
            f"{name}: bad prediction shape {prediction.shape}"
        )


    if not np.all(
        np.isfinite(
            prediction
        )
    ):

        raise RuntimeError(
            f"{name}: non-finite predictions."
        )


    if np.any(
        prediction < 0
    ) or np.any(
        prediction > 1
    ):

        raise RuntimeError(
            f"{name}: sigmoid predictions outside [0,1]."
        )


    print(
        f"{name}: prediction range "
        f"{prediction.min():.6f} to {prediction.max():.6f}"
    )


    return prediction


print("\n" + "=" * 100)
print("PTB-XL FOLD-10 MODEL INFERENCE")
print("=" * 100)


P_base = run_prediction(
    base_model,
    X_test,
    "BASE"
)


P_consultant = run_prediction(
    consultant_model,
    X_test,
    "CONSULTANT-GAZE"
)


P_all = run_prediction(
    all_model,
    X_test,
    "ALL-CLINICIAN-GAZE"
)


# ======================================================================================
# 11. VERIFY THAT GAZE SUPERVISION ACTUALLY CHANGED DIAGNOSTIC OUTPUTS
# ======================================================================================

consultant_output_change = np.abs(
    P_consultant
    -
    P_base
)


all_output_change = np.abs(
    P_all
    -
    P_base
)


print("\n" + "=" * 100)
print("PREDICTION CHANGE AUDIT")
print("=" * 100)


print(
    "Consultant vs base:"
)

print(
    "  mean absolute probability change:",
    float(
        consultant_output_change.mean()
    )
)

print(
    "  max absolute probability change:",
    float(
        consultant_output_change.max()
    )
)


print(
    "\nAll clinicians vs base:"
)

print(
    "  mean absolute probability change:",
    float(
        all_output_change.mean()
    )
)

print(
    "  max absolute probability change:",
    float(
        all_output_change.max()
    )
)


# ======================================================================================
# 12. PER-CLASS METRICS
# ======================================================================================

def evaluate_predictions(
    model_name,
    predictions
):

    rows = []


    for class_idx, class_name in enumerate(
        MODEL_CLASSES
    ):

        y_true = Y_test[
            :,
            class_idx
        ]


        y_score = predictions[
            :,
            class_idx
        ]


        auroc = roc_auc_score(
            y_true,
            y_score
        )


        ap = average_precision_score(
            y_true,
            y_score
        )


        rows.append(
            {
                "model":
                    model_name,

                "class":
                    class_name,

                "positive_n":
                    int(
                        y_true.sum()
                    ),

                "negative_n":
                    int(
                        len(
                            y_true
                        )
                        -
                        y_true.sum()
                    ),

                "AUROC":
                    float(
                        auroc
                    ),

                "AP":
                    float(
                        ap
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


metrics_base = evaluate_predictions(
    "base",
    P_base
)


metrics_consultant = evaluate_predictions(
    "consultant",
    P_consultant
)


metrics_all = evaluate_predictions(
    "all_clinicians",
    P_all
)


all_metrics = pd.concat(
    [
        metrics_base,
        metrics_consultant,
        metrics_all
    ],
    ignore_index=True
)


# ======================================================================================
# 13. PAIRED PER-CLASS COMPARISON TABLE
# ======================================================================================

comparison = (
    metrics_base[
        [
            "class",
            "positive_n",
            "negative_n",
            "AUROC",
            "AP"
        ]
    ]
    .rename(
        columns={
            "AUROC":
                "base_AUROC",

            "AP":
                "base_AP",
        }
    )
)


comparison = comparison.merge(

    metrics_consultant[
        [
            "class",
            "AUROC",
            "AP"
        ]
    ].rename(
        columns={
            "AUROC":
                "consultant_AUROC",

            "AP":
                "consultant_AP",
        }
    ),

    on="class",
    how="left"
)


comparison = comparison.merge(

    metrics_all[
        [
            "class",
            "AUROC",
            "AP"
        ]
    ].rename(
        columns={
            "AUROC":
                "all_clinicians_AUROC",

            "AP":
                "all_clinicians_AP",
        }
    ),

    on="class",
    how="left"
)


comparison[
    "consultant_delta_AUROC"
] = (
    comparison[
        "consultant_AUROC"
    ]
    -
    comparison[
        "base_AUROC"
    ]
)


comparison[
    "all_clinicians_delta_AUROC"
] = (
    comparison[
        "all_clinicians_AUROC"
    ]
    -
    comparison[
        "base_AUROC"
    ]
)


comparison[
    "consultant_delta_AP"
] = (
    comparison[
        "consultant_AP"
    ]
    -
    comparison[
        "base_AP"
    ]
)


comparison[
    "all_clinicians_delta_AP"
] = (
    comparison[
        "all_clinicians_AP"
    ]
    -
    comparison[
        "base_AP"
    ]
)


# Preserve frozen class order.
comparison[
    "_order"
] = comparison[
    "class"
].map(
    {
        class_name: i
        for i, class_name in enumerate(
            MODEL_CLASSES
        )
    }
)


comparison = (
    comparison
    .sort_values(
        "_order"
    )
    .drop(
        columns="_order"
    )
    .reset_index(
        drop=True
    )
)


# ======================================================================================
# 14. MACRO SUMMARY
# ======================================================================================

macro_rows = []


for name, predictions, model_metrics in [

    (
        "base",
        P_base,
        metrics_base
    ),

    (
        "consultant",
        P_consultant,
        metrics_consultant
    ),

    (
        "all_clinicians",
        P_all,
        metrics_all
    ),
]:

    macro_auroc = float(
        model_metrics[
            "AUROC"
        ].mean()
    )


    macro_ap = float(
        model_metrics[
            "AP"
        ].mean()
    )


    micro_auroc = float(
        roc_auc_score(
            Y_test.ravel(),
            predictions.ravel()
        )
    )


    micro_ap = float(
        average_precision_score(
            Y_test.ravel(),
            predictions.ravel()
        )
    )


    macro_rows.append(
        {
            "model":
                name,

            "macro_AUROC":
                macro_auroc,

            "macro_AP":
                macro_ap,

            "micro_AUROC":
                micro_auroc,

            "micro_AP":
                micro_ap,
        }
    )


macro_summary = pd.DataFrame(
    macro_rows
)


base_macro_auroc = float(
    macro_summary.loc[
        macro_summary[
            "model"
        ]
        ==
        "base",
        "macro_AUROC"
    ].iloc[
        0
    ]
)


base_macro_ap = float(
    macro_summary.loc[
        macro_summary[
            "model"
        ]
        ==
        "base",
        "macro_AP"
    ].iloc[
        0
    ]
)


macro_summary[
    "delta_macro_AUROC_vs_base"
] = (
    macro_summary[
        "macro_AUROC"
    ]
    -
    base_macro_auroc
)


macro_summary[
    "delta_macro_AP_vs_base"
] = (
    macro_summary[
        "macro_AP"
    ]
    -
    base_macro_ap
)


# ======================================================================================
# 15. COUNT PER-CLASS DIRECTIONS
# ======================================================================================

direction_summary = pd.DataFrame(
    [
        {
            "model":
                "consultant",

            "classes_AUROC_improved":
                int(
                    (
                        comparison[
                            "consultant_delta_AUROC"
                        ]
                        >
                        0
                    ).sum()
                ),

            "classes_AUROC_worsened":
                int(
                    (
                        comparison[
                            "consultant_delta_AUROC"
                        ]
                        <
                        0
                    ).sum()
                ),

            "classes_AP_improved":
                int(
                    (
                        comparison[
                            "consultant_delta_AP"
                        ]
                        >
                        0
                    ).sum()
                ),

            "classes_AP_worsened":
                int(
                    (
                        comparison[
                            "consultant_delta_AP"
                        ]
                        <
                        0
                    ).sum()
                ),
        },

        {
            "model":
                "all_clinicians",

            "classes_AUROC_improved":
                int(
                    (
                        comparison[
                            "all_clinicians_delta_AUROC"
                        ]
                        >
                        0
                    ).sum()
                ),

            "classes_AUROC_worsened":
                int(
                    (
                        comparison[
                            "all_clinicians_delta_AUROC"
                        ]
                        <
                        0
                    ).sum()
                ),

            "classes_AP_improved":
                int(
                    (
                        comparison[
                            "all_clinicians_delta_AP"
                        ]
                        >
                        0
                    ).sum()
                ),

            "classes_AP_worsened":
                int(
                    (
                        comparison[
                            "all_clinicians_delta_AP"
                        ]
                        <
                        0
                    ).sum()
                ),
        },
    ]
)


# ======================================================================================
# 16. SAVE PREDICTIONS
# ======================================================================================

prediction_output = pd.DataFrame(
    {
        "ecg_id":
            test_metadata.index.to_numpy(),

        "patient_id":
            test_metadata[
                "patient_id"
            ].to_numpy(),

        "all_zero_target":
            all_zero_rows.astype(
                int
            ),
    }
)


# Save each true label and each model probability.
for class_idx, class_name in enumerate(
    MODEL_CLASSES
):

    prediction_output[
        f"true__{class_name}"
    ] = Y_test[
        :,
        class_idx
    ]


    prediction_output[
        f"base__{class_name}"
    ] = P_base[
        :,
        class_idx
    ]


    prediction_output[
        f"consultant__{class_name}"
    ] = P_consultant[
        :,
        class_idx
    ]


    prediction_output[
        f"all_clinicians__{class_name}"
    ] = P_all[
        :,
        class_idx
    ]


prediction_output.to_csv(
    OUT_DIR
    /
    "ptbxl_fold10_predictions.csv",
    index=False
)


all_metrics.to_csv(
    OUT_DIR
    /
    "ptbxl_per_class_metrics_long.csv",
    index=False
)


comparison.to_csv(
    OUT_DIR
    /
    "ptbxl_per_class_comparison.csv",
    index=False
)


macro_summary.to_csv(
    OUT_DIR
    /
    "ptbxl_macro_summary.csv",
    index=False
)


direction_summary.to_csv(
    OUT_DIR
    /
    "ptbxl_direction_summary.csv",
    index=False
)


consultant_weight_audit.to_csv(
    OUT_DIR
    /
    "consultant_downstream_weight_audit.csv",
    index=False
)


all_weight_audit.to_csv(
    OUT_DIR
    /
    "all_clinicians_downstream_weight_audit.csv",
    index=False
)


# ======================================================================================
# 17. PRINT MACRO RESULTS
# ======================================================================================

print("\n\n" + "=" * 100)
print("PTB-XL FOLD-10 MACRO DIAGNOSTIC PERFORMANCE")
print("=" * 100)


print(
    macro_summary
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ======================================================================================
# 18. PRINT PER-CLASS RESULTS
# ======================================================================================

print("\n\n" + "=" * 100)
print("PTB-XL PER-CLASS PERFORMANCE")
print("=" * 100)


print(
    comparison[
        [
            "class",
            "positive_n",

            "base_AUROC",
            "consultant_AUROC",
            "consultant_delta_AUROC",
            "all_clinicians_AUROC",
            "all_clinicians_delta_AUROC",

            "base_AP",
            "consultant_AP",
            "consultant_delta_AP",
            "all_clinicians_AP",
            "all_clinicians_delta_AP",
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
# 19. PRINT DIRECTION SUMMARY
# ======================================================================================

print("\n\n" + "=" * 100)
print("NUMBER OF CLASSES IMPROVED / WORSENED")
print("=" * 100)


print(
    direction_summary.to_string(
        index=False
    )
)


# ======================================================================================
# 20. MANIFEST
# ======================================================================================

evaluation_manifest = {

    "stage":
        "Step 4B PTB-XL fold-10 frozen diagnostic evaluation",

    "test_partition":
        "PTB-XL strat_fold 10",

    "test_records":
        int(
            len(
                Y_test
            )
        ),

    "all_zero_target_records_retained":
        int(
            all_zero_rows.sum()
        ),

    "classes":
        MODEL_CLASSES,

    "n_classes":
        23,

    "input_sampling_frequency_Hz":
        100,

    "input_shape":
        [
            12,
            1000,
            1
        ],

    "base_model_sha256":
        BASE_HASH,

    "consultant_weights_sha256":
        CONSULTANT_HASH,

    "all_clinician_weights_sha256":
        ALL_HASH,

    "metrics":
        [
            "per-class AUROC",
            "per-class Average Precision",
            "macro AUROC",
            "macro Average Precision",
            "micro AUROC",
            "micro Average Precision",
        ],

    "threshold_tuning_performed":
        False,

    "training_performed":
        False,

    "records_excluded":
        0,

    "primary_comparison":
        "frozen gaze models versus unchanged base model on identical fold-10 records",
}


with open(
    OUT_DIR
    /
    "ptbxl_evaluation_manifest.json",
    "w"
) as f:

    json.dump(
        evaluation_manifest,
        f,
        indent=2
    )


print("\n\n" + "=" * 100)
print("STEP 4B COMPLETE — INFERENCE ONLY")
print("=" * 100)


print(
    """
Please paste back:

1. DOWNSTREAM MODEL WEIGHT AUDIT

2. PREDICTION CHANGE AUDIT

3. PTB-XL FOLD-10 MACRO DIAGNOSTIC PERFORMANCE

4. NUMBER OF CLASSES IMPROVED / WORSENED

5. PTB-XL PER-CLASS PERFORMANCE

Do not retrain or adjust either gaze model based on these results.

After we see the result, we will run the paired bootstrap confidence-interval
analysis from the SAVED predictions rather than touching the models again.
"""
)
