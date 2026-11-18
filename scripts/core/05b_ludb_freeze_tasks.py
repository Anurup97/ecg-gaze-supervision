# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 10
# Stage: freeze LUDB tasks/representations
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 5B — FREEZE LUDB TASKS, FEATURE LAYER AND PREPROCESSING
#
# NO LUDB MODEL INFERENCE
# NO CLASSIFIER TRAINING
#
# Purpose:
#   1. Define LUDB downstream tasks BEFORE seeing model performance.
#   2. Verify their exact counts.
#   3. Freeze the representation layer.
#   4. Freeze 500 Hz -> 100 Hz preprocessing.
#
# LUDB will be treated as:
#   EXTERNAL REPRESENTATION-TRANSFER / LINEAR-PROBE EVALUATION
#
# It is NOT direct validation of the frozen 23-class classifier.
# ======================================================================================

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.signal import resample_poly
import wfdb


# ======================================================================================
# 1. PATHS
# ======================================================================================

LUDB_ROOT = Path("Test ECG")
LUDB_CSV = LUDB_ROOT / "ludb.csv"
DATA_DIR = LUDB_ROOT / "data"

ROOT = Path("definitive_clean_gaze")

STEP1_MANIFEST = (
    ROOT
    /
    "step1_manifest.json"
)

OUT_DIR = (
    ROOT
    /
    "ludb_evaluation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================================
# 2. LOAD METADATA
# ======================================================================================

ludb = pd.read_csv(
    LUDB_CSV
)


if len(ludb) != 200:
    raise RuntimeError(
        f"Expected 200 LUDB records, found {len(ludb)}."
    )


if ludb["ID"].nunique() != 200:
    raise RuntimeError(
        "LUDB ID is not unique."
    )


print("=" * 100)
print("LUDB RECORD AUDIT")
print("=" * 100)

print(
    "Records:",
    len(ludb)
)

print(
    "Unique IDs:",
    ludb["ID"].nunique()
)


# ======================================================================================
# 3. STRING HELPERS
# ======================================================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace("\\n", "\n")
        .strip()
        .lower()
    )


def split_lines(value):

    text = clean_text(value)

    if not text:
        return []

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


# ======================================================================================
# 4. TASK 1 — ATRIAL FIBRILLATION
#
# Exploratory because prevalence is low.
# Includes:
#     Atrial fibrillation
#     Atrial fibrillation + aberrant conduction
# ======================================================================================

rhythm_text = ludb[
    "Rhythms"
].apply(
    clean_text
)


ludb[
    "target_AFib"
] = rhythm_text.str.contains(
    "atrial fibrillation",
    regex=False
).astype(
    int
)


# ======================================================================================
# 5. TASK 2 — ANY CONDUCTION ABNORMALITY
#
# Positive:
#     any non-missing entry in "Conduction abnormalities"
#
# Negative:
#     no documented conduction abnormality
#
# This is a LUDB-specific downstream task, NOT a direct PTB-XL output mapping.
# ======================================================================================

conduction_text = ludb[
    "Conduction abnormalities"
].apply(
    clean_text
)


ludb[
    "target_Conduction"
] = (
    conduction_text
    !=
    ""
).astype(
    int
)


# ======================================================================================
# 6. TASK 3 — LEFT VENTRICULAR HYPERTROPHY
#
# Exact LUDB diagnosis string.
# ======================================================================================

hypertrophy_text = ludb[
    "Hypertrophies"
].apply(
    clean_text
)


ludb[
    "target_LVH"
] = hypertrophy_text.str.contains(
    "left ventricular hypertrophy",
    regex=False
).astype(
    int
)


# ======================================================================================
# 7. TASK 4 — CARDIAC PACING
#
# Exploratory because only ~10 positives expected.
#
# Positive:
#     any non-empty Cardiac pacing field.
#
# IMPORTANT:
#     This is NOT mapped to PTB-XL "PMI".
# ======================================================================================

pacing_text = ludb[
    "Cardiac pacing"
].apply(
    clean_text
)


ludb[
    "target_Paced"
] = (
    pacing_text
    !=
    ""
).astype(
    int
)


# ======================================================================================
# 8. TASK 5 — DEFINITE ISCHAEMIA / SCAR / STEMI
#
# LUDB contains:
#
#   definite:
#       Ischemia: ...
#       Scar formation: ...
#       STEMI: ...
#
#   ambiguous:
#       Undefined ischemia/scar/supp.nstemi: ...
#
# Definition:
#
#   POSITIVE
#       >=1 definite line
#
#   NEGATIVE
#       completely empty Ischemia field
#
#   EXCLUDED / INDETERMINATE
#       field contains only "Undefined ischemia/scar/supp.nstemi"
#
# If definite and undefined findings coexist:
#       POSITIVE
#
# This definition is frozen before model evaluation.
# ======================================================================================

def classify_ischaemia(value):

    lines = split_lines(
        value
    )


    if len(lines) == 0:

        return 0


    definite = False
    ambiguous = False


    for line in lines:

        if (
            line.startswith(
                "ischemia:"
            )
            or
            line.startswith(
                "scar formation:"
            )
            or
            line.startswith(
                "stemi:"
            )
        ):

            definite = True


        if line.startswith(
            "undefined ischemia/scar/supp.nstemi:"
        ):

            ambiguous = True


    if definite:
        return 1


    if ambiguous:
        return np.nan


    # Defensive:
    # unexpected non-empty Ischemia content should not silently
    # become negative.
    raise RuntimeError(
        f"Unexpected Ischemia field content: {value}"
    )


ludb[
    "target_Ischaemia_related"
] = ludb[
    "Ischemia"
].apply(
    classify_ischaemia
)


# ======================================================================================
# 9. TASK COUNTS
# ======================================================================================

TASKS = {

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


PRIMARY_TASKS = [
    "Conduction",
    "LVH",
    "Ischaemia_related",
]


EXPLORATORY_TASKS = [
    "AFib",
    "Paced",
]


task_rows = []


for task_name, column in TASKS.items():

    valid = ludb[
        column
    ].notna()


    y = ludb.loc[
        valid,
        column
    ].astype(
        int
    )


    task_rows.append(
        {
            "task":
                task_name,

            "role":
                (
                    "primary"
                    if task_name in PRIMARY_TASKS
                    else
                    "exploratory"
                ),

            "N":
                int(
                    valid.sum()
                ),

            "positive":
                int(
                    y.sum()
                ),

            "negative":
                int(
                    len(y)
                    -
                    y.sum()
                ),

            "excluded_indeterminate":
                int(
                    (
                        ~valid
                    ).sum()
                ),

            "prevalence":
                float(
                    y.mean()
                ),
        }
    )


task_df = pd.DataFrame(
    task_rows
)


print("\n" + "=" * 100)
print("FROZEN LUDB TASK DEFINITIONS")
print("=" * 100)

print(
    task_df
    .round(4)
    .to_string(
        index=False
    )
)


# ======================================================================================
# 10. CHECK AGAINST HISTORICAL COUNTS
#
# We are NOT using the old results.
#
# These checks merely tell us whether our clean definitions reproduce
# the previously observed dataset counts.
# ======================================================================================

EXPECTED_COUNTS = {

    "AFib":
        (
            200,
            15
        ),

    "Ischaemia_related":
        (
            192,
            43
        ),

    "Conduction":
        (
            200,
            66
        ),

    "LVH":
        (
            200,
            108
        ),

    "Paced":
        (
            200,
            10
        ),
}


print("\n" + "=" * 100)
print("HISTORICAL COUNT REPRODUCTION")
print("=" * 100)


all_counts_match = True


for task_name, (
    expected_n,
    expected_positive
) in EXPECTED_COUNTS.items():

    row = task_df[
        task_df[
            "task"
        ]
        ==
        task_name
    ].iloc[
        0
    ]


    actual_n = int(
        row[
            "N"
        ]
    )


    actual_positive = int(
        row[
            "positive"
        ]
    )


    match = (
        actual_n
        ==
        expected_n
        and
        actual_positive
        ==
        expected_positive
    )


    print(
        f"{task_name:20s} "
        f"N={actual_n:3d} "
        f"positive={actual_positive:3d} "
        f"| expected N={expected_n:3d}, positive={expected_positive:3d} "
        f"| {'MATCH' if match else 'DIFFERENT'}"
    )


    if not match:

        all_counts_match = False


if all_counts_match:

    print(
        "\n✓ All five clean task definitions reproduce the historical dataset counts."
    )

else:

    print(
        "\n⚠ At least one clean task definition differs from the historical count."
    )


# ======================================================================================
# 11. FREEZE FEATURE LAYER
#
# Historical LUDB work used dense_1, a 64-dimensional frozen representation.
#
# We inspect it here before using LUDB signals.
# ======================================================================================

with open(
    STEP1_MANIFEST,
    "r"
) as f:

    step1_manifest = json.load(
        f
    )


BASE_MODEL_PATH = Path(
    step1_manifest[
        "base_model"
    ]
)


base_model = tf.keras.models.load_model(
    BASE_MODEL_PATH,
    compile=False
)


layer_names = [
    layer.name
    for layer in base_model.layers
]


if "dense_1" not in layer_names:

    raise RuntimeError(
        "dense_1 not found in base model."
    )


dense1_index = layer_names.index(
    "dense_1"
)


print("\n" + "=" * 100)
print("FROZEN REPRESENTATION LAYER AUDIT")
print("=" * 100)


start = max(
    0,
    dense1_index - 4
)

end = min(
    len(
        base_model.layers
    ),
    dense1_index + 4
)


for i in range(
    start,
    end
):

    layer = base_model.layers[
        i
    ]


    activation = ""


    if hasattr(
        layer,
        "activation"
    ):

        try:

            activation = (
                " | activation="
                +
                layer.activation.__name__
            )

        except Exception:
            pass


    print(
        f"{i:02d} | "
        f"{layer.name:28s} | "
        f"{layer.__class__.__name__:24s} | "
        f"output={layer.output.shape}"
        f"{activation}"
    )


dense1 = base_model.get_layer(
    "dense_1"
)


print(
    "\nSelected frozen feature layer:"
)

print(
    "    dense_1"
)

print(
    "Feature dimension:",
    dense1.output.shape
)


if dense1.output.shape[-1] != 64:

    raise RuntimeError(
        f"Expected dense_1 dimension 64; found {dense1.output.shape}."
    )


print(
    "\n✓ dense_1 frozen as the LUDB representation-transfer feature layer."
)


# ======================================================================================
# 12. FREEZE LUDB SIGNAL PREPROCESSING
#
# Native:
#     500 Hz × 5000 samples
#
# External model input:
#     100 Hz × 1000 samples
#
# Method:
#     scipy.signal.resample_poly
#     up=1, down=5
#
# This provides anti-aliasing before downsampling.
#
# No z-score or amplitude normalization is added.
# ======================================================================================

def preprocess_ludb_signal(
    signal,
    fs
):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )


    if signal.shape != (
        5000,
        12
    ):

        raise RuntimeError(
            f"Unexpected LUDB native shape: {signal.shape}"
        )


    if float(
        fs
    ) != 500.0:

        raise RuntimeError(
            f"Unexpected LUDB sampling frequency: {fs}"
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
            f"Unexpected resampled shape: {downsampled.shape}"
        )


    if not np.all(
        np.isfinite(
            downsampled
        )
    ):

        raise RuntimeError(
            "Non-finite values after LUDB resampling."
        )


    # Model convention:
    # (lead, time, channel)

    model_input = (
        downsampled.T[
            :,
            :,
            None
        ]
    ).astype(
        np.float32
    )


    if model_input.shape != (
        12,
        1000,
        1
    ):

        raise RuntimeError(
            f"Unexpected model-input shape: {model_input.shape}"
        )


    return model_input


# Test preprocessing on ONE record only.
first_id = int(
    ludb.iloc[
        0
    ][
        "ID"
    ]
)


sample_signal, sample_fields = wfdb.rdsamp(
    str(
        DATA_DIR
        /
        str(
            first_id
        )
    )
)


sample_preprocessed = preprocess_ludb_signal(
    sample_signal,
    sample_fields[
        "fs"
    ]
)


print("\n" + "=" * 100)
print("FROZEN LUDB PREPROCESSING")
print("=" * 100)


print(
    "Native shape:",
    sample_signal.shape
)

print(
    "Native fs:",
    sample_fields[
        "fs"
    ]
)

print(
    "Prepared model shape:",
    sample_preprocessed.shape
)

print(
    "Prepared minimum:",
    float(
        sample_preprocessed.min()
    )
)

print(
    "Prepared maximum:",
    float(
        sample_preprocessed.max()
    )
)


print(
    "\nFrozen preprocessing:"
)

print(
    "    500 Hz -> 100 Hz using scipy.signal.resample_poly(up=1, down=5)"
)

print(
    "    no additional amplitude normalisation"
)

print(
    "    lead order unchanged"
)


print(
    "\nNO MODEL INFERENCE WAS PERFORMED."
)


# ======================================================================================
# 13. SAVE CLEAN LUDB TASK TABLE
# ======================================================================================

save_columns = [
    "ID",

    "target_Conduction",
    "target_LVH",
    "target_Ischaemia_related",
    "target_AFib",
    "target_Paced",
]


ludb[
    save_columns
].to_csv(
    OUT_DIR
    /
    "ludb_frozen_targets.csv",
    index=False
)


task_df.to_csv(
    OUT_DIR
    /
    "ludb_task_counts.csv",
    index=False
)


# ======================================================================================
# 14. SAVE MANIFEST
# ======================================================================================

manifest = {

    "stage":
        "Step 5B frozen LUDB downstream-task definition",

    "analysis_type":
        "external frozen-representation linear-probe evaluation",

    "not_direct_23class_validation":
        True,

    "records":
        200,

    "primary_tasks":
        PRIMARY_TASKS,

    "exploratory_tasks":
        EXPLORATORY_TASKS,

    "task_counts":
        task_df.to_dict(
            orient="records"
        ),

    "feature_layer":
        "dense_1",

    "feature_dimension":
        64,

    "native_sampling_frequency_Hz":
        500,

    "native_samples":
        5000,

    "model_sampling_frequency_Hz":
        100,

    "model_samples":
        1000,

    "resampling":
        "scipy.signal.resample_poly(up=1, down=5, axis=time)",

    "amplitude_normalisation":
        "none",

    "model_inference_performed":
        False,

    "probe_training_performed":
        False,

    "gaze_model_training_performed":
        False,
}


with open(
    OUT_DIR
    /
    "ludb_step5b_manifest.json",
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ======================================================================================
# 15. FINISH
# ======================================================================================

print("\n\n" + "=" * 100)
print("STEP 5B COMPLETE — TASKS/PREPROCESSING FROZEN")
print("=" * 100)


print(
    """
Please paste back only:

1. FROZEN LUDB TASK DEFINITIONS

2. HISTORICAL COUNT REPRODUCTION

3. FROZEN REPRESENTATION LAYER AUDIT

4. FROZEN LUDB PREPROCESSING

If those pass, Step 5C will be the actual external representation-transfer
experiment:

    Base vs Consultant vs All-clinician

using identical 5-fold out-of-fold linear probes on these frozen LUDB tasks.
"""
)
