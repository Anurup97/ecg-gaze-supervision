# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 5
# Stage: PTB-XL metadata, classes, fold checks
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 4A — PTB-XL PROVENANCE + LABEL + SPLIT AUDIT
#
# READ ONLY.
# NO MODEL INFERENCE.
# NO TRAINING.
#
# Goals:
#   1. Locate the local PTB-XL dataset deterministically.
#   2. Audit record/fold counts.
#   3. Reconstruct the source repository's 23 diagnostic-subclass labels.
#   4. Reconstruct the MultiLabelBinarizer class order.
#   5. Confirm fold 10 is suitable as the source-model held-out test fold.
#   6. Confirm patient separation between fold 10 and folds 1–9.
#   7. Verify raw 100-Hz waveform shape and lead order.
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer


# ======================================================================================
# 1. EXACT EXPECTED MODEL OUTPUT ORDER
#
# This should be reproduced by MultiLabelBinarizer fitted on folds != 10.
# ======================================================================================

EXPECTED_MODEL_CLASSES = [
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


# ======================================================================================
# 2. LOCATE PTB-XL ROOT
#
# We require ptbxl_database.csv and scp_statements.csv in the same directory.
# ======================================================================================

database_candidates = list(
    Path(".").glob(
        "**/ptbxl_database.csv"
    )
)


roots = []


for database_path in database_candidates:

    root = database_path.parent

    if (
        root
        /
        "scp_statements.csv"
    ).exists():

        roots.append(
            root
        )


roots = sorted(
    set(
        roots
    ),
    key=lambda p: str(p)
)


print("=" * 100)
print("PTB-XL DATASET SEARCH")
print("=" * 100)


if len(roots) == 0:

    raise FileNotFoundError(
        "\nCould not find a directory containing BOTH:\n"
        "  ptbxl_database.csv\n"
        "  scp_statements.csv\n\n"
        "Place/unzip PTB-XL somewhere below the current working directory."
    )


print(
    "Candidate PTB-XL roots:"
)


for i, root in enumerate(
    roots,
    start=1
):

    print(
        f"  {i}. {root}"
    )


if len(roots) != 1:

    raise RuntimeError(
        "\nMore than one PTB-XL dataset was found. "
        "Do not automatically choose between different releases/copies."
    )


PTBXL_ROOT = roots[0]


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


print(
    "\nSelected root:"
)

print(
    PTBXL_ROOT
)


# ======================================================================================
# 3. LOAD METADATA
# ======================================================================================

metadata = pd.read_csv(
    DATABASE_PATH,
    index_col="ecg_id"
)


scp = pd.read_csv(
    SCP_PATH,
    index_col=0
)


print("\n" + "=" * 100)
print("PTB-XL METADATA")
print("=" * 100)


print(
    "ptbxl_database shape:",
    metadata.shape
)


print(
    "scp_statements shape:",
    scp.shape
)


print(
    "ECG IDs unique:",
    metadata.index.is_unique
)


required_metadata_columns = [
    "patient_id",
    "scp_codes",
    "strat_fold",
    "filename_lr",
]


missing = [
    column
    for column in required_metadata_columns
    if column not in metadata.columns
]


if missing:

    raise RuntimeError(
        f"Missing PTB-XL metadata columns: {missing}"
    )


required_scp_columns = [
    "diagnostic",
    "diagnostic_subclass",
]


missing_scp = [
    column
    for column in required_scp_columns
    if column not in scp.columns
]


if missing_scp:

    raise RuntimeError(
        f"Missing SCP columns: {missing_scp}"
    )


# ======================================================================================
# 4. AUDIT STRATIFIED FOLDS
# ======================================================================================

fold_counts = (
    metadata[
        "strat_fold"
    ]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 100)
print("STRATIFIED FOLD COUNTS")
print("=" * 100)


print(
    fold_counts.to_string()
)


observed_folds = set(
    metadata[
        "strat_fold"
    ].dropna().astype(int)
)


if observed_folds != set(
    range(
        1,
        11
    )
):

    raise RuntimeError(
        f"Unexpected strat_fold values: {sorted(observed_folds)}"
    )


train_mask = (
    metadata[
        "strat_fold"
    ]
    !=
    10
)


test_mask = (
    metadata[
        "strat_fold"
    ]
    ==
    10
)


print(
    "\nFolds 1–9:",
    int(
        train_mask.sum()
    )
)


print(
    "Fold 10:",
    int(
        test_mask.sum()
    )
)


# ======================================================================================
# 5. PATIENT-LEVEL TRAIN/TEST SEPARATION
# ======================================================================================

train_patients = set(
    metadata.loc[
        train_mask,
        "patient_id"
    ].dropna()
)


test_patients = set(
    metadata.loc[
        test_mask,
        "patient_id"
    ].dropna()
)


patient_overlap = (
    train_patients
    &
    test_patients
)


print("\n" + "=" * 100)
print("PATIENT-LEVEL SPLIT AUDIT")
print("=" * 100)


print(
    "Unique patients folds 1–9:",
    len(
        train_patients
    )
)


print(
    "Unique patients fold 10:",
    len(
        test_patients
    )
)


print(
    "Patients occurring in BOTH:",
    len(
        patient_overlap
    )
)


if len(
    patient_overlap
) != 0:

    raise RuntimeError(
        "Patient overlap exists between fold 10 and folds 1–9."
    )


print(
    "\n✓ No patient overlap between source training and test partitions."
)


# ======================================================================================
# 6. PARSE SCP CODES
# ======================================================================================

def parse_scp_codes(value):

    if isinstance(
        value,
        dict
    ):

        return value


    if pd.isna(
        value
    ):

        return {}


    parsed = ast.literal_eval(
        str(
            value
        )
    )


    if not isinstance(
        parsed,
        dict
    ):

        raise RuntimeError(
            f"scp_codes did not parse as dict: {value}"
        )


    return parsed


metadata[
    "_scp_dict"
] = metadata[
    "scp_codes"
].apply(
    parse_scp_codes
)


# ======================================================================================
# 7. RECONSTRUCT SOURCE REPOSITORY SUBCLASS MAPPING
#
# Source logic:
#
#   agg_df = scp_statements[diagnostic == 1]
#
# For every SCP code present in each ECG:
#     diagnostic_subclass is appended.
#
# No confidence-value threshold is applied in the source notebook.
# ======================================================================================

diagnostic_scp = scp[
    scp[
        "diagnostic"
    ]
    ==
    1
].copy()


def aggregate_diagnostic_subclasses(
    scp_dict
):

    labels = []


    for code in scp_dict.keys():

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


        subclass = str(
            subclass
        )


        labels.append(
            subclass
        )


    return sorted(
        set(
            labels
        )
    )


metadata[
    "_diagnostic_subclasses"
] = metadata[
    "_scp_dict"
].apply(
    aggregate_diagnostic_subclasses
)


# ======================================================================================
# 8. RECONSTRUCT MULTILABELBINARIZER ORDER FROM SOURCE TRAINING PARTITION
# ======================================================================================

train_label_lists = metadata.loc[
    train_mask,
    "_diagnostic_subclasses"
].tolist()


test_label_lists = metadata.loc[
    test_mask,
    "_diagnostic_subclasses"
].tolist()


mlb = MultiLabelBinarizer()


Y_train = mlb.fit_transform(
    train_label_lists
)


Y_test = mlb.transform(
    test_label_lists
)


reconstructed_classes = list(
    mlb.classes_
)


print("\n" + "=" * 100)
print("RECONSTRUCTED 23-CLASS MODEL OUTPUT ORDER")
print("=" * 100)


for i, class_name in enumerate(
    reconstructed_classes
):

    print(
        f"{i:02d}  {class_name}"
    )


print(
    "\nNumber of classes:",
    len(
        reconstructed_classes
    )
)


if reconstructed_classes != EXPECTED_MODEL_CLASSES:

    print(
        "\nEXPECTED:"
    )

    print(
        EXPECTED_MODEL_CLASSES
    )


    print(
        "\nRECONSTRUCTED:"
    )

    print(
        reconstructed_classes
    )


    raise RuntimeError(
        "Reconstructed PTB-XL subclass order does not match "
        "the frozen 23-output model vocabulary."
    )


print(
    "\n✓ Exact 23-output class vocabulary/order reproduced."
)


# ======================================================================================
# 9. CHECK WHETHER TEST FOLD CONTAINS SAME CLASS SET
# ======================================================================================

test_class_set = sorted(
    {
        label
        for labels in test_label_lists
        for label in labels
    }
)


train_class_set = sorted(
    reconstructed_classes
)


print("\n" + "=" * 100)
print("TEST-FOLD CLASS COVERAGE")
print("=" * 100)


print(
    "Classes represented in folds 1–9:",
    len(
        train_class_set
    )
)


print(
    "Classes represented in fold 10:",
    len(
        test_class_set
    )
)


missing_from_test = sorted(
    set(
        train_class_set
    )
    -
    set(
        test_class_set
    )
)


extra_in_test = sorted(
    set(
        test_class_set
    )
    -
    set(
        train_class_set
    )
)


print(
    "Training classes absent from fold 10:",
    missing_from_test
)


print(
    "Unexpected fold-10 classes:",
    extra_in_test
)


# ======================================================================================
# 10. CLASS PREVALENCE
# ======================================================================================

train_positive_counts = Y_train.sum(
    axis=0
)


test_positive_counts = Y_test.sum(
    axis=0
)


prevalence_df = pd.DataFrame(
    {
        "class":
            reconstructed_classes,

        "train_positive":
            train_positive_counts.astype(
                int
            ),

        "test_positive":
            test_positive_counts.astype(
                int
            ),

        "train_prevalence":
            train_positive_counts
            /
            len(
                Y_train
            ),

        "test_prevalence":
            test_positive_counts
            /
            len(
                Y_test
            ),
    }
)


print("\n" + "=" * 100)
print("23-CLASS PREVALENCE")
print("=" * 100)


print(
    prevalence_df
    .round(
        5
    )
    .to_string(
        index=False
    )
)


classes_with_no_test_positives = prevalence_df.loc[
    prevalence_df[
        "test_positive"
    ]
    ==
    0,
    "class"
].tolist()


classes_with_all_test_positives = prevalence_df.loc[
    prevalence_df[
        "test_positive"
    ]
    ==
    len(
        Y_test
    ),
    "class"
].tolist()


print(
    "\nClasses with zero positives in test:",
    classes_with_no_test_positives
)


print(
    "Classes with no negatives in test:",
    classes_with_all_test_positives
)


# ======================================================================================
# 11. ALL-ZERO TARGET ROWS
#
# The source preprocessing does not explicitly drop ECGs that have no mapped
# diagnostic subclass, so quantify them rather than silently removing them.
# ======================================================================================

train_all_zero = (
    Y_train.sum(
        axis=1
    )
    ==
    0
)


test_all_zero = (
    Y_test.sum(
        axis=1
    )
    ==
    0
)


print("\n" + "=" * 100)
print("ALL-ZERO DIAGNOSTIC-SUBCLASS ROWS")
print("=" * 100)


print(
    "Folds 1–9:",
    int(
        train_all_zero.sum()
    )
)


print(
    "Fold 10:",
    int(
        test_all_zero.sum()
    )
)


# ======================================================================================
# 12. CHECK RAW LOW-RESOLUTION FILE PATHS
# ======================================================================================

filename_lr = metadata[
    "filename_lr"
].astype(
    str
)


raw_exists = filename_lr.apply(
    lambda relative_path: (
        PTBXL_ROOT
        /
        relative_path
    ).with_suffix(
        ".dat"
    ).exists()
)


print("\n" + "=" * 100)
print("100-HZ WAVEFORM FILE AUDIT")
print("=" * 100)


print(
    "Records with .dat file present:",
    int(
        raw_exists.sum()
    ),
    "/",
    len(
        metadata
    )
)


test_raw_exists = raw_exists.loc[
    test_mask
]


print(
    "Fold-10 .dat files present:",
    int(
        test_raw_exists.sum()
    ),
    "/",
    int(
        test_mask.sum()
    )
)


if not test_raw_exists.all():

    missing_paths = metadata.loc[
        test_mask
        &
        ~raw_exists,
        "filename_lr"
    ].head(
        20
    )


    raise RuntimeError(
        "\nSome fold-10 raw files are missing:\n"
        +
        missing_paths.to_string()
    )


# ======================================================================================
# 13. READ ONE RAW ECG AND VERIFY SIGNAL GEOMETRY
# ======================================================================================

try:

    import wfdb

except ImportError as exc:

    raise ImportError(
        "\nThe metadata/label audit passed so far, but wfdb is not installed.\n"
        "Install it with:\n\n"
        "    pip install wfdb\n\n"
        "Then rerun this Step-4A cell."
    ) from exc


first_test_row = metadata.loc[
    test_mask
].iloc[
    0
]


sample_path = (
    PTBXL_ROOT
    /
    str(
        first_test_row[
            "filename_lr"
        ]
    )
)


signal, signal_metadata = wfdb.rdsamp(
    str(
        sample_path
    )
)


print("\n" + "=" * 100)
print("RAW SIGNAL GEOMETRY")
print("=" * 100)


print(
    "Example record:",
    sample_path
)


print(
    "Raw shape:",
    signal.shape
)


print(
    "Sampling frequency:",
    signal_metadata.get(
        "fs"
    )
)


signal_leads = [
    str(
        lead
    ).upper()
    for lead
    in signal_metadata.get(
        "sig_name",
        []
    )
]


print(
    "Lead order:",
    signal_leads
)


print(
    "Finite:",
    np.all(
        np.isfinite(
            signal
        )
    )
)


if signal.shape != (
    1000,
    12
):

    raise RuntimeError(
        f"Expected raw 100-Hz shape (1000, 12), found {signal.shape}."
    )


if float(
    signal_metadata.get(
        "fs"
    )
) != 100.0:

    raise RuntimeError(
        f"Expected 100 Hz, found {signal_metadata.get('fs')}."
    )


if signal_leads != EXPECTED_LEADS:

    raise RuntimeError(
        "\nUnexpected PTB-XL lead order.\n"
        f"Expected: {EXPECTED_LEADS}\n"
        f"Actual:   {signal_leads}"
    )


if not np.all(
    np.isfinite(
        signal
    )
):

    raise RuntimeError(
        "Example signal contains non-finite values."
    )


print(
    "\n✓ Raw PTB-XL geometry matches model input convention."
)


# ======================================================================================
# 14. PREPARED MODEL INPUT SHAPE
#
# Source preprocessing loads:
#     raw = (N, 1000, 12)
#
# then transposes:
#     (N, 12, 1000)
#
# and adds a single channel:
#     (N, 12, 1000, 1)
#
# We demonstrate this on ONE ECG only.
# ======================================================================================

sample_model_input = (
    signal.T[
        None,
        :,
        :,
        None
    ]
).astype(
    np.float32
)


print("\n" + "=" * 100)
print("MODEL-INPUT GEOMETRY CHECK")
print("=" * 100)


print(
    "One prepared ECG:",
    sample_model_input.shape
)


if sample_model_input.shape != (
    1,
    12,
    1000,
    1
):

    raise RuntimeError(
        "Prepared model-input shape is incorrect."
    )


print(
    "\n✓ Prepared PTB-XL input has shape (1, 12, 1000, 1)."
)


# ======================================================================================
# 15. SAVE READ-ONLY AUDIT OUTPUTS
# ======================================================================================

AUDIT_DIR = Path(
    "definitive_clean_gaze/ptbxl_audit"
)


AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


prevalence_df.to_csv(
    AUDIT_DIR
    /
    "ptbxl_23class_prevalence.csv",
    index=False
)


fold_counts.rename(
    "n_records"
).to_csv(
    AUDIT_DIR
    /
    "ptbxl_fold_counts.csv"
)


audit_manifest = {

    "stage":
        "Step 4A PTB-XL read-only provenance audit",

    "ptbxl_root":
        str(
            PTBXL_ROOT
        ),

    "metadata_records":
        int(
            len(
                metadata
            )
        ),

    "fold_10_records":
        int(
            test_mask.sum()
        ),

    "folds_1_to_9_records":
        int(
            train_mask.sum()
        ),

    "patient_overlap_between_folds_1_to_9_and_10":
        int(
            len(
                patient_overlap
            )
        ),

    "model_class_order":
        reconstructed_classes,

    "n_model_classes":
        int(
            len(
                reconstructed_classes
            )
        ),

    "fold10_missing_classes":
        missing_from_test,

    "fold10_all_zero_rows":
        int(
            test_all_zero.sum()
        ),

    "raw_signal_shape":
        list(
            signal.shape
        ),

    "sampling_frequency":
        float(
            signal_metadata.get(
                "fs"
            )
        ),

    "lead_order":
        signal_leads,

    "model_input_shape":
        list(
            sample_model_input.shape
        ),

    "model_inference_performed":
        False,

    "training_performed":
        False,
}


with open(
    AUDIT_DIR
    /
    "ptbxl_audit_manifest.json",
    "w"
) as f:

    json.dump(
        audit_manifest,
        f,
        indent=2
    )


# ======================================================================================
# 16. FINAL SUMMARY
# ======================================================================================

print("\n\n" + "=" * 100)
print("STEP 4A COMPLETE — READ-ONLY AUDIT")
print("=" * 100)


print(
    """
Please paste back these FIVE sections:

1. PTB-XL METADATA

2. STRATIFIED FOLD COUNTS

3. RECONSTRUCTED 23-CLASS MODEL OUTPUT ORDER

4. 23-CLASS PREVALENCE
   (including the zero-positive / no-negative lines)

5. RAW SIGNAL GEOMETRY

Do not run diagnostic model inference yet.
"""
)
