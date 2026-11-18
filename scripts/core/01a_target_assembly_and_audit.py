# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: New training 30.08.ipynb ; code cell 1
# Stage: freeze source data and source-model audit
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# DEFINITIVE CLEAN GAZE EXPERIMENT
# STEP 1 — FROZEN INPUT + FINAL GAZE TARGET CONSTRUCTION + BASE MODEL AUDIT
#
# RUN AFTER A FRESH KERNEL RESTART.
#
# IMPORTANT:
#     THIS SCRIPT DOES NOT TRAIN ANY MODEL.
#
# Frozen ECG input:
#     final_digitisation_v9/X_gaze_final.npy
#
# Gaze populations:
#     Consultant       = 9 consultants
#     All clinicians   = 9 consultants + 10 fellows + 1 resident = 20
#
# Gaze target construction:
#     1. Static AOI rows only
#     2. Exact source stimulus mapping
#     3. Exactly 12 diagnostic-grid AOIs
#     4. Per-reader 12-lead normalisation
#     5. Equal-weight averaging across readers
#
# Standard lead order:
#     I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
#
# NO TRAINING IN THIS STEP.
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import re
import json
import hashlib
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf


# ======================================================================================
# 1. REPRODUCIBILITY
# ======================================================================================

SEED = 2026

np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 100)
print("ENVIRONMENT")
print("=" * 100)

print("Python:", platform.python_version())
print("TensorFlow:", tf.__version__)
print("Working directory:", Path.cwd())


# ======================================================================================
# 2. FIXED EXPERIMENT DEFINITIONS
# ======================================================================================

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


# Exact strings observed in ParentStimulus.
EXACT_STIMULUS_MAP = {

    "Normal sinus rhythm":
        "NSR",

    "Atrial fibrillation":
        "AFib",

    "hyperkalemia":
        "Hyperkalemia",

    "Atrial flutter":
        "AFlutter",

    "Ventricular tachycardia":
        "VTach",

    "Wolf Parkinson White syndrome":
        "WPW",

    "Ventricular paced rhythm":
        "VentPaced",

    "Left bundle branch block":
        "LBBB",

    "ST elevation MI":
        "STEMI",

    "Complete heart block":
        "CompleteHeartBlock",
}


# Diagnosis suffix used inside the Label column.
LABEL_SUFFIX = {

    "NSR":
        "NSR",

    "AFib":
        "AFib",

    "Hyperkalemia":
        "Hyperkalemia",

    "AFlutter":
        "AFlutter",

    "VTach":
        "VTach",

    "WPW":
        "WPW",

    "VentPaced":
        "VentPaced",

    "LBBB":
        "LBBB",

    "STEMI":
        "STEMI",

    "CompleteHeartBlock":
        "CompleteHeartBlock",
}


# Exact diagnostic-grid AOIs.
#
# IMPORTANT:
# Atrial flutter uses V1 short and V5 short in the source AOI definitions.
#
# Other AOIs such as:
#     II-1 / II-2 / ...
#     V1-1 / V5-1 / ...
#     Information
#     V3R / V4R / V7 / V8
# are deliberately excluded.
BASE_AOI_TO_LEAD = {

    "1":
        "I",

    "2":
        "II",

    "3":
        "III",

    "aVR":
        "aVR",

    "aVL":
        "aVL",

    "aVF":
        "aVF",

    "V1":
        "V1",

    "V2":
        "V2",

    "V3":
        "V3",

    "V4":
        "V4",

    "V5":
        "V5",

    "V6":
        "V6",

    "V1 short":
        "V1",

    "V5 short":
        "V5",
}


# ======================================================================================
# 3. PATHS
# ======================================================================================

OUT_DIR = Path(
    "definitive_clean_gaze"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# IMPORTANT:
# Do not fall back to an older X_gaze_final.npy.
V9_PATH = Path(
    "final_digitisation_v9/X_gaze_final.npy"
)


if not V9_PATH.exists():

    raise FileNotFoundError(
        "\nCould not find the frozen V9 dataset at:\n"
        f"    {V9_PATH}\n\n"
        "Do not substitute an earlier digitisation version."
    )


# Find gaze CSV by exact filename.
gaze_candidates = list(
    Path(".").glob(
        "**/Grid_Anonymized.csv"
    )
)


if not gaze_candidates:

    raise FileNotFoundError(
        "Could not find Grid_Anonymized.csv"
    )


gaze_candidates = sorted(
    gaze_candidates,
    key=lambda p: (
        len(p.parts),
        str(p)
    )
)


GAZE_CSV_PATH = gaze_candidates[0]


# Base model
base_candidates = (
    list(
        Path(".").glob(
            "**/12 leads.h5"
        )
    )
    +
    list(
        Path(".").glob(
            "**/12_leads.h5"
        )
    )
)


if not base_candidates:

    raise FileNotFoundError(
        "Could not find base model '12 leads.h5' or '12_leads.h5'."
    )


base_candidates = sorted(
    base_candidates,
    key=lambda p: (
        len(p.parts),
        str(p)
    )
)


BASE_MODEL_PATH = base_candidates[0]


print("\n" + "=" * 100)
print("FILES")
print("=" * 100)

print("Frozen V9 ECG input:", V9_PATH)
print("Eye-tracking CSV:   ", GAZE_CSV_PATH)
print("Base model:         ", BASE_MODEL_PATH)


# ======================================================================================
# 4. HASH HELPER
# ======================================================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


V9_HASH = sha256_file(
    V9_PATH
)

GAZE_HASH = sha256_file(
    GAZE_CSV_PATH
)

BASE_HASH = sha256_file(
    BASE_MODEL_PATH
)


# ======================================================================================
# 5. LOAD FROZEN V9
# ======================================================================================

X_gaze = np.load(
    V9_PATH
).astype(
    np.float32
)


print("\n" + "=" * 100)
print("FROZEN V9 ECG INPUT")
print("=" * 100)

print("Shape:", X_gaze.shape)
print("Finite:", np.all(np.isfinite(X_gaze)))
print("SHA256:", V9_HASH)


if X_gaze.shape != (
    10,
    12,
    1000,
    1
):

    raise RuntimeError(
        f"Unexpected V9 shape: {X_gaze.shape}"
    )


if not np.all(
    np.isfinite(
        X_gaze
    )
):

    raise RuntimeError(
        "V9 contains non-finite values."
    )


# Each pseudo-waveform should have non-zero variance.
lead_sd = np.std(
    X_gaze[:, :, :, 0],
    axis=2
)


if np.any(
    lead_sd <= 1e-8
):

    bad = np.argwhere(
        lead_sd <= 1e-8
    )

    raise RuntimeError(
        f"At least one frozen pseudo-waveform has zero variance: {bad}"
    )


print(
    "\n✓ V9 shape and finite-value checks pass."
)


# ======================================================================================
# 6. LOAD RAW EYE-TRACKING CSV
# ======================================================================================

gaze = pd.read_csv(
    GAZE_CSV_PATH
)


required_columns = [
    "Respondent_Name",
    "Group",
    "Type",
    "Label",
    "ParentStimulus",
    "Time_spent_G",
]


missing_columns = [
    c
    for c in required_columns
    if c not in gaze.columns
]


if missing_columns:

    raise RuntimeError(
        f"Required columns missing: {missing_columns}"
    )


print("\n" + "=" * 100)
print("RAW EYE-TRACKING DATA")
print("=" * 100)

print("Shape:", gaze.shape)
print(
    "Unique respondents:",
    gaze["Respondent_Name"].nunique()
)
print(
    "Missing ParentStimulus rows:",
    gaze["ParentStimulus"].isna().sum()
)


# ======================================================================================
# 7. EXACT STIMULUS MAPPING
# ======================================================================================

actual_nonmissing_stimuli = set(
    gaze.loc[
        gaze["ParentStimulus"].notna(),
        "ParentStimulus"
    ].unique()
)


expected_stimuli = set(
    EXACT_STIMULUS_MAP.keys()
)


unexpected_stimuli = (
    actual_nonmissing_stimuli
    -
    expected_stimuli
)


missing_stimuli = (
    expected_stimuli
    -
    actual_nonmissing_stimuli
)


if unexpected_stimuli:

    raise RuntimeError(
        "\nUnexpected non-missing ParentStimulus values:\n"
        +
        "\n".join(
            sorted(unexpected_stimuli)
        )
    )


if missing_stimuli:

    raise RuntimeError(
        "\nExpected ParentStimulus values missing:\n"
        +
        "\n".join(
            sorted(missing_stimuli)
        )
    )


gaze["_ecg"] = gaze[
    "ParentStimulus"
].map(
    EXACT_STIMULUS_MAP
)


genuine_unmapped = gaze.loc[
    gaze["ParentStimulus"].notna(),
    "_ecg"
].isna().sum()


if genuine_unmapped != 0:

    raise RuntimeError(
        f"{genuine_unmapped} genuine stimulus rows remain unmapped."
    )


print("\n" + "=" * 100)
print("EXACT STIMULUS MAPPING")
print("=" * 100)


for source, ecg in EXACT_STIMULUS_MAP.items():

    n = int(
        (
            gaze["ParentStimulus"]
            ==
            source
        ).sum()
    )

    print(
        f"{source:35s} -> "
        f"{ecg:20s} "
        f"| rows={n}"
    )


print(
    "\n✓ All ten non-missing source stimuli mapped exactly."
)


# ======================================================================================
# 8. NORMALISE READER GROUPS
# ======================================================================================

gaze["_group_norm"] = (
    gaze["Group"]
    .astype(str)
    .str.strip()
    .str.lower()
)


reader_groups = (
    gaze[
        [
            "Respondent_Name",
            "_group_norm"
        ]
    ]
    .drop_duplicates()
)


# Every reader should belong to one group only.
groups_per_reader = (
    reader_groups
    .groupby("Respondent_Name")
    ["_group_norm"]
    .nunique()
)


if not (
    groups_per_reader == 1
).all():

    raise RuntimeError(
        "At least one Respondent_Name maps to multiple groups."
    )


print("\n" + "=" * 100)
print("READER GROUP AUDIT")
print("=" * 100)

print(
    reader_groups[
        "_group_norm"
    ]
    .value_counts()
    .to_string()
)


n_consultants = int(
    (
        reader_groups["_group_norm"]
        ==
        "consultant"
    ).sum()
)


n_fellows = int(
    (
        reader_groups["_group_norm"]
        ==
        "fellow"
    ).sum()
)


n_residents = int(
    (
        reader_groups["_group_norm"]
        ==
        "resident"
    ).sum()
)


print("\nConsultants:", n_consultants)
print("Fellows:    ", n_fellows)
print("Residents:  ", n_residents)


if n_consultants != 9:

    raise RuntimeError(
        f"Expected 9 consultants, found {n_consultants}."
    )


if n_fellows != 10:

    raise RuntimeError(
        f"Expected 10 fellows, found {n_fellows}."
    )


if n_residents != 1:

    raise RuntimeError(
        f"Expected 1 resident, found {n_residents}."
    )


print(
    "\n✓ 9 consultants + 10 fellows + 1 resident confirmed."
)


# ======================================================================================
# 9. FILTER STATIC AOIs
# ======================================================================================

static = gaze[
    gaze["Type"]
    .astype(str)
    .str.strip()
    .str.lower()
    ==
    "static aoi"
].copy()


print("\n" + "=" * 100)
print("STATIC AOI AUDIT")
print("=" * 100)

print(
    "Static AOI rows:",
    len(static)
)


if len(static) != 14742:

    raise RuntimeError(
        f"Expected 14,742 Static AOI rows, found {len(static)}."
    )


# Every Static AOI should have a mapped ECG.
if static["_ecg"].isna().any():

    bad = static.loc[
        static["_ecg"].isna(),
        [
            "ParentStimulus",
            "Label"
        ]
    ]

    raise RuntimeError(
        "\nAt least one Static AOI row has no ECG mapping:\n"
        +
        bad.head(20).to_string(index=False)
    )


print(
    "✓ 14,742 Static AOI rows confirmed."
)


# ======================================================================================
# 10. REMOVE ECG SUFFIX FROM AOI LABEL
# ======================================================================================

def extract_base_aoi(row):

    label = str(
        row["Label"]
    ).strip()

    ecg = row["_ecg"]

    suffix = LABEL_SUFFIX[
        ecg
    ]

    # Removes diagnosis suffix plus optional repeated "copy".
    #
    # Examples:
    #
    # V2 STEMI
    #     -> V2
    #
    # V1 short AFlutter
    #     -> V1 short
    #
    # II-2 CompleteHeartBlock copy
    #     -> II-2
    #
    # II-3 CompleteHeartBlock copy copy
    #     -> II-3

    pattern = (
        r"\s+"
        +
        re.escape(suffix)
        +
        r"(?:\s+copy)*$"
    )

    base = re.sub(
        pattern,
        "",
        label
    ).strip()

    return base


static["_base_aoi"] = static.apply(
    extract_base_aoi,
    axis=1
)


# ======================================================================================
# 11. MAP ONLY THE 12 DIAGNOSTIC LEADS
# ======================================================================================

static["_lead"] = static[
    "_base_aoi"
].map(
    BASE_AOI_TO_LEAD
)


lead_rows = static[
    static["_lead"].notna()
].copy()


lead_rows["_dwell"] = pd.to_numeric(
    lead_rows[
        "Time_spent_G"
    ],
    errors="coerce"
)


if lead_rows["_dwell"].isna().any():

    n_nan = int(
        lead_rows["_dwell"].isna().sum()
    )

    raise RuntimeError(
        f"{n_nan} selected diagnostic AOI rows have non-numeric Time_spent_G."
    )


if (
    lead_rows["_dwell"] < 0
).any():

    raise RuntimeError(
        "Negative dwell time detected."
    )


# ======================================================================================
# 12. STRUCTURAL AUDIT — 10 × 12 × 63
# ======================================================================================

audit_rows = []


for ecg in ECG_ORDER:

    temp = lead_rows[
        lead_rows["_ecg"]
        ==
        ecg
    ]


    lead_counts = (
        temp.groupby("_lead")
        .size()
        .reindex(LEAD_ORDER)
    )


    readers_per_lead = (
        temp.groupby("_lead")
        ["Respondent_Name"]
        .nunique()
        .reindex(LEAD_ORDER)
    )


    leads_present = set(
        temp["_lead"].unique()
    )


    audit_rows.append(
        {
            "ECG":
                ecg,

            "n_selected_rows":
                len(temp),

            "n_leads":
                len(leads_present),

            "min_rows_per_lead":
                lead_counts.min(),

            "max_rows_per_lead":
                lead_counts.max(),

            "min_readers_per_lead":
                readers_per_lead.min(),

            "max_readers_per_lead":
                readers_per_lead.max(),

            "missing_leads":
                (
                    ", ".join(
                        sorted(
                            set(LEAD_ORDER)
                            -
                            leads_present
                        )
                    )
                    or
                    "NONE"
                ),
        }
    )


structure_df = pd.DataFrame(
    audit_rows
)


print("\n" + "=" * 100)
print("FINAL 12-LEAD STRUCTURAL AUDIT")
print("=" * 100)

print(
    structure_df.to_string(
        index=False
    )
)


if not (
    structure_df["n_selected_rows"]
    ==
    12 * 63
).all():

    raise RuntimeError(
        "At least one ECG does not contain exactly 12 × 63 selected rows."
    )


if not (
    structure_df["n_leads"]
    ==
    12
).all():

    raise RuntimeError(
        "At least one ECG does not contain 12 selected diagnostic leads."
    )


if not (
    structure_df["min_rows_per_lead"]
    ==
    63
).all():

    raise RuntimeError(
        "At least one ECG/lead has fewer than 63 rows."
    )


if not (
    structure_df["max_rows_per_lead"]
    ==
    63
).all():

    raise RuntimeError(
        "At least one ECG/lead has more than 63 rows."
    )


if not (
    structure_df["min_readers_per_lead"]
    ==
    63
).all():

    raise RuntimeError(
        "At least one ECG/lead has fewer than 63 unique readers."
    )


if not (
    structure_df["max_readers_per_lead"]
    ==
    63
).all():

    raise RuntimeError(
        "At least one ECG/lead has more than 63 unique readers."
    )


print(
    "\n✓ 10 ECGs × 12 diagnostic leads × 63 readers confirmed."
)


# ======================================================================================
# 13. SHOW SPECIAL ATRIAL-FLUTTER MAPPING
# ======================================================================================

aflutter_mapping = (
    lead_rows[
        lead_rows["_ecg"]
        ==
        "AFlutter"
    ][
        [
            "_base_aoi",
            "_lead"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "_lead"
    )
)


print("\n" + "=" * 100)
print("ATRIAL FLUTTER DIAGNOSTIC AOIs")
print("=" * 100)

print(
    aflutter_mapping.to_string(
        index=False
    )
)


# Must explicitly use source special labels.
af_source = set(
    aflutter_mapping["_base_aoi"]
)


if "V1 short" not in af_source:

    raise RuntimeError(
        "AFlutter V1 short AOI was not selected."
    )


if "V5 short" not in af_source:

    raise RuntimeError(
        "AFlutter V5 short AOI was not selected."
    )


print(
    "\n✓ AFlutter V1 short -> V1 and V5 short -> V5 confirmed."
)


# ======================================================================================
# 14. EACH READER × ECG MUST HAVE EXACTLY 12 SELECTED AOIs
# ======================================================================================

reader_ecg_counts = (
    lead_rows
    .groupby(
        [
            "Respondent_Name",
            "_ecg"
        ]
    )
    .size()
)


if not (
    reader_ecg_counts
    ==
    12
).all():

    bad = reader_ecg_counts[
        reader_ecg_counts
        !=
        12
    ]

    raise RuntimeError(
        "\nAt least one reader × ECG does not have exactly 12 selected AOIs:\n"
        +
        bad.head(30).to_string()
    )


n_reader_ecg_pairs = len(
    reader_ecg_counts
)


expected_pairs = (
    63
    *
    10
)


if n_reader_ecg_pairs != expected_pairs:

    raise RuntimeError(
        f"Expected {expected_pairs} reader×ECG pairs, "
        f"found {n_reader_ecg_pairs}."
    )


print("\n" + "=" * 100)
print("READER × ECG COMPLETENESS")
print("=" * 100)

print(
    f"Reader × ECG pairs: {n_reader_ecg_pairs}"
)

print(
    "Selected diagnostic AOIs per pair:",
    reader_ecg_counts.min(),
    "to",
    reader_ecg_counts.max()
)

print(
    "\n✓ Every one of the 63 readers has exactly 12 selected leads for all 10 ECGs."
)


# ======================================================================================
# 15. CREATE ONE RAW 12-LEAD VECTOR PER READER × ECG
# ======================================================================================

# Because structural audit proved uniqueness, pivot is now safe.
reader_matrix = (
    lead_rows
    .pivot(
        index=[
            "Respondent_Name",
            "_group_norm",
            "_ecg"
        ],
        columns="_lead",
        values="_dwell"
    )
    .reindex(
        columns=LEAD_ORDER
    )
)


if reader_matrix.isna().any().any():

    raise RuntimeError(
        "Missing values appeared in reader × ECG × lead pivot."
    )


# Preserve raw totals before normalisation.
reader_matrix[
    "_raw_total"
] = reader_matrix[
    LEAD_ORDER
].sum(
    axis=1
)


if (
    reader_matrix["_raw_total"]
    <=
    0
).any():

    bad = reader_matrix[
        reader_matrix["_raw_total"]
        <=
        0
    ]

    raise RuntimeError(
        "\nAt least one reader × ECG has zero total dwell time:\n"
        +
        bad.head(20).to_string()
    )


# ======================================================================================
# 16. PER-READER NORMALISATION
# ======================================================================================

reader_norm = reader_matrix.copy()


reader_norm[
    LEAD_ORDER
] = (
    reader_norm[
        LEAD_ORDER
    ]
    .div(
        reader_norm["_raw_total"],
        axis=0
    )
)


reader_norm_sums = (
    reader_norm[
        LEAD_ORDER
    ].sum(
        axis=1
    )
)


if not np.allclose(
    reader_norm_sums.values,
    1.0,
    atol=1e-10
):

    raise RuntimeError(
        "Per-reader gaze normalisation failed."
    )


print("\n" + "=" * 100)
print("PER-READER NORMALISATION")
print("=" * 100)

print(
    "Min normalised row sum:",
    reader_norm_sums.min()
)

print(
    "Max normalised row sum:",
    reader_norm_sums.max()
)

print(
    "\n✓ Every reader × ECG vector independently sums to 1."
)


# ======================================================================================
# 17. DEFINE THE TWO CLEAN POPULATIONS
# ======================================================================================

consultant_reader_ids = sorted(
    reader_groups.loc[
        reader_groups["_group_norm"]
        ==
        "consultant",
        "Respondent_Name"
    ].unique()
)


all_clinician_reader_ids = sorted(
    reader_groups.loc[
        reader_groups["_group_norm"].isin(
            [
                "consultant",
                "fellow",
                "resident"
            ]
        ),
        "Respondent_Name"
    ].unique()
)


if len(
    consultant_reader_ids
) != 9:

    raise RuntimeError(
        "Consultant reader population is not exactly 9."
    )


if len(
    all_clinician_reader_ids
) != 20:

    raise RuntimeError(
        "All-clinician reader population is not exactly 20."
    )


# ======================================================================================
# 18. VERIFY ALL SELECTED CLINICIANS HAVE ALL 10 ECGs
# ======================================================================================

def audit_population_completeness(
    reader_ids,
    population_name
):

    temp = (
        reader_norm
        .reset_index()
    )


    temp = temp[
        temp[
            "Respondent_Name"
        ].isin(
            reader_ids
        )
    ]


    counts = (
        temp
        .groupby(
            "Respondent_Name"
        )["_ecg"]
        .nunique()
    )


    if not (
        counts == 10
    ).all():

        bad = counts[
            counts != 10
        ]

        raise RuntimeError(
            f"{population_name}: incomplete ECG coverage:\n"
            +
            bad.to_string()
        )


    return counts


consultant_complete = audit_population_completeness(
    consultant_reader_ids,
    "Consultant"
)


all_complete = audit_population_completeness(
    all_clinician_reader_ids,
    "All clinicians"
)


print("\n" + "=" * 100)
print("CLINICAL POPULATION COMPLETENESS")
print("=" * 100)

print(
    "Consultants:",
    len(consultant_complete),
    "readers × 10 ECGs"
)

print(
    "All clinicians:",
    len(all_complete),
    "readers × 10 ECGs"
)

print(
    "\n✓ All 9 consultants contributed to all ten ECGs."
)

print(
    "✓ All 20 clinicians contributed to all ten ECGs."
)


# ======================================================================================
# 19. BUILD EQUAL-READER-WEIGHTED TARGETS
# ======================================================================================

reader_norm_reset = (
    reader_norm
    .reset_index()
)


def build_population_target(
    reader_ids,
    population_name
):

    temp = reader_norm_reset[
        reader_norm_reset[
            "Respondent_Name"
        ].isin(
            reader_ids
        )
    ].copy()


    targets = []


    for ecg in ECG_ORDER:

        ecg_temp = temp[
            temp["_ecg"]
            ==
            ecg
        ].copy()


        # There must be exactly one normalised vector per reader.
        if len(ecg_temp) != len(reader_ids):

            raise RuntimeError(
                f"{population_name} / {ecg}: "
                f"expected {len(reader_ids)} readers, found {len(ecg_temp)}."
            )


        if ecg_temp[
            "Respondent_Name"
        ].nunique() != len(reader_ids):

            raise RuntimeError(
                f"{population_name} / {ecg}: duplicate or missing readers."
            )


        matrix = ecg_temp[
            LEAD_ORDER
        ].to_numpy(
            dtype=np.float64
        )


        # Equal reader weighting.
        target = matrix.mean(
            axis=0
        )


        # Defensive renormalisation against floating-point drift.
        target = (
            target
            /
            target.sum()
        )


        targets.append(
            target
        )


    targets = np.stack(
        targets,
        axis=0
    )


    return targets


Y_consultant = build_population_target(
    consultant_reader_ids,
    "Consultant"
).astype(
    np.float32
)


Y_all_clinicians = build_population_target(
    all_clinician_reader_ids,
    "All clinicians"
).astype(
    np.float32
)


if Y_consultant.shape != (
    10,
    12
):

    raise RuntimeError(
        f"Unexpected consultant target shape: {Y_consultant.shape}"
    )


if Y_all_clinicians.shape != (
    10,
    12
):

    raise RuntimeError(
        f"Unexpected all-clinician target shape: {Y_all_clinicians.shape}"
    )


if not np.allclose(
    Y_consultant.sum(axis=1),
    1.0,
    atol=1e-6
):

    raise RuntimeError(
        "Consultant population targets do not sum to one."
    )


if not np.allclose(
    Y_all_clinicians.sum(axis=1),
    1.0,
    atol=1e-6
):

    raise RuntimeError(
        "All-clinician population targets do not sum to one."
    )


# ======================================================================================
# 20. DISPLAY FINAL TARGETS
# ======================================================================================

consultant_df = pd.DataFrame(
    Y_consultant,
    index=ECG_ORDER,
    columns=LEAD_ORDER
)


all_df = pd.DataFrame(
    Y_all_clinicians,
    index=ECG_ORDER,
    columns=LEAD_ORDER
)


print("\n" + "=" * 100)
print("FINAL CONSULTANT GAZE TARGET — 9 READERS")
print("=" * 100)

print(
    consultant_df
    .round(4)
    .to_string()
)


print("\n" + "=" * 100)
print("FINAL ALL-CLINICIAN GAZE TARGET — 20 READERS")
print("=" * 100)

print(
    all_df
    .round(4)
    .to_string()
)


# ======================================================================================
# 21. CONSULTANT vs ALL-CLINICIAN TARGET DIFFERENCE
# ======================================================================================

def cosine_similarity(
    a,
    b
):

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator <= 0:
        raise RuntimeError(
            "Zero norm in cosine similarity."
        )

    return float(
        np.dot(a, b)
        /
        denominator
    )


comparison_rows = []


for i, ecg in enumerate(
    ECG_ORDER
):

    consultant = Y_consultant[i]
    all_clin = Y_all_clinicians[i]


    comparison_rows.append(
        {
            "ECG":
                ecg,

            "cosine_similarity":
                cosine_similarity(
                    consultant,
                    all_clin
                ),

            "L1_distance":
                float(
                    np.sum(
                        np.abs(
                            consultant
                            -
                            all_clin
                        )
                    )
                ),

            "max_abs_lead_difference":
                float(
                    np.max(
                        np.abs(
                            consultant
                            -
                            all_clin
                        )
                    )
                ),
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


print("\n" + "=" * 100)
print("CONSULTANT vs ALL-CLINICIAN TARGET DIFFERENCE")
print("=" * 100)

print(
    comparison_df
    .round(6)
    .to_string(
        index=False
    )
)


# ======================================================================================
# 22. SAVE FINAL TARGETS
# ======================================================================================

TARGET_NPZ = (
    OUT_DIR
    /
    "clean_gaze_targets.npz"
)


np.savez_compressed(

    TARGET_NPZ,

    consultant=
        Y_consultant,

    all_clinicians=
        Y_all_clinicians,

    ecg_order=
        np.asarray(
            ECG_ORDER
        ),

    lead_order=
        np.asarray(
            LEAD_ORDER
        ),
)


# Long population-target table
target_long_rows = []


for ei, ecg in enumerate(
    ECG_ORDER
):

    for li, lead in enumerate(
        LEAD_ORDER
    ):

        target_long_rows.append(
            {
                "ECG":
                    ecg,

                "lead":
                    lead,

                "consultant":
                    float(
                        Y_consultant[
                            ei,
                            li
                        ]
                    ),

                "all_clinicians":
                    float(
                        Y_all_clinicians[
                            ei,
                            li
                        ]
                    ),
            }
        )


pd.DataFrame(
    target_long_rows
).to_csv(

    OUT_DIR
    /
    "clean_gaze_targets_long.csv",

    index=False
)


# Per-reader normalised vectors for audit/reproducibility
reader_norm_export = (
    reader_norm_reset[
        [
            "Respondent_Name",
            "_group_norm",
            "_ecg",
            "_raw_total",
            *LEAD_ORDER
        ]
    ]
    .rename(
        columns={
            "_group_norm":
                "group",

            "_ecg":
                "ECG",

            "_raw_total":
                "raw_total_dwell"
        }
    )
)


reader_norm_export.to_csv(

    OUT_DIR
    /
    "per_reader_normalised_gaze.csv",

    index=False
)


structure_df.to_csv(

    OUT_DIR
    /
    "structural_audit.csv",

    index=False
)


comparison_df.to_csv(

    OUT_DIR
    /
    "population_target_comparison.csv",

    index=False
)


# ======================================================================================
# 23. HASH SAVED TARGET FILE
# ======================================================================================

TARGET_HASH = sha256_file(
    TARGET_NPZ
)


print("\nSaved target file:")
print(TARGET_NPZ)

print("\nTarget SHA256:")
print(TARGET_HASH)


# ======================================================================================
# 24. LOAD BASE MODEL — NO TRAINING
# ======================================================================================

print("\n" + "=" * 100)
print("BASE MODEL AUDIT")
print("=" * 100)


base_model = tf.keras.models.load_model(
    BASE_MODEL_PATH,
    compile=False
)


print(
    "Model name:",
    base_model.name
)

print(
    "Input shape:",
    base_model.input_shape
)

print(
    "Output shape:",
    base_model.output_shape
)

print(
    "Number of layers:",
    len(base_model.layers)
)


# Required trainable layer
try:

    conv10 = base_model.get_layer(
        "conv2d_10"
    )

except Exception as exc:

    raise RuntimeError(
        "Base model does not contain conv2d_10."
    ) from exc


print(
    "\nconv2d_10 output shape:",
    conv10.output.shape
)


# Representation layer
try:

    add3 = base_model.get_layer(
        "add_3"
    )

except Exception as exc:

    raise RuntimeError(
        "Base model does not contain add_3."
    ) from exc


print(
    "add_3 output shape:",
    add3.output.shape
)


final_layer = base_model.layers[-1]


print(
    "\nFinal layer:",
    final_layer.name
)


if hasattr(
    final_layer,
    "activation"
):

    print(
        "Final activation:",
        final_layer.activation.__name__
    )


print(
    "\nBase model SHA256:",
    BASE_HASH
)


# ======================================================================================
# 25. FORWARD PASS — STILL NO TRAINING
# ======================================================================================

print("\n" + "=" * 100)
print("FORWARD-PASS CHECK")
print("=" * 100)


test_output = base_model(
    X_gaze[
        0:1
    ],
    training=False
)


if isinstance(
    test_output,
    (
        list,
        tuple
    )
):

    print(
        "Number of outputs:",
        len(test_output)
    )

    for i, out in enumerate(
        test_output
    ):

        print(
            f"Output {i}:",
            tuple(out.shape)
        )

else:

    print(
        "Output shape:",
        tuple(test_output.shape)
    )


if not np.all(
    np.isfinite(
        np.asarray(
            test_output
        )
    )
):

    raise RuntimeError(
        "Base-model forward pass produced non-finite values."
    )


print(
    "\n✓ Base-model forward pass successful."
)


# ======================================================================================
# 26. SAVE DEFINITIVE STEP-1 MANIFEST
# ======================================================================================

manifest = {

    "created":
        datetime.now().isoformat(),

    "experiment":
        "Definitive clean gaze-aligned ECG experiment",

    "stage":
        "Step 1: frozen inputs and gaze target construction",

    "random_seed":
        SEED,

    "frozen_ecg_input":
        str(V9_PATH),

    "frozen_ecg_sha256":
        V9_HASH,

    "frozen_ecg_shape":
        list(X_gaze.shape),

    "eye_tracking_csv":
        str(GAZE_CSV_PATH),

    "eye_tracking_csv_sha256":
        GAZE_HASH,

    "base_model":
        str(BASE_MODEL_PATH),

    "base_model_sha256":
        BASE_HASH,

    "clean_gaze_targets":
        str(TARGET_NPZ),

    "clean_gaze_targets_sha256":
        TARGET_HASH,

    "ecg_order":
        ECG_ORDER,

    "lead_order":
        LEAD_ORDER,

    "consultant_reader_count":
        9,

    "all_clinician_reader_count":
        20,

    "all_clinician_composition": {
        "consultant":
            9,

        "fellow":
            10,

        "resident":
            1,
    },

    "stimulus_mapping":
        EXACT_STIMULUS_MAP,

    "diagnostic_aoi_mapping":
        BASE_AOI_TO_LEAD,

    "gaze_target_method": (
        "For each reader and ECG, dwell times within the 12 selected "
        "diagnostic-grid AOIs were arranged in fixed standard lead order "
        "and independently normalised to sum to one. Population targets "
        "were then calculated as the arithmetic mean of these normalised "
        "reader vectors, giving every reader equal weight."
    ),

    "static_aoi_rows":
        int(len(static)),

    "selected_diagnostic_rows":
        int(len(lead_rows)),

    "training_performed":
        False,
}


with open(

    OUT_DIR
    /
    "step1_manifest.json",

    "w"

) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ======================================================================================
# 27. FINAL REPORT
# ======================================================================================

print("\n\n" + "=" * 100)
print("STEP 1 COMPLETE — NO TRAINING PERFORMED")
print("=" * 100)


print(
    """
If this cell reaches here without an exception, the clean gaze targets
have been constructed from a fresh kernel using a single immutable pipeline.

PLEASE PASTE BACK THESE FOUR OUTPUT SECTIONS:

1. FINAL 12-LEAD STRUCTURAL AUDIT

2. CLINICAL POPULATION COMPLETENESS

3. CONSULTANT vs ALL-CLINICIAN TARGET DIFFERENCE

4. BASE MODEL AUDIT + FORWARD-PASS CHECK

Do not train anything yet.
"""
)
