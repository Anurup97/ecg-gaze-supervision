# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 9
# Stage: LUDB records and outcome provenance
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 5A — LUDB PROVENANCE / SIGNAL / DIAGNOSTIC-ENDPOINT AUDIT
#
# READ ONLY.
#
# NO MODEL INFERENCE.
# NO TRAINING.
# NO DIAGNOSTIC LABEL MAPPING.
# NO RESAMPLING FOR MODEL USE.
#
# Questions:
#   1. Which LUDB release/copy is present locally?
#   2. Are all 200 records present?
#   3. What are the native signal geometry, lead order and sampling frequency?
#   4. What diagnostic information is actually supplied?
#   5. Is there a direct, defensible correspondence with any of the frozen
#      PTB-XL model's 23 output classes?
#   6. What did our historical LUDB code actually do?
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError as exc:
    raise ImportError(
        "wfdb is required. Install with:\n\n"
        "    pip install wfdb\n"
    ) from exc


# ======================================================================================
# 1. FROZEN MODEL VOCABULARY — FOR COMPARISON ONLY
#
# IMPORTANT:
# We do NOT map LUDB diagnoses to these classes in this audit.
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


EXPECTED_NATIVE_LEADS = [
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
# 2. LOCATE LUDB
#
# Prefer a directory containing:
#     ludb.csv
#     data/
#
# Do not silently choose between multiple copies.
# ======================================================================================

csv_candidates = list(
    Path(".").glob(
        "**/ludb.csv"
    )
)


candidate_roots = []


for csv_path in csv_candidates:

    root = csv_path.parent

    data_dir = (
        root
        /
        "data"
    )

    if data_dir.exists():

        candidate_roots.append(
            root
        )


candidate_roots = sorted(
    set(
        candidate_roots
    ),
    key=lambda p: str(p)
)


print("=" * 100)
print("LUDB DATASET SEARCH")
print("=" * 100)


if len(candidate_roots) == 0:

    raise FileNotFoundError(
        "\nCould not find LUDB locally.\n\n"
        "Expected something like:\n"
        "    ludb/\n"
        "        ludb.csv\n"
        "        data/\n"
    )


print(
    "Candidate LUDB roots:"
)


for i, root in enumerate(
    candidate_roots,
    start=1
):

    print(
        f"  {i}. {root}"
    )


if len(candidate_roots) != 1:

    raise RuntimeError(
        "\nMore than one LUDB copy was found.\n"
        "Do not automatically choose between them."
    )


LUDB_ROOT = candidate_roots[0]

LUDB_CSV = (
    LUDB_ROOT
    /
    "ludb.csv"
)

DATA_DIR = (
    LUDB_ROOT
    /
    "data"
)


print(
    "\nSelected root:"
)

print(
    LUDB_ROOT
)


# ======================================================================================
# 3. TOP-LEVEL RELEASE FILES
# ======================================================================================

print("\n" + "=" * 100)
print("LUDB TOP-LEVEL FILE AUDIT")
print("=" * 100)


for filename in [
    "README",
    "RECORDS",
    "ANNOTATORS",
    "SHA256SUMS.txt",
    "LICENSE.txt",
    "ludb.csv",
]:

    path = (
        LUDB_ROOT
        /
        filename
    )

    print(
        f"{filename:20s}",
        "FOUND"
        if path.exists()
        else
        "MISSING"
    )


# ======================================================================================
# 4. LOAD ludb.csv WITHOUT ASSUMING ITS SCHEMA
# ======================================================================================

ludb = pd.read_csv(
    LUDB_CSV
)


print("\n" + "=" * 100)
print("LUDB.CSV STRUCTURE")
print("=" * 100)


print(
    "Shape:",
    ludb.shape
)


print(
    "\nColumns:"
)


for i, column in enumerate(
    ludb.columns
):

    print(
        f"{i:02d}  {column}"
    )


print(
    "\nFirst 5 rows:"
)


print(
    ludb.head(
        5
    ).to_string(
        index=False
    )
)


# ======================================================================================
# 5. IDENTIFY RECORD FILES
#
# WFDB headers are the safest source of record identity.
# ======================================================================================

header_paths = sorted(
    DATA_DIR.glob(
        "*.hea"
    )
)


record_names = [
    path.stem
    for path in header_paths
]


print("\n" + "=" * 100)
print("WFDB RECORD COUNT")
print("=" * 100)


print(
    "Header files:",
    len(
        header_paths
    )
)


print(
    "Unique record names:",
    len(
        set(
            record_names
        )
    )
)


if len(
    header_paths
) != 200:

    print(
        "\nWARNING:"
        f" official LUDB contains 200 records, but this copy has {len(header_paths)} headers."
    )


if len(
    set(
        record_names
    )
) != len(
    record_names
):

    raise RuntimeError(
        "Duplicate WFDB record names detected."
    )


# ======================================================================================
# 6. READ ALL RECORD HEADERS / SIGNAL GEOMETRY
#
# We load each record to verify:
#     n samples
#     n leads
#     fs
#     lead order
#     finiteness
#
# This is READ-ONLY.
# ======================================================================================

geometry_rows = []

header_comment_rows = []


for i, record_name in enumerate(
    record_names
):

    record_path = (
        DATA_DIR
        /
        record_name
    )


    signal, fields = wfdb.rdsamp(
        str(
            record_path
        )
    )


    leads = [
        str(x).upper()
        for x in fields.get(
            "sig_name",
            []
        )
    ]


    comments = fields.get(
        "comments",
        []
    )


    geometry_rows.append(
        {
            "record":
                record_name,

            "n_samples":
                int(
                    signal.shape[0]
                ),

            "n_leads":
                int(
                    signal.shape[1]
                ),

            "fs":
                float(
                    fields.get(
                        "fs",
                        np.nan
                    )
                ),

            "lead_order":
                " | ".join(
                    leads
                ),

            "finite":
                bool(
                    np.all(
                        np.isfinite(
                            signal
                        )
                    )
                ),

            "min_value":
                float(
                    np.min(
                        signal
                    )
                ),

            "max_value":
                float(
                    np.max(
                        signal
                    )
                ),
        }
    )


    for comment in comments:

        header_comment_rows.append(
            {
                "record":
                    record_name,

                "comment":
                    str(
                        comment
                    ),
            }
        )


    if (
        (i + 1) % 50 == 0
        or
        (i + 1) == len(
            record_names
        )
    ):

        print(
            f"Audited {i + 1}/{len(record_names)} records"
        )


geometry_df = pd.DataFrame(
    geometry_rows
)


comments_df = pd.DataFrame(
    header_comment_rows
)


print("\n" + "=" * 100)
print("NATIVE LUDB SIGNAL GEOMETRY")
print("=" * 100)


print(
    "Unique sampling frequencies:"
)

print(
    geometry_df[
        "fs"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    "\nUnique sample lengths:"
)

print(
    geometry_df[
        "n_samples"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    "\nUnique lead counts:"
)

print(
    geometry_df[
        "n_leads"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    "\nUnique lead orders:"
)

print(
    geometry_df[
        "lead_order"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nAll finite:",
    bool(
        geometry_df[
            "finite"
        ].all()
    )
)


# Strong geometry checks
if not (
    geometry_df[
        "n_leads"
    ]
    ==
    12
).all():

    raise RuntimeError(
        "At least one LUDB record does not have 12 leads."
    )


if not (
    geometry_df[
        "fs"
    ]
    ==
    500.0
).all():

    raise RuntimeError(
        "At least one LUDB record is not sampled at 500 Hz."
    )


if not (
    geometry_df[
        "n_samples"
    ]
    ==
    5000
).all():

    raise RuntimeError(
        "At least one LUDB record is not exactly 10 seconds / 5000 samples."
    )


expected_lead_string = " | ".join(
    EXPECTED_NATIVE_LEADS
)


if not (
    geometry_df[
        "lead_order"
    ]
    ==
    expected_lead_string
).all():

    raise RuntimeError(
        "\nAt least one LUDB record has unexpected lead order."
    )


if not geometry_df[
    "finite"
].all():

    raise RuntimeError(
        "Non-finite waveform values detected."
    )


print(
    "\n✓ All records are native 500-Hz, 5000-sample, 12-lead ECGs "
    "in the expected lead order."
)


# ======================================================================================
# 7. MODEL-GEOMETRY IMPLICATION
#
# NO RESAMPLING IS ACTUALLY PERFORMED HERE.
# ======================================================================================

print("\n" + "=" * 100)
print("MODEL-GEOMETRY COMPATIBILITY")
print("=" * 100)


print(
    "Native LUDB record:"
)

print(
    "    (12 leads, 5000 samples @ 500 Hz)"
)


print(
    "\nFrozen PTB-XL model expects:"
)

print(
    "    (12 leads, 1000 samples @ 100 Hz, 1 channel)"
)


print(
    "\nTherefore a predefined 500 -> 100 Hz resampling operation "
    "would be required before any model inference."
)


print(
    "\nNO RESAMPLING WAS PERFORMED IN THIS AUDIT."
)


# ======================================================================================
# 8. SUMMARISE HEADER COMMENTS
#
# This lets us see exactly how diagnosis/rhythm metadata are written.
# ======================================================================================

print("\n" + "=" * 100)
print("WFDB HEADER COMMENT EXAMPLES")
print("=" * 100)


if len(
    comments_df
) == 0:

    print(
        "No WFDB comments found."
    )

else:

    print(
        comments_df.head(
            80
        ).to_string(
            index=False
        )
    )


# Frequency of comment prefixes
if len(
    comments_df
) > 0:

    comments_df[
        "prefix"
    ] = comments_df[
        "comment"
    ].astype(
        str
    ).str.split(
        ":",
        n=1
    ).str[
        0
    ].str.strip()


    print(
        "\nComment-prefix counts:"
    )


    print(
        comments_df[
            "prefix"
        ]
        .value_counts()
        .head(
            30
        )
        .to_string()
    )


# ======================================================================================
# 9. AUTOMATIC ludb.csv COLUMN AUDIT
#
# For categorical/string-like columns, print unique values and frequencies.
#
# We deliberately DO NOT interpret these yet.
# ======================================================================================

print("\n" + "=" * 100)
print("LUDB.CSV CATEGORICAL / DIAGNOSTIC CONTENT")
print("=" * 100)


categorical_columns = []


for column in ludb.columns:

    series = ludb[
        column
    ]


    # Candidate categorical/diagnosis-like columns:
    # object/string OR modest number of unique values.
    nunique = series.nunique(
        dropna=True
    )


    if (
        pd.api.types.is_object_dtype(
            series
        )
        or
        pd.api.types.is_string_dtype(
            series
        )
        or
        nunique <= 50
    ):

        categorical_columns.append(
            column
        )


for column in categorical_columns:

    series = ludb[
        column
    ]


    print(
        "\n" +
        "-" * 100
    )

    print(
        f"COLUMN: {column}"
    )


    print(
        "Missing:",
        int(
            series.isna().sum()
        )
    )


    print(
        "Unique non-missing:",
        int(
            series.nunique(
                dropna=True
            )
        )
    )


    value_counts = (
        series.astype(
            "string"
        )
        .fillna(
            "<NA>"
        )
        .value_counts(
            dropna=False
        )
    )


    # Prevent pathological output if an unexpected free-text column exists.
    print(
        value_counts.head(
            100
        ).to_string()
    )


# ======================================================================================
# 10. SEARCH COLUMN NAMES FOR LIKELY CLINICAL ENDPOINTS
# ======================================================================================

clinical_keywords = [
    "diagn",
    "rhythm",
    "block",
    "conduction",
    "ische",
    "infar",
    "scar",
    "repolar",
    "axis",
    "hyper",
    "pace",
    "extra",
    "abnormal",
]


likely_clinical_columns = []


for column in ludb.columns:

    normalized = str(
        column
    ).lower()


    if any(
        keyword in normalized
        for keyword in clinical_keywords
    ):

        likely_clinical_columns.append(
            column
        )


print("\n" + "=" * 100)
print("LIKELY CLINICAL-ENDPOINT COLUMNS")
print("=" * 100)


if likely_clinical_columns:

    for column in likely_clinical_columns:

        print(
            column
        )

else:

    print(
        "NONE IDENTIFIED FROM COLUMN NAMES — inspect the categorical output above."
    )


# ======================================================================================
# 11. SEARCH EXACT MODEL-CLASS TOKENS IN LUDB METADATA
#
# IMPORTANT:
# Exact lexical overlap is NOT proof of label equivalence.
# This is merely an audit of whether the PTB-XL abbreviations themselves
# occur anywhere in LUDB metadata.
# ======================================================================================

ludb_text = "\n".join(
    ludb.astype(
        str
    ).fillna(
        ""
    ).agg(
        " | ".join,
        axis=1
    ).tolist()
)


comment_text = "\n".join(
    comments_df[
        "comment"
    ].astype(
        str
    ).tolist()
) if len(
    comments_df
) else ""


combined_metadata_text = (
    ludb_text
    +
    "\n"
    +
    comment_text
).upper()


lexical_rows = []


for class_name in MODEL_CLASSES:

    literal_count = combined_metadata_text.count(
        class_name.upper()
    )


    lexical_rows.append(
        {
            "PTBXL_model_class":
                class_name,

            "literal_occurrences_in_LUDB_metadata":
                literal_count,
        }
    )


lexical_df = pd.DataFrame(
    lexical_rows
)


print("\n" + "=" * 100)
print("EXACT PTB-XL CLASS-TOKEN OVERLAP")
print("=" * 100)


print(
    lexical_df.to_string(
        index=False
    )
)


print(
    "\nIMPORTANT: lexical overlap does NOT establish diagnostic equivalence."
)


# ======================================================================================
# 12. IDENTIFY POSSIBLE RECORD-ID COLUMN IN ludb.csv
# ======================================================================================

record_id_candidates = []


for column in ludb.columns:

    values = set(
        ludb[
            column
        ]
        .dropna()
        .astype(
            str
        )
    )


    overlap = len(
        values
        &
        set(
            record_names
        )
    )


    if overlap > 0:

        record_id_candidates.append(
            {
                "column":
                    column,

                "matching_record_names":
                    overlap,

                "column_unique_values":
                    len(
                        values
                    ),
            }
        )


record_id_df = pd.DataFrame(
    record_id_candidates
)


print("\n" + "=" * 100)
print("LUDB.CSV ↔ WFDB RECORD-ID MATCH")
print("=" * 100)


if len(
    record_id_df
) == 0:

    print(
        "No direct string-identical record-ID column found."
    )

else:

    print(
        record_id_df
        .sort_values(
            "matching_record_names",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ======================================================================================
# 13. PATIENT / RECORD UNIQUENESS AUDIT
#
# We do not assume whether each record = one patient.
# ======================================================================================

patient_candidates = [
    column
    for column in ludb.columns
    if any(
        token in str(
            column
        ).lower()
        for token in [
            "patient",
            "subject",
            "id",
        ]
    )
]


print("\n" + "=" * 100)
print("POSSIBLE PATIENT / SUBJECT IDENTIFIERS")
print("=" * 100)


if len(
    patient_candidates
) == 0:

    print(
        "No obvious patient/subject ID column by name."
    )

else:

    for column in patient_candidates:

        print(
            f"{column}: "
            f"{ludb[column].nunique(dropna=True)} unique / "
            f"{len(ludb)} rows"
        )


# ======================================================================================
# 14. HISTORICAL LOCAL-CODE SEARCH
#
# We want to know what the OLD LUDB evaluation actually did.
#
# Search only text-like coding files beneath the current project.
# We do NOT execute them.
# ======================================================================================

SEARCH_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".txt",
    ".md",
}


SEARCH_TERMS = [
    "ludb",
    "by_name",
    "skip_mismatch",
    "gaze_model.h5",
    "gaze_finetuned.h5",
    "gaze_model_3.h5",
    "resample",
    "downsample",
]


historical_hits = []


for path in Path(".").rglob("*"):

    if not path.is_file():
        continue


    if path.suffix.lower() not in SEARCH_EXTENSIONS:
        continue


    # Avoid generated outputs / environments.
    path_text = str(
        path
    ).lower()


    if any(
        excluded in path_text
        for excluded in [
            ".venv",
            "site-packages",
            "__pycache__",
            "node_modules",
            "definitive_clean_gaze",
        ]
    ):
        continue


    try:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:

        continue


    text_lower = text.lower()


    matched_terms = [
        term
        for term in SEARCH_TERMS
        if term.lower() in text_lower
    ]


    if matched_terms:

        historical_hits.append(
            {
                "file":
                    str(
                        path
                    ),

                "matched_terms":
                    ", ".join(
                        matched_terms
                    ),
            }
        )


historical_hits_df = pd.DataFrame(
    historical_hits
)


print("\n" + "=" * 100)
print("HISTORICAL LUDB-RELATED CODE FILES")
print("=" * 100)


if len(
    historical_hits_df
) == 0:

    print(
        "No historical LUDB-related code/notebook files found."
    )

else:

    print(
        historical_hits_df.to_string(
            index=False
        )
    )


# ======================================================================================
# 15. PRINT TARGETED CODE SNIPPETS
#
# Only snippets around important LUDB/preprocessing/loading terms.
# Maximum output is deliberately capped.
# ======================================================================================

print("\n" + "=" * 100)
print("HISTORICAL LUDB CODE SNIPPETS")
print("=" * 100)


MAX_FILES = 10
MAX_MATCHES_PER_FILE = 12


if len(
    historical_hits_df
) == 0:

    print(
        "NONE"
    )

else:

    for file_number, file_path in enumerate(
        historical_hits_df[
            "file"
        ].tolist()[
            :MAX_FILES
        ],
        start=1
    ):

        path = Path(
            file_path
        )


        try:

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        except Exception:

            continue


        # For .ipynb the raw JSON is still searchable.
        lines = text.splitlines()


        print(
            "\n" +
            "#" * 100
        )

        print(
            f"FILE {file_number}: {path}"
        )

        print(
            "#" * 100
        )


        match_count = 0


        for line_number, line in enumerate(
            lines
        ):

            line_lower = line.lower()


            if not any(
                term.lower() in line_lower
                for term in SEARCH_TERMS
            ):

                continue


            start = max(
                0,
                line_number - 2
            )

            end = min(
                len(
                    lines
                ),
                line_number + 3
            )


            print(
                f"\n--- around line {line_number + 1} ---"
            )


            for j in range(
                start,
                end
            ):

                # Cap individual very-long notebook JSON lines.
                display_line = lines[
                    j
                ]


                if len(
                    display_line
                ) > 500:

                    display_line = (
                        display_line[
                            :500
                        ]
                        +
                        " ... [TRUNCATED]"
                    )


                print(
                    f"{j + 1:05d}: {display_line}"
                )


            match_count += 1


            if match_count >= MAX_MATCHES_PER_FILE:

                print(
                    "\n[Further matches in this file omitted]"
                )

                break


# ======================================================================================
# 16. SAVE AUDIT TABLES
# ======================================================================================

AUDIT_DIR = Path(
    "definitive_clean_gaze/ludb_audit"
)


AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


geometry_df.to_csv(
    AUDIT_DIR
    /
    "ludb_signal_geometry.csv",
    index=False
)


comments_df.to_csv(
    AUDIT_DIR
    /
    "ludb_header_comments.csv",
    index=False
)


lexical_df.to_csv(
    AUDIT_DIR
    /
    "ludb_ptbxl_literal_class_overlap.csv",
    index=False
)


if len(
    historical_hits_df
) > 0:

    historical_hits_df.to_csv(
        AUDIT_DIR
        /
        "historical_ludb_code_files.csv",
        index=False
    )


if len(
    record_id_df
) > 0:

    record_id_df.to_csv(
        AUDIT_DIR
        /
        "ludb_record_id_candidates.csv",
        index=False
    )


# Save raw ludb.csv column inventory.
column_inventory = []


for column in ludb.columns:

    column_inventory.append(
        {
            "column":
                column,

            "dtype":
                str(
                    ludb[
                        column
                    ].dtype
                ),

            "missing_n":
                int(
                    ludb[
                        column
                    ].isna().sum()
                ),

            "unique_nonmissing_n":
                int(
                    ludb[
                        column
                    ].nunique(
                        dropna=True
                    )
                ),
        }
    )


pd.DataFrame(
    column_inventory
).to_csv(
    AUDIT_DIR
    /
    "ludb_column_inventory.csv",
    index=False
)


# ======================================================================================
# 17. AUDIT MANIFEST
# ======================================================================================

manifest = {

    "stage":
        "Step 5A LUDB read-only provenance and endpoint audit",

    "LUDB_root":
        str(
            LUDB_ROOT
        ),

    "ludb_csv_shape":
        [
            int(
                ludb.shape[0]
            ),
            int(
                ludb.shape[1]
            ),
        ],

    "WFDB_record_count":
        int(
            len(
                record_names
            )
        ),

    "sampling_frequencies":
        sorted(
            geometry_df[
                "fs"
            ].unique().tolist()
        ),

    "sample_lengths":
        sorted(
            geometry_df[
                "n_samples"
            ].unique().tolist()
        ),

    "lead_counts":
        sorted(
            geometry_df[
                "n_leads"
            ].unique().tolist()
        ),

    "lead_orders":
        sorted(
            geometry_df[
                "lead_order"
            ].unique().tolist()
        ),

    "frozen_classifier_expected_sampling_frequency_Hz":
        100,

    "frozen_classifier_expected_samples":
        1000,

    "LUDB_to_classifier_resampling_required":
        True,

    "resampling_performed_in_this_step":
        False,

    "diagnostic_mapping_performed":
        False,

    "model_inference_performed":
        False,

    "training_performed":
        False,

    "historical_code_files_found":
        (
            historical_hits_df[
                "file"
            ].tolist()
            if len(
                historical_hits_df
            ) > 0
            else []
        ),
}


with open(
    AUDIT_DIR
    /
    "ludb_audit_manifest.json",
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ======================================================================================
# 18. FINAL REPORT
# ======================================================================================

print("\n\n" + "=" * 100)
print("STEP 5A COMPLETE — READ ONLY")
print("=" * 100)


print(
    """
Please paste back these FOUR sections:

1. LUDB.CSV STRUCTURE
   - shape
   - column names
   - first five rows

2. NATIVE LUDB SIGNAL GEOMETRY

3. LUDB.CSV CATEGORICAL / DIAGNOSTIC CONTENT
   This may be long — most importantly include any columns referring to:
       rhythm
       conduction abnormalities
       myocardial infarction / ischemia / scar
       pacemaker
       hypertrophy
       other diagnoses

4. HISTORICAL LUDB CODE FILES + relevant HISTORICAL LUDB CODE SNIPPETS

DO NOT run LUDB through the three models yet.

Once we see those outputs, we will decide whether LUDB supports:
    A. a legitimate mapped diagnostic evaluation,
    B. a narrower endpoint such as normal/abnormal,
or
    C. no defensible classifier endpoint at all.
"""
)
