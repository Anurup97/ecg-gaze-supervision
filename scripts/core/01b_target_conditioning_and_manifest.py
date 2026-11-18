# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: New training 30.08.ipynb ; code cell 4
# Stage: final clinician-population targets and step1 manifest
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.
# NOTE: Uses reader_groups and reader_matrix from 01a; run via %run -i or same notebook kernel.

# ======================================================================================
# DEFINITIVE CLEAN GAZE EXPERIMENT
# STEP 1 — CONTINUATION AFTER AOI-COVERAGE AUDIT
#
# IMPORTANT METHODOLOGICAL DEFINITION
# -----------------------------------
# Primary gaze target = gaze CONDITIONAL ON THE 12 MODEL-VISIBLE SHORT-LEAD AOIs.
#
# Excluded:
#     - long rhythm-strip grid cells (II-*, V1-*, V5-*, etc.)
#     - Information
#     - non-standard leads such as V3R, V4R, V7, V8
#
# These regions are not represented as separate inputs in X_gaze.
#
# No training occurs here.
# ======================================================================================

import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf


# ======================================================================================
# 1. PRESPECIFIED CLINICAL POPULATIONS
# ======================================================================================

CONSULTANT_GROUPS = {
    "consultant"
}

ALL_CLINICIAN_GROUPS = {
    "consultant",
    "fellow",
    "resident",
}


consultant_reader_ids = sorted(
    reader_groups.loc[
        reader_groups["_group_norm"].isin(
            CONSULTANT_GROUPS
        ),
        "Respondent_Name"
    ].unique()
)


all_clinician_reader_ids = sorted(
    reader_groups.loc[
        reader_groups["_group_norm"].isin(
            ALL_CLINICIAN_GROUPS
        ),
        "Respondent_Name"
    ].unique()
)


if len(consultant_reader_ids) != 9:
    raise RuntimeError(
        f"Expected 9 consultants, found {len(consultant_reader_ids)}."
    )


if len(all_clinician_reader_ids) != 20:
    raise RuntimeError(
        f"Expected 20 clinicians, found {len(all_clinician_reader_ids)}."
    )


print("=" * 100)
print("PRESPECIFIED CLINICAL POPULATIONS")
print("=" * 100)

print("Consultants:    ", len(consultant_reader_ids))
print("All clinicians:", len(all_clinician_reader_ids))

print(
    "\n✓ 9-consultant and 20-clinician populations confirmed."
)


# ======================================================================================
# 2. RESTRICT READER MATRIX BEFORE NORMALISATION
# ======================================================================================

clinical_matrix = (
    reader_matrix
    .reset_index()
)


clinical_matrix = clinical_matrix[
    clinical_matrix[
        "Respondent_Name"
    ].isin(
        all_clinician_reader_ids
    )
].copy()


if len(clinical_matrix) != 20 * 10:

    raise RuntimeError(
        f"Expected 200 clinician×ECG vectors, found {len(clinical_matrix)}."
    )


if clinical_matrix[
    "Respondent_Name"
].nunique() != 20:

    raise RuntimeError(
        "Expected 20 unique clinicians."
    )


# ======================================================================================
# 3. VERIFY COMPLETE ECG COVERAGE
# ======================================================================================

clinical_completeness = (
    clinical_matrix
    .groupby(
        [
            "Respondent_Name",
            "_group_norm"
        ]
    )["_ecg"]
    .nunique()
    .reset_index(
        name="n_ECGs"
    )
)


print("\n" + "=" * 100)
print("CLINICAL POPULATION COMPLETENESS")
print("=" * 100)

print(
    clinical_completeness.to_string(
        index=False
    )
)


if not (
    clinical_completeness["n_ECGs"]
    ==
    10
).all():

    raise RuntimeError(
        "At least one included clinician does not have all ten ECGs."
    )


# ======================================================================================
# 4. VERIFY POSITIVE MODEL-VISIBLE DWELL
# ======================================================================================

zero_clinical = clinical_matrix[
    clinical_matrix["_raw_total"]
    <=
    0
]


if len(zero_clinical) > 0:

    raise RuntimeError(
        "\nAn INCLUDED clinician has zero model-visible 12-lead dwell:\n"
        +
        zero_clinical[
            [
                "Respondent_Name",
                "_group_norm",
                "_ecg",
                "_raw_total"
            ]
        ].to_string(index=False)
    )


print(
    "\n✓ All 200 included clinician×ECG vectors have positive "
    "dwell within the model-visible 12 leads."
)


# ======================================================================================
# 5. NORMALISE EACH READER × ECG INDEPENDENTLY
# ======================================================================================

clinical_norm = clinical_matrix.copy()


clinical_norm[
    LEAD_ORDER
] = clinical_norm[
    LEAD_ORDER
].div(
    clinical_norm["_raw_total"],
    axis=0
)


normalised_sums = clinical_norm[
    LEAD_ORDER
].sum(
    axis=1
)


if not np.allclose(
    normalised_sums,
    1.0,
    atol=1e-10
):

    raise RuntimeError(
        "Per-reader normalisation failed."
    )


print("\n" + "=" * 100)
print("PER-READER MODEL-VISIBLE GAZE NORMALISATION")
print("=" * 100)

print(
    "Minimum vector sum:",
    normalised_sums.min()
)

print(
    "Maximum vector sum:",
    normalised_sums.max()
)

print(
    "\n✓ Each clinician×ECG 12-lead gaze vector independently sums to 1."
)


# ======================================================================================
# 6. RECORD HOW MUCH WHOLE-IMAGE STATIC GAZE IS REPRESENTED
#
# This is an audit/limitation metric, not used to create the target.
# ======================================================================================

clinical_static = static[
    static["_group_norm"].isin(
        ALL_CLINICIAN_GROUPS
    )
].copy()


clinical_static[
    "_dwell_numeric"
] = pd.to_numeric(
    clinical_static[
        "Time_spent_G"
    ],
    errors="coerce"
).fillna(
    0.0
)


total_static = (
    clinical_static
    .groupby(
        [
            "Respondent_Name",
            "_group_norm",
            "_ecg"
        ]
    )["_dwell_numeric"]
    .sum()
    .rename(
        "whole_static_dwell"
    )
)


selected_static = (
    clinical_static[
        clinical_static["_lead"].notna()
    ]
    .groupby(
        [
            "Respondent_Name",
            "_group_norm",
            "_ecg"
        ]
    )["_dwell_numeric"]
    .sum()
    .rename(
        "model_visible_dwell"
    )
)


coverage_df = pd.concat(
    [
        total_static,
        selected_static
    ],
    axis=1
).fillna(
    0
).reset_index()


coverage_df[
    "excluded_dwell"
] = (
    coverage_df[
        "whole_static_dwell"
    ]
    -
    coverage_df[
        "model_visible_dwell"
    ]
)


coverage_df[
    "model_visible_fraction"
] = (
    coverage_df[
        "model_visible_dwell"
    ]
    /
    coverage_df[
        "whole_static_dwell"
    ]
)


coverage_df[
    "excluded_fraction"
] = (
    coverage_df[
        "excluded_dwell"
    ]
    /
    coverage_df[
        "whole_static_dwell"
    ]
)


print("\n" + "=" * 100)
print("MODEL-VISIBLE GAZE COVERAGE — AUDIT ONLY")
print("=" * 100)

print(
    coverage_df[
        [
            "model_visible_fraction",
            "excluded_fraction"
        ]
    ]
    .describe()
    .round(4)
    .to_string()
)


print(
    "\nMean model-visible fraction:",
    round(
        coverage_df[
            "model_visible_fraction"
        ].mean(),
        4
    )
)

print(
    "Median model-visible fraction:",
    round(
        coverage_df[
            "model_visible_fraction"
        ].median(),
        4
    )
)


# ======================================================================================
# 7. FUNCTION TO BUILD EQUAL-READER TARGET
# ======================================================================================

def build_target(
    reader_ids,
    population_name
):

    pop = clinical_norm[
        clinical_norm[
            "Respondent_Name"
        ].isin(
            reader_ids
        )
    ].copy()


    targets = []


    for ecg in ECG_ORDER:

        temp = pop[
            pop["_ecg"]
            ==
            ecg
        ].copy()


        if len(temp) != len(reader_ids):

            raise RuntimeError(
                f"{population_name}/{ecg}: "
                f"expected {len(reader_ids)} vectors, found {len(temp)}."
            )


        if temp[
            "Respondent_Name"
        ].nunique() != len(reader_ids):

            raise RuntimeError(
                f"{population_name}/{ecg}: duplicate/missing reader."
            )


        matrix = temp[
            LEAD_ORDER
        ].to_numpy(
            dtype=np.float64
        )


        # Equal reader weighting
        target = matrix.mean(
            axis=0
        )


        # Defensive numerical renormalisation
        target /= target.sum()


        targets.append(
            target
        )


    return np.stack(
        targets,
        axis=0
    ).astype(
        np.float32
    )


# ======================================================================================
# 8. BUILD BOTH DEFINITIVE TARGETS
# ======================================================================================

Y_consultant = build_target(
    consultant_reader_ids,
    "Consultant"
)


Y_all_clinicians = build_target(
    all_clinician_reader_ids,
    "All clinicians"
)


if Y_consultant.shape != (
    10,
    12
):

    raise RuntimeError(
        f"Bad consultant target shape: {Y_consultant.shape}"
    )


if Y_all_clinicians.shape != (
    10,
    12
):

    raise RuntimeError(
        f"Bad all-clinician target shape: {Y_all_clinicians.shape}"
    )


if not np.allclose(
    Y_consultant.sum(axis=1),
    1.0,
    atol=1e-6
):

    raise RuntimeError(
        "Consultant targets do not sum to 1."
    )


if not np.allclose(
    Y_all_clinicians.sum(axis=1),
    1.0,
    atol=1e-6
):

    raise RuntimeError(
        "All-clinician targets do not sum to 1."
    )


print("\n" + "=" * 100)
print("FINAL GAZE TARGET SHAPES")
print("=" * 100)

print(
    "Consultant:",
    Y_consultant.shape
)

print(
    "All clinicians:",
    Y_all_clinicians.shape
)

print(
    "\n✓ Definitive equal-reader targets constructed."
)


# ======================================================================================
# 9. DISPLAY TARGETS
# ======================================================================================

consultant_df = pd.DataFrame(
    Y_consultant,
    index=ECG_ORDER,
    columns=LEAD_ORDER
)


all_clinician_df = pd.DataFrame(
    Y_all_clinicians,
    index=ECG_ORDER,
    columns=LEAD_ORDER
)


print("\n" + "=" * 100)
print("CONSULTANT TARGET — MODEL-VISIBLE 12-LEAD GAZE")
print("=" * 100)

print(
    consultant_df
    .round(4)
    .to_string()
)


print("\n" + "=" * 100)
print("ALL-CLINICIAN TARGET — MODEL-VISIBLE 12-LEAD GAZE")
print("=" * 100)

print(
    all_clinician_df
    .round(4)
    .to_string()
)


# ======================================================================================
# 10. COMPARE POPULATION TARGETS
# ======================================================================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator <= 0:
        raise RuntimeError(
            "Zero vector in cosine calculation."
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

    c = Y_consultant[i]
    a = Y_all_clinicians[i]


    comparison_rows.append(
        {
            "ECG":
                ecg,

            "cosine_similarity":
                cosine_similarity(
                    c,
                    a
                ),

            "L1_distance":
                float(
                    np.sum(
                        np.abs(
                            c - a
                        )
                    )
                ),

            "max_abs_lead_difference":
                float(
                    np.max(
                        np.abs(
                            c - a
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
# 11. SAVE TARGETS + AUDITS
# ======================================================================================

OUT_DIR = Path(
    "definitive_clean_gaze"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TARGET_PATH = (
    OUT_DIR
    /
    "clean_gaze_targets.npz"
)


np.savez_compressed(

    TARGET_PATH,

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


# Long-format targets
long_rows = []


for ei, ecg in enumerate(
    ECG_ORDER
):

    for li, lead in enumerate(
        LEAD_ORDER
    ):

        long_rows.append(
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
    long_rows
).to_csv(
    OUT_DIR
    /
    "clean_gaze_targets_long.csv",
    index=False
)


# Per-reader normalised gaze
clinical_norm[
    [
        "Respondent_Name",
        "_group_norm",
        "_ecg",
        "_raw_total",
        *LEAD_ORDER
    ]
].to_csv(
    OUT_DIR
    /
    "per_reader_model_visible_gaze.csv",
    index=False
)


coverage_df.to_csv(
    OUT_DIR
    /
    "model_visible_gaze_coverage.csv",
    index=False
)


comparison_df.to_csv(
    OUT_DIR
    /
    "population_target_comparison.csv",
    index=False
)


# ======================================================================================
# 12. HASH HELPER
# ======================================================================================

def sha256_file_local(path):

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

            h.update(block)

    return h.hexdigest()


TARGET_HASH = sha256_file_local(
    TARGET_PATH
)


print(
    "\nTarget SHA256:"
)

print(
    TARGET_HASH
)


# ======================================================================================
# 13. BASE MODEL AUDIT
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


conv10 = base_model.get_layer(
    "conv2d_10"
)


add3 = base_model.get_layer(
    "add_3"
)


print(
    "\nconv2d_10 output shape:",
    conv10.output.shape
)

print(
    "add_3 output shape:",
    add3.output.shape
)


final_layer = base_model.layers[
    -1
]


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


# ======================================================================================
# 14. FORWARD PASS — NO TRAINING
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


print(
    "Output shape:",
    tuple(
        test_output.shape
    )
)


if not np.all(
    np.isfinite(
        test_output.numpy()
    )
):

    raise RuntimeError(
        "Forward pass produced non-finite values."
    )


print(
    "✓ Base-model forward pass successful."
)


# ======================================================================================
# 15. SAVE FINAL STEP-1 MANIFEST
# ======================================================================================

manifest = {

    "created":
        datetime.now().isoformat(),

    "experiment":
        "Definitive clean gaze-aligned ECG experiment",

    "stage":
        "Step 1 complete",

    "frozen_ecg_input":
        str(
            V9_PATH
        ),

    "frozen_ecg_sha256":
        V9_HASH,

    "eye_tracking_csv":
        str(
            GAZE_CSV_PATH
        ),

    "eye_tracking_csv_sha256":
        GAZE_HASH,

    "base_model":
        str(
            BASE_MODEL_PATH
        ),

    "base_model_sha256":
        BASE_HASH,

    "clean_gaze_targets":
        str(
            TARGET_PATH
        ),

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

    "gaze_target_scope":
        (
            "Conditional distribution of dwell time within the 12 "
            "model-visible short diagnostic lead AOIs."
        ),

    "excluded_aoi_scope":
        (
            "Long rhythm-strip grid cells, Information AOIs, and "
            "non-standard additional leads not represented in the "
            "12-channel model input."
        ),

    "gaze_target_method":
        (
            "For each included clinician and ECG, dwell time within "
            "the 12 model-visible diagnostic lead AOIs was arranged "
            "in fixed standard lead order and normalised independently "
            "to sum to one. Population targets were then calculated "
            "by equally averaging the reader-level normalised vectors."
        ),

    "mean_model_visible_gaze_fraction":
        float(
            coverage_df[
                "model_visible_fraction"
            ].mean()
        ),

    "median_model_visible_gaze_fraction":
        float(
            coverage_df[
                "model_visible_fraction"
            ].median()
        ),

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
# 16. FINISH
# ======================================================================================

print("\n\n" + "=" * 100)
print("STEP 1 COMPLETE — DEFINITIVE TARGETS SAVED — NO TRAINING")
print("=" * 100)

print(
    """
Please paste back:

1. CLINICAL POPULATION COMPLETENESS
2. PER-READER MODEL-VISIBLE GAZE NORMALISATION
3. CONSULTANT vs ALL-CLINICIAN TARGET DIFFERENCE
4. BASE MODEL AUDIT
5. FORWARD-PASS CHECK

If those pass, the next step is the prespecified 5-fold
held-out-ECG gaze-alignment experiment.
"""
)
