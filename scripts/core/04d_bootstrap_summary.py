# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 8
# Stage: per-class bootstrap summary
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 4D — SUMMARISE SAVED PER-CLASS PATIENT-BOOTSTRAP RESULTS
#
# NO TRAINING.
# NO INFERENCE.
# ======================================================================================

from pathlib import Path

import numpy as np
import pandas as pd


PATH = Path(
    "definitive_clean_gaze/ptbxl_evaluation/"
    "ptbxl_patient_cluster_bootstrap_per_class.csv"
)


df = pd.read_csv(PATH)


# Restrict primary interpretation to the two gaze-vs-base comparisons.
primary = df[
    df["comparison"].isin(
        [
            "consultant_vs_base",
            "all_clinicians_vs_base",
        ]
    )
].copy()


# CI classification:
#
# CI entirely > 0  -> bootstrap-supported improvement
# CI entirely < 0  -> bootstrap-supported deterioration
# otherwise        -> uncertain
def classify_effect(row):

    if row["CI_low"] > 0:
        return "improved"

    if row["CI_high"] < 0:
        return "worsened"

    return "uncertain"


primary["bootstrap_direction"] = primary.apply(
    classify_effect,
    axis=1
)


# ======================================================================================
# 1. COUNTS
# ======================================================================================

counts = (
    primary
    .groupby(
        [
            "comparison",
            "metric",
            "bootstrap_direction",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)


for column in [
    "improved",
    "worsened",
    "uncertain",
]:

    if column not in counts.columns:
        counts[column] = 0


print("=" * 100)
print("PER-CLASS BOOTSTRAP DIRECTION COUNTS")
print("=" * 100)

print(
    counts[
        [
            "comparison",
            "metric",
            "improved",
            "worsened",
            "uncertain",
        ]
    ].to_string(
        index=False
    )
)


# ======================================================================================
# 2. FULL CLASS TABLE
# ======================================================================================

display_cols = [
    "comparison",
    "class",
    "metric",
    "positive_n",
    "point_delta",
    "CI_low",
    "CI_high",
    "valid_bootstraps",
    "bootstrap_direction",
]


print("\n" + "=" * 100)
print("PER-CLASS BOOTSTRAP RESULTS")
print("=" * 100)

print(
    primary[
        display_cols
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ======================================================================================
# 3. ROBUST IMPROVEMENTS ONLY
# ======================================================================================

improved = primary[
    primary[
        "bootstrap_direction"
    ]
    ==
    "improved"
].copy()


print("\n" + "=" * 100)
print("CLASSES WITH CI ENTIRELY ABOVE ZERO")
print("=" * 100)

if len(improved) == 0:

    print("NONE")

else:

    print(
        improved[
            display_cols
        ]
        .sort_values(
            [
                "comparison",
                "metric",
                "point_delta"
            ],
            ascending=[
                True,
                True,
                False
            ]
        )
        .round(6)
        .to_string(
            index=False
        )
    )


# ======================================================================================
# 4. ROBUST DETERIORATIONS ONLY
# ======================================================================================

worsened = primary[
    primary[
        "bootstrap_direction"
    ]
    ==
    "worsened"
].copy()


print("\n" + "=" * 100)
print("CLASSES WITH CI ENTIRELY BELOW ZERO")
print("=" * 100)

print(
    worsened[
        display_cols
    ]
    .sort_values(
        [
            "comparison",
            "metric",
            "point_delta"
        ]
    )
    .round(6)
    .to_string(
        index=False
    )
)


# ======================================================================================
# 5. FLAG VERY SMALL POSITIVE COUNTS
#
# This does NOT exclude them from any primary analysis.
# It is only an interpretation flag.
# ======================================================================================

primary[
    "rare_class_flag"
] = np.where(
    primary[
        "positive_n"
    ]
    <
    20,
    "YES (<20 positives)",
    "NO"
)


primary.to_csv(
    "definitive_clean_gaze/ptbxl_evaluation/"
    "ptbxl_per_class_bootstrap_interpreted.csv",
    index=False
)


print(
    "\n✓ Interpreted per-class bootstrap table saved."
)
