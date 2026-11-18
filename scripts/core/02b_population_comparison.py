# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 2
# Stage: compare consultant and all-clinician populations
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 2B — CONSULTANT vs ALL-CLINICIAN GAZE SUPERVISION
#
# NO TRAINING.
#
# Question:
# Does the CHANGE in held-out gaze alignment differ between the
# consultant-only and all-clinician targets?
#
# Statistical unit = ECG, n = 10 paired observations.
# ======================================================================================

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


RESULT_PATH = Path(
    "definitive_clean_gaze/step2_cv/heldout_metrics.csv"
)

if not RESULT_PATH.exists():
    raise FileNotFoundError(RESULT_PATH)


df = pd.read_csv(
    RESULT_PATH
)


# ======================================================================================
# 1. PIVOT SO EACH ECG HAS BOTH POPULATIONS
# ======================================================================================

consultant = (
    df[
        df["population"]
        ==
        "consultant"
    ]
    .set_index("ECG")
    .sort_index()
)


all_clinicians = (
    df[
        df["population"]
        ==
        "all_clinicians"
    ]
    .set_index("ECG")
    .sort_index()
)


if list(consultant.index) != list(all_clinicians.index):
    raise RuntimeError(
        "ECGs do not align between populations."
    )


if len(consultant) != 10:
    raise RuntimeError(
        f"Expected 10 paired ECGs, found {len(consultant)}."
    )


# ======================================================================================
# 2. EXPRESS CHANGE SO POSITIVE ALWAYS MEANS IMPROVEMENT
#
# cosine:
#     trained - baseline
#
# MSE/JSD:
#     baseline - trained
#     because LOWER is better
# ======================================================================================

comparison = pd.DataFrame(
    index=consultant.index
)


comparison["consultant_cosine_improvement"] = (
    consultant["trained_cosine"]
    -
    consultant["baseline_cosine"]
)


comparison["all_clinicians_cosine_improvement"] = (
    all_clinicians["trained_cosine"]
    -
    all_clinicians["baseline_cosine"]
)


comparison["consultant_MSE_improvement"] = (
    consultant["baseline_MSE"]
    -
    consultant["trained_MSE"]
)


comparison["all_clinicians_MSE_improvement"] = (
    all_clinicians["baseline_MSE"]
    -
    all_clinicians["trained_MSE"]
)


comparison["consultant_JSD_improvement"] = (
    consultant["baseline_JSD"]
    -
    consultant["trained_JSD"]
)


comparison["all_clinicians_JSD_improvement"] = (
    all_clinicians["baseline_JSD"]
    -
    all_clinicians["trained_JSD"]
)


# Positive advantage means all-clinician supervision improved more.
comparison["all_minus_consultant_cosine"] = (
    comparison["all_clinicians_cosine_improvement"]
    -
    comparison["consultant_cosine_improvement"]
)


comparison["all_minus_consultant_MSE"] = (
    comparison["all_clinicians_MSE_improvement"]
    -
    comparison["consultant_MSE_improvement"]
)


comparison["all_minus_consultant_JSD"] = (
    comparison["all_clinicians_JSD_improvement"]
    -
    comparison["consultant_JSD_improvement"]
)


comparison = comparison.reset_index()


print("=" * 100)
print("PER-ECG POPULATION COMPARISON")
print("=" * 100)

print(
    comparison[
        [
            "ECG",
            "all_minus_consultant_cosine",
            "all_minus_consultant_MSE",
            "all_minus_consultant_JSD",
        ]
    ]
    .round(7)
    .to_string(index=False)
)


# ======================================================================================
# 3. PAIRED WILCOXON TESTS
# ======================================================================================

summary_rows = []


metric_definitions = [

    (
        "cosine",
        "consultant_cosine_improvement",
        "all_clinicians_cosine_improvement",
    ),

    (
        "MSE",
        "consultant_MSE_improvement",
        "all_clinicians_MSE_improvement",
    ),

    (
        "JSD",
        "consultant_JSD_improvement",
        "all_clinicians_JSD_improvement",
    ),
]


for metric, consultant_col, all_col in metric_definitions:

    c = comparison[
        consultant_col
    ].to_numpy(
        dtype=float
    )


    a = comparison[
        all_col
    ].to_numpy(
        dtype=float
    )


    result = wilcoxon(
        a,
        c,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )


    difference = (
        a - c
    )


    summary_rows.append(
        {
            "metric":
                metric,

            "consultant_mean_improvement":
                float(
                    np.mean(c)
                ),

            "all_clinician_mean_improvement":
                float(
                    np.mean(a)
                ),

            "all_minus_consultant_mean":
                float(
                    np.mean(
                        difference
                    )
                ),

            "all_minus_consultant_median":
                float(
                    np.median(
                        difference
                    )
                ),

            "ECGs_favouring_all_clinicians":
                int(
                    np.sum(
                        difference > 0
                    )
                ),

            "ECGs_favouring_consultants":
                int(
                    np.sum(
                        difference < 0
                    )
                ),

            "wilcoxon_statistic":
                float(
                    result.statistic
                ),

            "wilcoxon_p":
                float(
                    result.pvalue
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print(
    "\n" +
    "=" * 100
)

print(
    "CONSULTANT vs ALL-CLINICIAN SUPERVISION SUMMARY"
)

print(
    "=" * 100
)


print(
    summary
    .round(7)
    .to_string(index=False)
)


# ======================================================================================
# 4. SAVE
# ======================================================================================

comparison.to_csv(
    "definitive_clean_gaze/step2_cv/"
    "population_improvement_by_ECG.csv",
    index=False
)


summary.to_csv(
    "definitive_clean_gaze/step2_cv/"
    "population_improvement_comparison.csv",
    index=False
)


print(
    "\n✓ Population comparison saved."
)

print(
    "\nNo model weights were changed."
)
