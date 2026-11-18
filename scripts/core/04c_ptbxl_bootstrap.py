# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 7
# Stage: patient-cluster bootstrap for PTB-XL
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 4C — PATIENT-CLUSTER PAIRED BOOTSTRAP
#
# NO TRAINING.
# NO MODEL INFERENCE.
#
# Uses only:
#   definitive_clean_gaze/ptbxl_evaluation/ptbxl_fold10_predictions.csv
#
# Resampling unit:
#   PATIENT
#
# Comparisons:
#   Consultant - Base
#   All clinicians - Base
#   All clinicians - Consultant
#
# Metrics:
#   macro AUROC
#   macro AP
#   micro AUROC
#   micro AP
#
# Also produces paired per-class AUROC/AP confidence intervals.
# ======================================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ======================================================================================
# 1. SETTINGS
# ======================================================================================

SEED = 2026

N_BOOT = 3000

CI_LOW = 2.5
CI_HIGH = 97.5


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


MODELS = [
    "base",
    "consultant",
    "all_clinicians",
]


COMPARISONS = [
    (
        "consultant_vs_base",
        "consultant",
        "base",
    ),
    (
        "all_clinicians_vs_base",
        "all_clinicians",
        "base",
    ),
    (
        "all_clinicians_vs_consultant",
        "all_clinicians",
        "consultant",
    ),
]


# ======================================================================================
# 2. PATHS
# ======================================================================================

OUT_DIR = Path(
    "definitive_clean_gaze/ptbxl_evaluation"
)


PRED_PATH = (
    OUT_DIR
    /
    "ptbxl_fold10_predictions.csv"
)


if not PRED_PATH.exists():
    raise FileNotFoundError(
        PRED_PATH
    )


# ======================================================================================
# 3. LOAD SAVED PREDICTIONS
# ======================================================================================

df = pd.read_csv(
    PRED_PATH
)


if len(df) != 2198:
    raise RuntimeError(
        f"Expected 2198 ECGs, found {len(df)}."
    )


if df["patient_id"].nunique() != 1904:
    raise RuntimeError(
        "Expected 1904 unique patients."
    )


# True-label matrix
Y = np.column_stack(
    [
        df[
            f"true__{class_name}"
        ].to_numpy(
            dtype=np.int8
        )
        for class_name in MODEL_CLASSES
    ]
)


# Prediction matrices
P = {}


for model in MODELS:

    P[model] = np.column_stack(
        [
            df[
                f"{model}__{class_name}"
            ].to_numpy(
                dtype=np.float64
            )
            for class_name in MODEL_CLASSES
        ]
    )


if Y.shape != (
    2198,
    23
):
    raise RuntimeError(
        f"Unexpected Y shape: {Y.shape}"
    )


for model in MODELS:

    if P[model].shape != (
        2198,
        23
    ):
        raise RuntimeError(
            f"Unexpected prediction shape for {model}."
        )


print("=" * 100)
print("SAVED PREDICTION AUDIT")
print("=" * 100)

print(
    "ECGs:",
    len(df)
)

print(
    "Patients:",
    df["patient_id"].nunique()
)

print(
    "Classes:",
    Y.shape[1]
)

print(
    "All-zero target ECGs:",
    int(
        (
            Y.sum(
                axis=1
            )
            ==
            0
        ).sum()
    )
)


# ======================================================================================
# 4. PATIENT CLUSTERS
# ======================================================================================

patient_groups = (
    df
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


N_PATIENTS = len(
    patient_ids
)


if N_PATIENTS != 1904:
    raise RuntimeError(
        f"Expected 1904 patient clusters, found {N_PATIENTS}."
    )


# ======================================================================================
# 5. METRIC FUNCTION
# ======================================================================================

def calculate_metrics(
    y_true,
    y_score
):

    n_classes = y_true.shape[1]


    per_auc = np.full(
        n_classes,
        np.nan,
        dtype=np.float64
    )


    per_ap = np.full(
        n_classes,
        np.nan,
        dtype=np.float64
    )


    for j in range(
        n_classes
    ):

        yj = y_true[
            :,
            j
        ]


        # AUROC/AP only considered valid for the paired
        # class analysis when both classes occur.
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
            continue


        per_auc[j] = roc_auc_score(
            yj,
            y_score[
                :,
                j
            ]
        )


        per_ap[j] = average_precision_score(
            yj,
            y_score[
                :,
                j
            ]
        )


    # Macro result is only retained when every one of the
    # 23 classes is represented in that bootstrap sample.
    if np.all(
        np.isfinite(
            per_auc
        )
    ):

        macro_auc = float(
            np.mean(
                per_auc
            )
        )

        macro_ap = float(
            np.mean(
                per_ap
            )
        )

    else:

        macro_auc = np.nan
        macro_ap = np.nan


    micro_auc = roc_auc_score(
        y_true.ravel(),
        y_score.ravel()
    )


    micro_ap = average_precision_score(
        y_true.ravel(),
        y_score.ravel()
    )


    return {
        "macro_AUROC":
            macro_auc,

        "macro_AP":
            macro_ap,

        "micro_AUROC":
            float(
                micro_auc
            ),

        "micro_AP":
            float(
                micro_ap
            ),

        "per_class_AUROC":
            per_auc,

        "per_class_AP":
            per_ap,
    }


# ======================================================================================
# 6. POINT ESTIMATES
# ======================================================================================

point = {}


for model in MODELS:

    point[model] = calculate_metrics(
        Y,
        P[model]
    )


print("\n" + "=" * 100)
print("POINT ESTIMATES")
print("=" * 100)


point_rows = []


for model in MODELS:

    point_rows.append(
        {
            "model":
                model,

            "macro_AUROC":
                point[
                    model
                ][
                    "macro_AUROC"
                ],

            "macro_AP":
                point[
                    model
                ][
                    "macro_AP"
                ],

            "micro_AUROC":
                point[
                    model
                ][
                    "micro_AUROC"
                ],

            "micro_AP":
                point[
                    model
                ][
                    "micro_AP"
                ],
        }
    )


point_df = pd.DataFrame(
    point_rows
)


print(
    point_df
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ======================================================================================
# 7. STORAGE
# ======================================================================================

OVERALL_METRICS = [
    "macro_AUROC",
    "macro_AP",
    "micro_AUROC",
    "micro_AP",
]


overall_delta = {}


for comparison_name, _, _ in COMPARISONS:

    overall_delta[
        comparison_name
    ] = {
        metric: np.full(
            N_BOOT,
            np.nan,
            dtype=np.float64
        )
        for metric in OVERALL_METRICS
    }


per_class_auc_delta = {
    comparison_name:
        np.full(
            (
                N_BOOT,
                len(
                    MODEL_CLASSES
                )
            ),
            np.nan,
            dtype=np.float64
        )
    for comparison_name, _, _ in COMPARISONS
}


per_class_ap_delta = {
    comparison_name:
        np.full(
            (
                N_BOOT,
                len(
                    MODEL_CLASSES
                )
            ),
            np.nan,
            dtype=np.float64
        )
    for comparison_name, _, _ in COMPARISONS
}


# ======================================================================================
# 8. PATIENT-LEVEL PAIRED BOOTSTRAP
# ======================================================================================

rng = np.random.default_rng(
    SEED
)


print("\n" + "=" * 100)
print("PATIENT-CLUSTER BOOTSTRAP")
print("=" * 100)


for b in range(
    N_BOOT
):

    sampled_positions = rng.integers(
        low=0,
        high=N_PATIENTS,
        size=N_PATIENTS
    )


    sampled_patients = patient_ids[
        sampled_positions
    ]


    # If a patient is sampled multiple times, all ECGs belonging
    # to that patient are replicated correspondingly.
    bootstrap_indices = np.concatenate(
        [
            np.asarray(
                patient_groups[
                    patient_id
                ],
                dtype=int
            )
            for patient_id in sampled_patients
        ]
    )


    Y_boot = Y[
        bootstrap_indices
    ]


    metrics_boot = {}


    for model in MODELS:

        metrics_boot[
            model
        ] = calculate_metrics(
            Y_boot,
            P[
                model
            ][
                bootstrap_indices
            ]
        )


    # Same bootstrap sample used for every model -> paired comparison.
    for comparison_name, model_a, model_b in COMPARISONS:

        for metric in OVERALL_METRICS:

            a = metrics_boot[
                model_a
            ][
                metric
            ]

            base = metrics_boot[
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
                    base
                )
            ):

                overall_delta[
                    comparison_name
                ][
                    metric
                ][b] = (
                    a
                    -
                    base
                )


        per_class_auc_delta[
            comparison_name
        ][b, :] = (
            metrics_boot[
                model_a
            ][
                "per_class_AUROC"
            ]
            -
            metrics_boot[
                model_b
            ][
                "per_class_AUROC"
            ]
        )


        per_class_ap_delta[
            comparison_name
        ][b, :] = (
            metrics_boot[
                model_a
            ][
                "per_class_AP"
            ]
            -
            metrics_boot[
                model_b
            ][
                "per_class_AP"
            ]
        )


    if (
        (b + 1) % 250 == 0
        or
        (b + 1) == N_BOOT
    ):

        print(
            f"Completed {b + 1}/{N_BOOT}"
        )


# ======================================================================================
# 9. CI HELPER
# ======================================================================================

def summarize_bootstrap(
    values,
    point_difference
):

    values = np.asarray(
        values,
        dtype=np.float64
    )


    valid = values[
        np.isfinite(
            values
        )
    ]


    if len(valid) == 0:

        return {
            "point_delta":
                point_difference,

            "bootstrap_mean_delta":
                np.nan,

            "CI_low":
                np.nan,

            "CI_high":
                np.nan,

            "valid_bootstraps":
                0,

            "fraction_delta_below_zero":
                np.nan,

            "fraction_delta_above_zero":
                np.nan,
        }


    ci_low, ci_high = np.percentile(
        valid,
        [
            CI_LOW,
            CI_HIGH
        ]
    )


    return {
        "point_delta":
            float(
                point_difference
            ),

        "bootstrap_mean_delta":
            float(
                np.mean(
                    valid
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
                    valid
                )
            ),

        "fraction_delta_below_zero":
            float(
                np.mean(
                    valid < 0
                )
            ),

        "fraction_delta_above_zero":
            float(
                np.mean(
                    valid > 0
                )
            ),
    }


# ======================================================================================
# 10. OVERALL BOOTSTRAP SUMMARY
# ======================================================================================

overall_rows = []


for comparison_name, model_a, model_b in COMPARISONS:

    for metric in OVERALL_METRICS:

        point_difference = (
            point[
                model_a
            ][
                metric
            ]
            -
            point[
                model_b
            ][
                metric
            ]
        )


        result = summarize_bootstrap(
            overall_delta[
                comparison_name
            ][
                metric
            ],
            point_difference
        )


        overall_rows.append(
            {
                "comparison":
                    comparison_name,

                "metric":
                    metric,

                **result,
            }
        )


overall_summary = pd.DataFrame(
    overall_rows
)


print("\n\n" + "=" * 100)
print("PATIENT-CLUSTER BOOTSTRAP — OVERALL PERFORMANCE DIFFERENCE")
print("=" * 100)


print(
    overall_summary
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ======================================================================================
# 11. PER-CLASS BOOTSTRAP SUMMARY
# ======================================================================================

per_class_rows = []


for comparison_name, model_a, model_b in COMPARISONS:

    for class_idx, class_name in enumerate(
        MODEL_CLASSES
    ):

        # --------------------------------------------------------------
        # AUROC
        # --------------------------------------------------------------

        point_delta_auc = (
            point[
                model_a
            ][
                "per_class_AUROC"
            ][
                class_idx
            ]
            -
            point[
                model_b
            ][
                "per_class_AUROC"
            ][
                class_idx
            ]
        )


        auc_result = summarize_bootstrap(
            per_class_auc_delta[
                comparison_name
            ][
                :,
                class_idx
            ],
            point_delta_auc
        )


        per_class_rows.append(
            {
                "comparison":
                    comparison_name,

                "class":
                    class_name,

                "metric":
                    "AUROC",

                "positive_n":
                    int(
                        Y[
                            :,
                            class_idx
                        ].sum()
                    ),

                **auc_result,
            }
        )


        # --------------------------------------------------------------
        # AP
        # --------------------------------------------------------------

        point_delta_ap = (
            point[
                model_a
            ][
                "per_class_AP"
            ][
                class_idx
            ]
            -
            point[
                model_b
            ][
                "per_class_AP"
            ][
                class_idx
            ]
        )


        ap_result = summarize_bootstrap(
            per_class_ap_delta[
                comparison_name
            ][
                :,
                class_idx
            ],
            point_delta_ap
        )


        per_class_rows.append(
            {
                "comparison":
                    comparison_name,

                "class":
                    class_name,

                "metric":
                    "AP",

                "positive_n":
                    int(
                        Y[
                            :,
                            class_idx
                        ].sum()
                    ),

                **ap_result,
            }
        )


per_class_summary = pd.DataFrame(
    per_class_rows
)


# ======================================================================================
# 12. SAVE
# ======================================================================================

overall_summary.to_csv(
    OUT_DIR
    /
    "ptbxl_patient_cluster_bootstrap_overall.csv",
    index=False
)


per_class_summary.to_csv(
    OUT_DIR
    /
    "ptbxl_patient_cluster_bootstrap_per_class.csv",
    index=False
)


point_df.to_csv(
    OUT_DIR
    /
    "ptbxl_bootstrap_point_estimates.csv",
    index=False
)


np.savez_compressed(
    OUT_DIR
    /
    "ptbxl_patient_cluster_bootstrap_replicates.npz",

    consultant_vs_base_macro_AUROC=
        overall_delta[
            "consultant_vs_base"
        ][
            "macro_AUROC"
        ],

    consultant_vs_base_macro_AP=
        overall_delta[
            "consultant_vs_base"
        ][
            "macro_AP"
        ],

    all_vs_base_macro_AUROC=
        overall_delta[
            "all_clinicians_vs_base"
        ][
            "macro_AUROC"
        ],

    all_vs_base_macro_AP=
        overall_delta[
            "all_clinicians_vs_base"
        ][
            "macro_AP"
        ],

    all_vs_consultant_macro_AUROC=
        overall_delta[
            "all_clinicians_vs_consultant"
        ][
            "macro_AUROC"
        ],

    all_vs_consultant_macro_AP=
        overall_delta[
            "all_clinicians_vs_consultant"
        ][
            "macro_AP"
        ],
)


# ======================================================================================
# 13. MANIFEST
# ======================================================================================

manifest = {

    "stage":
        "Step 4C patient-cluster paired bootstrap",

    "source":
        str(
            PRED_PATH
        ),

    "bootstrap_seed":
        SEED,

    "bootstrap_replicates":
        N_BOOT,

    "resampling_unit":
        "patient",

    "unique_patients":
        N_PATIENTS,

    "ECG_records":
        len(
            df
        ),

    "confidence_interval":
        "2.5th to 97.5th percentile",

    "comparisons": [
        comparison_name
        for comparison_name, _, _ in COMPARISONS
    ],

    "model_inference_performed":
        False,

    "training_performed":
        False,
}


with open(
    OUT_DIR
    /
    "ptbxl_patient_cluster_bootstrap_manifest.json",
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


print("\n\n" + "=" * 100)
print("STEP 4C COMPLETE — NO MODEL INFERENCE OR TRAINING")
print("=" * 100)


print(
    """
Please paste back ONLY:

PATIENT-CLUSTER BOOTSTRAP — OVERALL PERFORMANCE DIFFERENCE

The per-class bootstrap file is already saved and we can inspect it
after establishing the overall result.
"""
)
