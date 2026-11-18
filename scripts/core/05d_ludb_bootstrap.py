# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 13
# Stage: LUDB paired bootstrap
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 5D — LUDB PAIRED BOOTSTRAP OF SAVED OOF PREDICTIONS
#
# NO MODEL INFERENCE.
# NO MODEL TRAINING.
# NO LINEAR-PROBE TRAINING.
#
# Uses ONLY:
#   definitive_clean_gaze/ludb_evaluation/ludb_step5c_oof_predictions.csv
#
# Purpose:
#   Quantify sampling uncertainty around the paired OOF performance
#   differences observed in Step 5C.
#
# Primary metrics:
#   AUROC
#   Average Precision (AP)
#
# Secondary:
#   balanced accuracy
#
# Comparisons:
#   CONSULTANT vs BASE
#   ALL_CLINICIANS vs BASE
#   ALL_CLINICIANS vs CONSULTANT
#
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)


# ======================================================================================
# 1. SETTINGS
# ======================================================================================

SEED = 2026
N_BOOT = 5000

CI_PERCENTILES = [
    2.5,
    97.5,
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


TASKS = (
    PRIMARY_TASKS
    +
    EXPLORATORY_TASKS
)


MODELS = [
    "BASE",
    "CONSULTANT",
    "ALL_CLINICIANS",
]


COMPARISONS = [

    (
        "CONSULTANT_vs_BASE",
        "CONSULTANT",
        "BASE",
    ),

    (
        "ALL_CLINICIANS_vs_BASE",
        "ALL_CLINICIANS",
        "BASE",
    ),

    (
        "ALL_CLINICIANS_vs_CONSULTANT",
        "ALL_CLINICIANS",
        "CONSULTANT",
    ),
]


# ======================================================================================
# 2. PATHS
# ======================================================================================

LUDB_DIR = Path(
    "definitive_clean_gaze/ludb_evaluation"
)


PRED_PATH = (
    LUDB_DIR
    /
    "ludb_step5c_oof_predictions.csv"
)


if not PRED_PATH.exists():

    raise FileNotFoundError(
        PRED_PATH
    )


# ======================================================================================
# 3. LOAD SAVED OOF PREDICTIONS
# ======================================================================================

pred = pd.read_csv(
    PRED_PATH
)


print("=" * 110)
print("SAVED LUDB OOF PREDICTION AUDIT")
print("=" * 110)


print(
    "Rows:",
    len(
        pred
    )
)


print(
    "Tasks:",
    sorted(
        pred[
            "task"
        ].unique()
    )
)


print(
    "Models:",
    sorted(
        pred[
            "model"
        ].unique()
    )
)


if set(
    pred[
        "task"
    ].unique()
) != set(
    TASKS
):

    raise RuntimeError(
        "Unexpected LUDB tasks in saved predictions."
    )


if set(
    pred[
        "model"
    ].unique()
) != set(
    MODELS
):

    raise RuntimeError(
        "Unexpected model names in saved predictions."
    )


# ======================================================================================
# 4. VERIFY EXACT PAIRED RECORD SETS
# ======================================================================================

for task in TASKS:

    sub = pred[
        pred[
            "task"
        ]
        ==
        task
    ]


    reference_ids = None


    for model in MODELS:

        m = (
            sub[
                sub[
                    "model"
                ]
                ==
                model
            ]
            .sort_values(
                "record_id"
            )
        )


        ids = m[
            "record_id"
        ].to_numpy()


        if reference_ids is None:

            reference_ids = ids

        else:

            if not np.array_equal(
                reference_ids,
                ids
            ):

                raise RuntimeError(
                    f"{task}: record sets differ between models."
                )


    print(
        f"{task:20s} paired records = {len(reference_ids)}"
    )


print(
    "\n✓ Every model was evaluated on identical records within each task."
)


# ======================================================================================
# 5. METRIC FUNCTION
# ======================================================================================

def calculate_metrics(
    y,
    prob
):

    y = np.asarray(
        y,
        dtype=int
    )


    prob = np.asarray(
        prob,
        dtype=float
    )


    if len(
        np.unique(
            y
        )
    ) < 2:

        return {
            "AUROC":
                np.nan,

            "AP":
                np.nan,

            "balanced_accuracy":
                np.nan,
        }


    predicted = (
        prob
        >=
        0.5
    ).astype(
        int
    )


    return {

        "AUROC":
            float(
                roc_auc_score(
                    y,
                    prob
                )
            ),

        "AP":
            float(
                average_precision_score(
                    y,
                    prob
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    predicted
                )
            ),
    }


# ======================================================================================
# 6. CONSTRUCT ONE PAIRED TABLE PER TASK
# ======================================================================================

task_data = {}


for task in TASKS:

    sub = pred[
        pred[
            "task"
        ]
        ==
        task
    ]


    model_frames = {}


    for model in MODELS:

        model_frames[
            model
        ] = (
            sub[
                sub[
                    "model"
                ]
                ==
                model
            ]
            [
                [
                    "record_id",
                    "true_label",
                    "probability",
                ]
            ]
            .sort_values(
                "record_id"
            )
            .reset_index(
                drop=True
            )
        )


    reference = model_frames[
        "BASE"
    ]


    y = reference[
        "true_label"
    ].to_numpy(
        dtype=int
    )


    record_ids = reference[
        "record_id"
    ].to_numpy()


    probabilities = {}


    for model in MODELS:

        frame = model_frames[
            model
        ]


        if not np.array_equal(
            frame[
                "record_id"
            ].to_numpy(),
            record_ids
        ):

            raise RuntimeError(
                f"{task}/{model}: ID mismatch."
            )


        if not np.array_equal(
            frame[
                "true_label"
            ].to_numpy(
                dtype=int
            ),
            y
        ):

            raise RuntimeError(
                f"{task}/{model}: target mismatch."
            )


        probabilities[
            model
        ] = frame[
            "probability"
        ].to_numpy(
            dtype=float
        )


    task_data[
        task
    ] = {

        "record_ids":
            record_ids,

        "y":
            y,

        "probabilities":
            probabilities,
    }


# ======================================================================================
# 7. POINT ESTIMATES FROM SAVED OOF PREDICTIONS
# ======================================================================================

point_rows = []


for task in TASKS:

    data = task_data[
        task
    ]


    y = data[
        "y"
    ]


    role = (
        "primary"
        if task in PRIMARY_TASKS
        else "exploratory"
    )


    for model in MODELS:

        m = calculate_metrics(
            y,
            data[
                "probabilities"
            ][
                model
            ]
        )


        point_rows.append(
            {
                "task":
                    task,

                "role":
                    role,

                "model":
                    model,

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

                **m,
            }
        )


point_df = pd.DataFrame(
    point_rows
)


print(
    "\n"
    +
    "=" * 110
)

print(
    "POINT ESTIMATES RECONSTRUCTED FROM SAVED OOF PREDICTIONS"
)

print(
    "=" * 110
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
# 8. BOOTSTRAP STORAGE
# ======================================================================================

METRICS = [
    "AUROC",
    "AP",
    "balanced_accuracy",
]


bootstrap_store = {}


for task in TASKS:

    bootstrap_store[
        task
    ] = {}


    for comparison_name, _, _ in COMPARISONS:

        bootstrap_store[
            task
        ][
            comparison_name
        ] = {

            metric:
                np.full(
                    N_BOOT,
                    np.nan,
                    dtype=np.float64
                )

            for metric in METRICS
        }


# ======================================================================================
# 9. PAIRED RECORD-LEVEL BOOTSTRAP
# ======================================================================================

rng = np.random.default_rng(
    SEED
)


print(
    "\n"
    +
    "=" * 110
)

print(
    "LUDB PAIRED BOOTSTRAP"
)

print(
    "=" * 110
)


for task in TASKS:

    data = task_data[
        task
    ]


    y = data[
        "y"
    ]


    probabilities = data[
        "probabilities"
    ]


    n = len(
        y
    )


    print(
        f"\nTask: {task} | N={n} | positive={int(y.sum())}"
    )


    for b in range(
        N_BOOT
    ):

        # Same sampled records for all three models:
        # paired bootstrap.
        idx = rng.integers(
            low=0,
            high=n,
            size=n
        )


        yb = y[
            idx
        ]


        # Rare tasks can occasionally yield a one-class bootstrap.
        if len(
            np.unique(
                yb
            )
        ) < 2:

            continue


        boot_metrics = {}


        for model in MODELS:

            boot_metrics[
                model
            ] = calculate_metrics(
                yb,
                probabilities[
                    model
                ][
                    idx
                ]
            )


        for (
            comparison_name,
            model_a,
            model_b
        ) in COMPARISONS:

            for metric in METRICS:

                a = boot_metrics[
                    model_a
                ][
                    metric
                ]


                bb = boot_metrics[
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
                        bb
                    )
                ):

                    bootstrap_store[
                        task
                    ][
                        comparison_name
                    ][
                        metric
                    ][
                        b
                    ] = (
                        a
                        -
                        bb
                    )


        if (
            (b + 1) % 1000 == 0
            or
            (b + 1) == N_BOOT
        ):

            print(
                f"  {b + 1}/{N_BOOT}"
            )


# ======================================================================================
# 10. SUMMARY HELPER
# ======================================================================================

def bootstrap_summary(
    values,
    point_delta
):

    values = np.asarray(
        values,
        dtype=float
    )


    valid = values[
        np.isfinite(
            values
        )
    ]


    if len(
        valid
    ) == 0:

        return {

            "point_delta":
                float(
                    point_delta
                ),

            "bootstrap_mean_delta":
                np.nan,

            "CI_low":
                np.nan,

            "CI_high":
                np.nan,

            "valid_bootstraps":
                0,

            "fraction_below_zero":
                np.nan,

            "fraction_above_zero":
                np.nan,

            "direction":
                "undefined",
        }


    ci_low, ci_high = np.percentile(
        valid,
        CI_PERCENTILES
    )


    if ci_low > 0:

        direction = "improved"

    elif ci_high < 0:

        direction = "worsened"

    else:

        direction = "uncertain"


    return {

        "point_delta":
            float(
                point_delta
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

        "fraction_below_zero":
            float(
                np.mean(
                    valid < 0
                )
            ),

        "fraction_above_zero":
            float(
                np.mean(
                    valid > 0
                )
            ),

        "direction":
            direction,
    }


# ======================================================================================
# 11. BUILD BOOTSTRAP SUMMARY
# ======================================================================================

summary_rows = []


for task in TASKS:

    role = (
        "primary"
        if task in PRIMARY_TASKS
        else "exploratory"
    )


    data = task_data[
        task
    ]


    y = data[
        "y"
    ]


    point_metrics = {}


    for model in MODELS:

        point_metrics[
            model
        ] = calculate_metrics(
            y,
            data[
                "probabilities"
            ][
                model
            ]
        )


    for (
        comparison_name,
        model_a,
        model_b
    ) in COMPARISONS:

        for metric in METRICS:

            point_delta = (
                point_metrics[
                    model_a
                ][
                    metric
                ]
                -
                point_metrics[
                    model_b
                ][
                    metric
                ]
            )


            result = bootstrap_summary(
                bootstrap_store[
                    task
                ][
                    comparison_name
                ][
                    metric
                ],
                point_delta
            )


            summary_rows.append(
                {
                    "task":
                        task,

                    "role":
                        role,

                    "comparison":
                        comparison_name,

                    "metric":
                        metric,

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

                    **result,
                }
            )


bootstrap_summary_df = pd.DataFrame(
    summary_rows
)


# ======================================================================================
# 12. SAVE BOOTSTRAP REPLICATES
# ======================================================================================

npz_payload = {}


for task in TASKS:

    for comparison_name, _, _ in COMPARISONS:

        for metric in METRICS:

            key = (
                f"{task}"
                f"__{comparison_name}"
                f"__{metric}"
            )


            npz_payload[
                key
            ] = (
                bootstrap_store[
                    task
                ][
                    comparison_name
                ][
                    metric
                ]
            )


np.savez_compressed(
    LUDB_DIR
    /
    "ludb_step5d_bootstrap_replicates.npz",

    **npz_payload
)


# ======================================================================================
# 13. SAVE TABLES
# ======================================================================================

bootstrap_summary_df.to_csv(
    LUDB_DIR
    /
    "ludb_step5d_paired_bootstrap_summary.csv",
    index=False
)


point_df.to_csv(
    LUDB_DIR
    /
    "ludb_step5d_point_estimates.csv",
    index=False
)


# ======================================================================================
# 14. PRINT PRIMARY AUROC/AP RESULTS
# ======================================================================================

primary_main = bootstrap_summary_df[
    (
        bootstrap_summary_df[
            "role"
        ]
        ==
        "primary"
    )
    &
    (
        bootstrap_summary_df[
            "metric"
        ].isin(
            [
                "AUROC",
                "AP",
            ]
        )
    )
]


print(
    "\n\n"
    +
    "=" * 120
)

print(
    "LUDB PRIMARY TASKS — PAIRED BOOTSTRAP DIFFERENCES"
)

print(
    "=" * 120
)


print(
    primary_main[
        [
            "task",
            "comparison",
            "metric",
            "N",
            "positive",
            "point_delta",
            "CI_low",
            "CI_high",
            "valid_bootstraps",
            "direction",
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
# 15. PRINT EXPLORATORY AUROC/AP RESULTS
# ======================================================================================

exploratory_main = bootstrap_summary_df[
    (
        bootstrap_summary_df[
            "role"
        ]
        ==
        "exploratory"
    )
    &
    (
        bootstrap_summary_df[
            "metric"
        ].isin(
            [
                "AUROC",
                "AP",
            ]
        )
    )
]


print(
    "\n\n"
    +
    "=" * 120
)

print(
    "LUDB EXPLORATORY TASKS — PAIRED BOOTSTRAP DIFFERENCES"
)

print(
    "=" * 120
)


print(
    exploratory_main[
        [
            "task",
            "comparison",
            "metric",
            "N",
            "positive",
            "point_delta",
            "CI_low",
            "CI_high",
            "valid_bootstraps",
            "direction",
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
# 16. PRIMARY SUMMARY: ONLY GAZE VS BASE
# ======================================================================================

primary_vs_base = primary_main[
    primary_main[
        "comparison"
    ].isin(
        [
            "CONSULTANT_vs_BASE",
            "ALL_CLINICIANS_vs_BASE",
        ]
    )
]


print(
    "\n\n"
    +
    "=" * 120
)

print(
    "PRIMARY GAZE-vs-BASE SUMMARY"
)

print(
    "=" * 120
)


print(
    primary_vs_base[
        [
            "task",
            "comparison",
            "metric",
            "point_delta",
            "CI_low",
            "CI_high",
            "direction",
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
# 17. MANIFEST
# ======================================================================================

manifest = {

    "stage":
        "Step 5D LUDB paired bootstrap from frozen OOF predictions",

    "source_predictions":
        str(
            PRED_PATH
        ),

    "bootstrap_repetitions":
        N_BOOT,

    "bootstrap_seed":
        SEED,

    "resampling_unit":
        "LUDB record (one record per unique LUDB ID)",

    "paired":
        True,

    "same_bootstrap_sample_for_all_models":
        True,

    "confidence_interval":
        "percentile 95% CI (2.5th, 97.5th percentiles)",

    "primary_tasks":
        PRIMARY_TASKS,

    "exploratory_tasks":
        EXPLORATORY_TASKS,

    "primary_metrics":
        [
            "AUROC",
            "Average Precision",
        ],

    "secondary_metric":
        "balanced accuracy",

    "comparisons":
        [
            comparison_name
            for comparison_name, _, _
            in COMPARISONS
        ],

    "model_inference_performed":
        False,

    "linear_probe_training_performed":
        False,

    "gaze_model_training_performed":
        False,

    "endpoint_changes_performed":
        False,
}


with open(
    LUDB_DIR
    /
    "ludb_step5d_manifest.json",
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ======================================================================================
# 18. FINISH
# ======================================================================================

print(
    "\n\n"
    +
    "=" * 110
)

print(
    "STEP 5D COMPLETE — SAVED OOF PREDICTIONS ONLY"
)

print(
    "=" * 110
)


print(
    """
Please paste back:

1. LUDB PRIMARY TASKS — PAIRED BOOTSTRAP DIFFERENCES

2. LUDB EXPLORATORY TASKS — PAIRED BOOTSTRAP DIFFERENCES

3. PRIMARY GAZE-vs-BASE SUMMARY

Do not rerun or modify any models, probes, endpoints, folds, or preprocessing.
"""
)
