# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 14
# Stage: model lead-profile visualisation
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 6A — VISUALISE CLINICIAN GAZE vs MODEL LEAD ATTENTION
#
# READ-ONLY MODEL INFERENCE.
# NO TRAINING.
# NO WEIGHT CHANGES.
#
# Produces:
#
#   Consultant supervision:
#       Clinician gaze
#       Base-model lead attention
#       Consultant-gaze model attention
#       Change = Consultant model - Base
#
#   All-clinician supervision:
#       Clinician gaze
#       Base-model lead attention
#       All-clinician model attention
#       Change = All-clinician model - Base
#
# Attention definition is EXACTLY the frozen primary definition:
#
#   add_3 : (batch, 12, 55, 128)
#   lead RMS = sqrt(mean(add_3^2 across time and channels))
#   normalise across 12 leads
#
# ======================================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt


# ======================================================================================
# 1. PATHS
# ======================================================================================

ROOT = Path("definitive_clean_gaze")

X_PATH = Path(
    "final_digitisation_v9/X_gaze_final.npy"
)

GAZE_TARGET_PATH = (
    ROOT
    /
    "clean_gaze_targets.npz"
)

BASE_MODEL_PATH = Path(
    "12 leads.h5"
)

CONSULTANT_WEIGHTS = (
    ROOT
    /
    "final_models"
    /
    "clean_consultant.weights.h5"
)

ALL_WEIGHTS = (
    ROOT
    /
    "final_models"
    /
    "clean_all_clinicians.weights.h5"
)

OUT_DIR = (
    ROOT
    /
    "visualisations"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for path in [
    X_PATH,
    GAZE_TARGET_PATH,
    BASE_MODEL_PATH,
    CONSULTANT_WEIGHTS,
    ALL_WEIGHTS,
]:
    if not path.exists():
        raise FileNotFoundError(path)


# ======================================================================================
# 2. FIXED ORDERS
# ======================================================================================

ECG_KEYS = [
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


STANDARD_LEADS = [
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


# Physical 3 × 4 ECG display layout
DISPLAY_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


LEAD_INDEX = {
    lead: i
    for i, lead in enumerate(STANDARD_LEADS)
}


# ======================================================================================
# 3. LOAD FROZEN ECGs
# ======================================================================================

X = np.load(
    X_PATH
).astype(
    np.float32
)


print("=" * 100)
print("FROZEN GAZE ECG DATA")
print("=" * 100)

print(
    "Shape:",
    X.shape
)


if X.shape != (
    10,
    12,
    1000,
    1
):
    raise RuntimeError(
        f"Unexpected X shape: {X.shape}"
    )


# ======================================================================================
# 4. LOAD CLEAN GAZE TARGETS
#
# Robustly identify the two 10 × 12 matrices.
# ======================================================================================

npz = np.load(
    GAZE_TARGET_PATH
)


print(
    "\nNPZ keys:",
    npz.files
)


candidate_arrays = {
    key: np.asarray(npz[key])
    for key in npz.files
    if np.asarray(npz[key]).shape == (10, 12)
}


if len(candidate_arrays) < 2:
    raise RuntimeError(
        "Could not find two (10,12) gaze-target arrays."
    )


consultant_target = None
all_target = None


for key, array in candidate_arrays.items():

    lower = key.lower()

    if (
        "consult" in lower
        and
        "all" not in lower
    ):
        consultant_target = array.astype(
            np.float64
        )

    if (
        "all" in lower
        or
        "clinician" in lower
    ):
        if "consult" not in lower or "all" in lower:
            all_target = array.astype(
                np.float64
            )


# Fallback if names are different but exactly two matrices exist
if (
    consultant_target is None
    or
    all_target is None
):

    print(
        "\nAutomatic name matching incomplete."
    )

    print(
        "Candidate arrays:"
    )

    for key in candidate_arrays:
        print(
            key
        )

    # Prefer arrays by mean difference from one another only if exactly two.
    if len(candidate_arrays) == 2:

        keys = list(
            candidate_arrays.keys()
        )

        consultant_target = (
            candidate_arrays[
                keys[0]
            ]
            .astype(
                np.float64
            )
        )

        all_target = (
            candidate_arrays[
                keys[1]
            ]
            .astype(
                np.float64
            )
        )

        print(
            "\nWARNING:"
            "\nUsing candidate arrays in stored order:"
            f"\n  Consultant assumed = {keys[0]}"
            f"\n  All clinicians assumed = {keys[1]}"
        )

    else:

        raise RuntimeError(
            "Please inspect the NPZ keys before proceeding."
        )


# Validate target distributions
for name, target in [
    (
        "Consultant",
        consultant_target
    ),
    (
        "All clinicians",
        all_target
    ),
]:

    if target.shape != (
        10,
        12
    ):
        raise RuntimeError(
            f"{name} target shape incorrect."
        )

    if np.any(
        target < 0
    ):
        raise RuntimeError(
            f"{name}: negative gaze weights."
        )

    sums = target.sum(
        axis=1
    )

    if not np.allclose(
        sums,
        1.0,
        atol=1e-5
    ):
        raise RuntimeError(
            f"{name}: target rows do not sum to 1."
        )


print(
    "\n✓ Clean gaze targets loaded."
)


# ======================================================================================
# 5. LOAD THREE FROZEN MODELS
# ======================================================================================

def load_variant(
    weights=None
):

    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )

    if weights is not None:

        model.load_weights(
            weights
        )

    return model


base_model = load_variant()

consultant_model = load_variant(
    CONSULTANT_WEIGHTS
)

all_model = load_variant(
    ALL_WEIGHTS
)


# ======================================================================================
# 6. FROZEN RMS LEAD-ATTENTION READOUT
# ======================================================================================

def rms_lead_attention(
    model,
    X
):

    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(
            "add_3"
        ).output
    )


    feature = feature_model.predict(
        X,
        batch_size=10,
        verbose=0
    )


    print(
        model.name,
        "add_3:",
        feature.shape
    )


    if feature.shape != (
        10,
        12,
        55,
        128
    ):
        raise RuntimeError(
            f"Unexpected add_3 shape: {feature.shape}"
        )


    # EXACT frozen attention definition
    lead_energy = np.sqrt(
        np.mean(
            np.square(
                feature.astype(
                    np.float64
                )
            ),
            axis=(
                2,
                3
            )
        )
    )


    lead_attention = (
        lead_energy
        /
        lead_energy.sum(
            axis=1,
            keepdims=True
        )
    )


    return lead_attention


print("\n" + "=" * 100)
print("MODEL ATTENTION EXTRACTION")
print("=" * 100)


base_attention = rms_lead_attention(
    base_model,
    X
)

consultant_attention = rms_lead_attention(
    consultant_model,
    X
)

all_attention = rms_lead_attention(
    all_model,
    X
)


for name, array in [
    (
        "BASE",
        base_attention
    ),
    (
        "CONSULTANT",
        consultant_attention
    ),
    (
        "ALL_CLINICIANS",
        all_attention
    ),
]:

    print(
        name,
        "shape:",
        array.shape,
        "| row sums:",
        np.round(
            array.sum(
                axis=1
            ),
            6
        )
    )


# ======================================================================================
# 7. CONVERT STANDARD LEAD ORDER -> PHYSICAL ECG GRID
# ======================================================================================

def vector_to_display_grid(
    vector
):

    grid = np.zeros(
        (
            3,
            4
        ),
        dtype=float
    )


    for row in range(
        3
    ):

        for col in range(
            4
        ):

            lead = DISPLAY_LAYOUT[
                row
            ][
                col
            ]

            grid[
                row,
                col
            ] = vector[
                LEAD_INDEX[
                    lead
                ]
            ]


    return grid


# ======================================================================================
# 8. COSINE SIMILARITY HELPER
# ======================================================================================

def cosine(
    a,
    b
):

    return float(
        np.dot(
            a,
            b
        )
        /
        (
            np.linalg.norm(
                a
            )
            *
            np.linalg.norm(
                b
            )
            +
            1e-12
        )
    )


# ======================================================================================
# 9. DRAW ONE 3×4 LEAD MAP
# ======================================================================================

def draw_lead_map(
    ax,
    vector,
    title,
    vmin,
    vmax,
    cmap,
    show_values=True,
):

    grid = vector_to_display_grid(
        vector
    )


    image = ax.imshow(
        grid,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal"
    )


    ax.set_xticks([])
    ax.set_yticks([])


    for row in range(
        3
    ):

        for col in range(
            4
        ):

            lead = DISPLAY_LAYOUT[
                row
            ][
                col
            ]

            value = grid[
                row,
                col
            ]


            if show_values:

                label = (
                    f"{lead}\n"
                    f"{100 * value:.1f}%"
                )

            else:

                label = lead


            # Select text contrast using local cell intensity.
            midpoint = (
                vmin
                +
                vmax
            ) / 2


            text_colour = (
                "white"
                if value > midpoint
                else "black"
            )


            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=text_colour,
                fontweight="bold",
            )


    ax.set_title(
        title,
        fontsize=11,
        pad=8
    )


    return image


# ======================================================================================
# 10. INDIVIDUAL ECG FIGURES
# ======================================================================================

def save_individual_figures(
    population_name,
    gaze_target,
    guided_attention,
    filename_prefix,
):

    absolute_max = max(
        float(
            gaze_target.max()
        ),
        float(
            base_attention.max()
        ),
        float(
            guided_attention.max()
        ),
    )


    delta = (
        guided_attention
        -
        base_attention
    )


    delta_abs_max = float(
        np.max(
            np.abs(
                delta
            )
        )
    )


    for i, ecg in enumerate(
        ECG_KEYS
    ):

        base_cos = cosine(
            base_attention[
                i
            ],
            gaze_target[
                i
            ]
        )


        guided_cos = cosine(
            guided_attention[
                i
            ],
            gaze_target[
                i
            ]
        )


        fig, axes = plt.subplots(
            1,
            4,
            figsize=(
                15,
                3.7
            )
        )


        im1 = draw_lead_map(
            axes[
                0
            ],
            gaze_target[
                i
            ],
            f"{population_name} gaze",
            vmin=0,
            vmax=absolute_max,
            cmap="viridis",
        )


        draw_lead_map(
            axes[
                1
            ],
            base_attention[
                i
            ],
            (
                "Base model\n"
                f"cosine={base_cos:.3f}"
            ),
            vmin=0,
            vmax=absolute_max,
            cmap="viridis",
        )


        draw_lead_map(
            axes[
                2
            ],
            guided_attention[
                i
            ],
            (
                "Gaze-guided model\n"
                f"cosine={guided_cos:.3f}"
            ),
            vmin=0,
            vmax=absolute_max,
            cmap="viridis",
        )


        im4 = draw_lead_map(
            axes[
                3
            ],
            delta[
                i
            ],
            (
                "Change in model attention\n"
                "guided − base"
            ),
            vmin=-delta_abs_max,
            vmax=delta_abs_max,
            cmap="RdBu_r",
        )


        fig.suptitle(
            (
                f"{ecg} — {population_name} supervision"
                f" | Δ cosine = {guided_cos - base_cos:+.3f}"
            ),
            fontsize=13,
            fontweight="bold",
        )


        # Shared attention colour bar
        cb1 = fig.colorbar(
            im1,
            ax=axes[
                :3
            ],
            fraction=0.025,
            pad=0.02,
        )

        cb1.set_label(
            "Normalised lead attention"
        )


        # Delta colour bar
        cb2 = fig.colorbar(
            im4,
            ax=axes[
                3
            ],
            fraction=0.08,
            pad=0.04,
        )

        cb2.set_label(
            "Change in attention"
        )


        fig.subplots_adjust(
            left=0.03,
            right=0.96,
            top=0.78,
            bottom=0.08,
            wspace=0.35,
        )


        png_path = (
            OUT_DIR
            /
            f"{filename_prefix}_{ecg}.png"
        )


        pdf_path = (
            OUT_DIR
            /
            f"{filename_prefix}_{ecg}.pdf"
        )


        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight"
        )


        fig.savefig(
            pdf_path,
            bbox_inches="tight"
        )


        plt.close(
            fig
        )


# ======================================================================================
# 11. MAKE INDIVIDUAL FIGURES
# ======================================================================================

save_individual_figures(
    population_name="Consultant",
    gaze_target=consultant_target,
    guided_attention=consultant_attention,
    filename_prefix="consultant_attention",
)


save_individual_figures(
    population_name="All-clinician",
    gaze_target=all_target,
    guided_attention=all_attention,
    filename_prefix="all_clinician_attention",
)


# ======================================================================================
# 12. MASTER 10-ECG FIGURE
#
# One figure per gaze population.
# Values omitted inside cells to keep the master figure readable.
# ======================================================================================

def make_master_figure(
    population_name,
    gaze_target,
    guided_attention,
    filename,
):

    delta = (
        guided_attention
        -
        base_attention
    )


    absolute_max = max(
        float(
            gaze_target.max()
        ),
        float(
            base_attention.max()
        ),
        float(
            guided_attention.max()
        ),
    )


    delta_abs_max = float(
        np.max(
            np.abs(
                delta
            )
        )
    )


    fig, axes = plt.subplots(
        10,
        4,
        figsize=(
            14,
            28
        )
    )


    last_absolute_image = None
    last_delta_image = None


    for i, ecg in enumerate(
        ECG_KEYS
    ):

        base_cos = cosine(
            base_attention[
                i
            ],
            gaze_target[
                i
            ]
        )


        guided_cos = cosine(
            guided_attention[
                i
            ],
            gaze_target[
                i
            ]
        )


        last_absolute_image = draw_lead_map(
            axes[
                i,
                0
            ],
            gaze_target[
                i
            ],
            (
                f"{ecg}\n"
                f"{population_name} gaze"
            ),
            0,
            absolute_max,
            "viridis",
            show_values=False,
        )


        draw_lead_map(
            axes[
                i,
                1
            ],
            base_attention[
                i
            ],
            (
                "Base"
                f"\ncos={base_cos:.3f}"
            ),
            0,
            absolute_max,
            "viridis",
            show_values=False,
        )


        draw_lead_map(
            axes[
                i,
                2
            ],
            guided_attention[
                i
            ],
            (
                "Gaze-guided"
                f"\ncos={guided_cos:.3f}"
            ),
            0,
            absolute_max,
            "viridis",
            show_values=False,
        )


        last_delta_image = draw_lead_map(
            axes[
                i,
                3
            ],
            delta[
                i
            ],
            (
                "Δ attention"
                f"\nΔcos={guided_cos - base_cos:+.3f}"
            ),
            -delta_abs_max,
            delta_abs_max,
            "RdBu_r",
            show_values=False,
        )


    fig.suptitle(
        (
            f"Clinician gaze and model lead-attention distributions — "
            f"{population_name} supervision"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )


    fig.subplots_adjust(
        left=0.04,
        right=0.94,
        top=0.975,
        bottom=0.03,
        hspace=0.65,
        wspace=0.30,
    )


    # Shared colour bars
    cbar_ax1 = fig.add_axes(
        [
            0.955,
            0.54,
            0.012,
            0.34,
        ]
    )


    cb1 = fig.colorbar(
        last_absolute_image,
        cax=cbar_ax1,
    )

    cb1.set_label(
        "Normalised attention"
    )


    cbar_ax2 = fig.add_axes(
        [
            0.955,
            0.12,
            0.012,
            0.34,
        ]
    )


    cb2 = fig.colorbar(
        last_delta_image,
        cax=cbar_ax2,
    )

    cb2.set_label(
        "Guided − base"
    )


    png_path = (
        OUT_DIR
        /
        f"{filename}.png"
    )


    pdf_path = (
        OUT_DIR
        /
        f"{filename}.pdf"
    )


    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )


    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# ======================================================================================
# 13. CREATE MASTER FIGURES
# ======================================================================================

make_master_figure(
    population_name="Consultant",
    gaze_target=consultant_target,
    guided_attention=consultant_attention,
    filename="MASTER_consultant_gaze_vs_model_attention",
)


make_master_figure(
    population_name="All-clinician",
    gaze_target=all_target,
    guided_attention=all_attention,
    filename="MASTER_all_clinician_gaze_vs_model_attention",
)


# ======================================================================================
# 14. SAVE EXACT NUMERICAL DISTRIBUTIONS
# ======================================================================================

rows = []


for i, ecg in enumerate(
    ECG_KEYS
):

    for j, lead in enumerate(
        STANDARD_LEADS
    ):

        rows.append(
            {
                "ECG":
                    ecg,

                "lead":
                    lead,

                "consultant_gaze":
                    consultant_target[
                        i,
                        j
                    ],

                "all_clinician_gaze":
                    all_target[
                        i,
                        j
                    ],

                "base_attention":
                    base_attention[
                        i,
                        j
                    ],

                "consultant_model_attention":
                    consultant_attention[
                        i,
                        j
                    ],

                "all_clinician_model_attention":
                    all_attention[
                        i,
                        j
                    ],

                "consultant_change_vs_base":
                    consultant_attention[
                        i,
                        j
                    ]
                    -
                    base_attention[
                        i,
                        j
                    ],

                "all_clinician_change_vs_base":
                    all_attention[
                        i,
                        j
                    ]
                    -
                    base_attention[
                        i,
                        j
                    ],
            }
        )


attention_df = pd.DataFrame(
    rows
)


attention_df.to_csv(
    OUT_DIR
    /
    "lead_attention_visualisation_values.csv",
    index=False
)


# ======================================================================================
# 15. SUMMARY
# ======================================================================================

print("\n" + "=" * 100)
print("STEP 6A COMPLETE")
print("=" * 100)


print(
    "\nFigures saved to:"
)

print(
    OUT_DIR
)


print(
    """
Main files:

  MASTER_consultant_gaze_vs_model_attention.png
  MASTER_consultant_gaze_vs_model_attention.pdf

  MASTER_all_clinician_gaze_vs_model_attention.png
  MASTER_all_clinician_gaze_vs_model_attention.pdf

Plus 20 individual ECG figures:
  consultant_attention_*.png / .pdf
  all_clinician_attention_*.png / .pdf

Numerical source data:
  lead_attention_visualisation_values.csv
"""
)
