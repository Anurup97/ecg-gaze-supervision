# EXTRACTED FROM HISTORICAL RECONSTRUCTION NOTEBOOK, CELL 4.
# PORTABILITY MODIFICATION ONLY: original machine-specific home path replaced by
# annotations/local_final_patches (local-only; missing final V2 annotations).
# Requires frozen V8 waveform and final annotation JSON, NOT included here.
# DO NOT run this if you already have the verified V9 input.

# ======================================================================================
# FINAL DIGITISATION PATCH — V9
#
# Parent:
#     final_digitisation_v8/X_gaze_final.npy
#
# New annotation:
#     annotations/local_final_patches/ecg_anchor_annotations_LBBB_V2_final*.json
#
# Replaces ONLY:
#     LBBB V2
#
# Output:
#     final_digitisation_v9/
#         X_gaze_final.npy
#         FINAL_OVERLAY_LBBB_V2.png
#         FINAL_SIGNAL_LBBB.png
#         final_LBBB_V2_metrics.csv
#         metadata.json
#
# DO NOT START MODEL TRAINING UNTIL THIS FINAL OVERLAY IS CHECKED.
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import json
import hashlib
import warnings

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from scipy.ndimage import gaussian_filter
from scipy.interpolate import PchipInterpolator


warnings.filterwarnings("ignore")


# ======================================================================================
# 1. PATHS
# ======================================================================================

OUT_DIR = Path(
    "final_digitisation_v9"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PARENT_PATH = Path(
    "final_digitisation_v8/X_gaze_final.npy"
)


GEOMETRY_PATH = Path(
    "manual_digitization_qc/manual_geometry.json"
)


DOWNLOADS = Path(
    "annotations/local_final_patches"
)


json_candidates = list(
    DOWNLOADS.glob(
        "ecg_anchor_annotations_LBBB_V2_final*.json"
    )
)


if not json_candidates:

    raise FileNotFoundError(
        "Could not find "
        "ecg_anchor_annotations_LBBB_V2_final*.json "
        "in annotations/local_final_patches/"
    )


ANNOTATION_PATH = max(
    json_candidates,
    key=lambda p: p.stat().st_mtime
)


for p in [
    PARENT_PATH,
    GEOMETRY_PATH,
    ANNOTATION_PATH
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


print("=" * 100)
print("FINAL V9 INPUTS")
print("=" * 100)

print(
    "Parent V8:",
    PARENT_PATH
)

print(
    "Geometry: ",
    GEOMETRY_PATH
)

print(
    "V2 JSON:  ",
    ANNOTATION_PATH
)


# ======================================================================================
# 2. LOAD
# ======================================================================================

X_v8 = np.load(
    PARENT_PATH
).astype(
    np.float32
)


if X_v8.shape != (
    10,
    12,
    1000,
    1
):

    raise ValueError(
        f"Unexpected parent array shape: {X_v8.shape}"
    )


X_final = X_v8.copy()


with open(
    GEOMETRY_PATH,
    "r"
) as f:

    geometry = json.load(
        f
    )


with open(
    ANNOTATION_PATH,
    "r"
) as f:

    annotation_json = json.load(
        f
    )


annotations = annotation_json.get(
    "annotations",
    {}
)


KEY = "LBBB::V2"


if KEY not in annotations:

    raise RuntimeError(
        f"{KEY} not found in annotation JSON."
    )


ann = annotations[
    KEY
]


# ======================================================================================
# 3. FIXED AXIS ORDERS
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


LBBB_IDX = ECG_ORDER.index(
    "LBBB"
)


V2_IDX = LEAD_ORDER.index(
    "V2"
)


# ======================================================================================
# 4. VERIFY FINAL ANNOTATION
# ======================================================================================

status = ann.get(
    "status"
)


n_points = len(
    ann.get(
        "points",
        []
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "ANNOTATION CHECK"
)

print(
    "=" * 100
)


print(
    "Status:",
    status
)

print(
    "Anchors:",
    n_points
)


if status != "corrected":

    raise RuntimeError(
        "Final LBBB V2 annotation is not marked corrected."
    )


if n_points < 20:

    raise RuntimeError(
        f"Expected >=20 anchors, found {n_points}."
    )


print(
    "\n✓ Final LBBB V2 annotation is complete."
)


# ======================================================================================
# 5. SOURCE IMAGE HELPERS
# ======================================================================================

def resolve_source_image(
    stored_path
):

    p = Path(
        stored_path
    )


    if p.exists():

        return p


    matches = list(
        Path(".").glob(
            f"**/{p.name}"
        )
    )


    if not matches:

        raise FileNotFoundError(
            f"Could not locate source ECG image: {p.name}"
        )


    return matches[
        0
    ]


def load_gray(
    path
):

    return (
        np.asarray(
            Image.open(
                path
            )
            .convert(
                "L"
            )
        )
        .astype(
            np.float32
        )
        /
        255.0
    )


# ======================================================================================
# 6. MAP BROWSER POINTS TO ORIGINAL ULTRA-DEEP CROP
# ======================================================================================

def browser_points_to_crop(
    ann
):

    points = np.asarray(

        [
            [
                float(
                    p["x"]
                ),
                float(
                    p["y"]
                )
            ]

            for p in ann[
                "points"
            ]
        ],

        dtype=np.float32
    )


    display_w = float(
        ann[
            "display_width"
        ]
    )


    display_h = float(
        ann[
            "display_height"
        ]
    )


    original_w = float(
        ann[
            "original_crop_width"
        ]
    )


    original_h = float(
        ann[
            "original_crop_height"
        ]
    )


    points[
        :,
        0
    ] *= (
        original_w
        /
        display_w
    )


    points[
        :,
        1
    ] *= (
        original_h
        /
        display_h
    )


    points = points[
        np.argsort(
            points[
                :,
                0
            ]
        )
    ]


    # Remove near-duplicate x coordinates
    cleaned = [
        points[
            0
        ]
    ]


    for p in points[
        1:
    ]:

        if (
            p[
                0
            ]
            -
            cleaned[
                -1
            ][
                0
            ]
            >=
            0.5
        ):

            cleaned.append(
                p
            )


    points = np.asarray(
        cleaned,
        dtype=np.float32
    )


    if len(
        points
    ) < 18:

        raise RuntimeError(
            "Too few unique anchors after coordinate conversion."
        )


    return points


# ======================================================================================
# 7. ECG INK EVIDENCE MAP
# ======================================================================================

def ecg_score_map(
    gray
):

    gray = np.asarray(
        gray,
        dtype=np.float32
    )


    p1 = np.percentile(
        gray,
        1
    )


    p99 = np.percentile(
        gray,
        99
    )


    if p99 > p1 + 1e-8:

        norm = (
            gray
            -
            p1
        ) / (
            p99
            -
            p1
        )

    else:

        norm = gray.copy()


    norm = np.clip(
        norm,
        0,
        1
    )


    darkness = (
        1.0
        -
        norm
    )


    fine = gaussian_filter(
        norm,
        sigma=0.45
    )


    background = gaussian_filter(
        norm,
        sigma=2.2
    )


    ridge = np.clip(
        background
        -
        fine,
        0,
        None
    )


    ridge_scale = np.percentile(
        ridge,
        99.5
    )


    if ridge_scale > 1e-8:

        ridge /= ridge_scale


    ridge = np.clip(
        ridge,
        0,
        1.5
    )


    score = (
        0.75
        *
        ridge
        +
        0.25
        *
        darkness
    )


    score_scale = np.percentile(
        score,
        99.5
    )


    if score_scale > 1e-8:

        score /= score_scale


    return np.clip(
        score,
        0,
        1.5
    )


# ======================================================================================
# 8. FULL-WIDTH MANUAL GUIDE
# ======================================================================================

def make_full_width_guide(
    anchors,
    width
):

    xs = np.arange(
        width,
        dtype=np.float32
    )


    first_x = float(
        anchors[
            0,
            0
        ]
    )


    last_x = float(
        anchors[
            -1,
            0
        ]
    )


    first_y = float(
        anchors[
            0,
            1
        ]
    )


    last_y = float(
        anchors[
            -1,
            1
        ]
    )


    interpolator = PchipInterpolator(

        anchors[
            :,
            0
        ],

        anchors[
            :,
            1
        ],

        extrapolate=False
    )


    guide = np.empty(
        width,
        dtype=np.float32
    )


    inside = (
        (
            xs
            >=
            first_x
        )
        &
        (
            xs
            <=
            last_x
        )
    )


    guide[
        inside
    ] = interpolator(
        xs[
            inside
        ]
    ).astype(
        np.float32
    )


    # Preserve full horizontal duration.
    guide[
        xs
        <
        first_x
    ] = first_y


    guide[
        xs
        >
        last_x
    ] = last_y


    return (
        xs,
        guide
    )


# ======================================================================================
# 9. FINAL ONE-LEAD RECONSTRUCTION
# ======================================================================================

def reconstruct_final_v2(
    crop,
    anchors
):

    H, W = crop.shape


    anchors = np.asarray(
        anchors,
        dtype=np.float32
    )


    anchors[
        :,
        0
    ] = np.clip(
        anchors[
            :,
            0
        ],
        0,
        W - 1
    )


    anchors[
        :,
        1
    ] = np.clip(
        anchors[
            :,
            1
        ],
        0,
        H - 1
    )


    first_fraction = (
        anchors[
            0,
            0
        ]
        /
        max(
            W - 1,
            1
        )
    )


    last_fraction = (
        anchors[
            -1,
            0
        ]
        /
        max(
            W - 1,
            1
        )
    )


    span_fraction = (
        anchors[
            -1,
            0
        ]
        -
        anchors[
            0,
            0
        ]
    ) / max(
        W - 1,
        1
    )


    # Final annotator required stricter coverage.
    if first_fraction > 0.09:

        raise RuntimeError(
            f"First anchor too far from left edge: {first_fraction:.3f}"
        )


    if last_fraction < 0.91:

        raise RuntimeError(
            f"Last anchor too far from right edge: {last_fraction:.3f}"
        )


    if span_fraction < 0.82:

        raise RuntimeError(
            f"Insufficient horizontal coverage: {span_fraction:.3f}"
        )


    xs, guide = make_full_width_guide(
        anchors,
        W
    )


    guide = np.clip(
        guide,
        0,
        H - 1
    )


    score = ecg_score_map(
        crop
    )


    # The manual path remains dominant.
    snap_radius = min(
        6,
        max(
            3,
            int(
                round(
                    0.025
                    *
                    H
                )
            )
        )
    )


    path = np.zeros(
        W,
        dtype=np.float32
    )


    first_x = float(
        anchors[
            0,
            0
        ]
    )


    last_x = float(
        anchors[
            -1,
            0
        ]
    )


    previous_y = None


    for x in range(
        W
    ):

        gy = float(
            guide[
                x
            ]
        )


        centre = int(
            round(
                gy
            )
        )


        if (
            x
            <
            first_x
            or
            x
            >
            last_x
        ):

            radius = min(
                3,
                snap_radius
            )

            distance_weight = 0.35

        else:

            radius = snap_radius

            distance_weight = 0.22


        lo = max(
            0,
            centre
            -
            radius
        )


        hi = min(
            H - 1,
            centre
            +
            radius
        )


        ys = np.arange(
            lo,
            hi + 1,
            dtype=int
        )


        objective = (

            2.0
            *
            score[
                ys,
                x
            ]

            -

            distance_weight
            *
            np.abs(
                ys
                -
                gy
            )
        )


        # Very weak continuity preference.
        if previous_y is not None:

            objective -= (
                0.015
                *
                np.abs(
                    ys
                    -
                    previous_y
                )
            )


        best_y = ys[
            np.argmax(
                objective
            )
        ]


        path[
            x
        ] = float(
            best_y
        )


        previous_y = float(
            best_y
        )


    # Exact agreement at manually annotated x positions.
    for ax, ay in anchors:

        xi = int(
            np.clip(
                round(
                    ax
                ),
                0,
                W - 1
            )
        )


        path[
            xi
        ] = float(
            ay
        )


    # ==================================================================================
    # ECG SIGNAL
    # ==================================================================================

    signal = (
        -path
    )


    signal -= np.median(
        signal
    )


    signal_1000 = np.interp(

        np.linspace(
            0,
            1,
            1000
        ),

        np.linspace(
            0,
            1,
            W
        ),

        signal

    ).astype(
        np.float32
    )


    signal_1000 -= np.mean(
        signal_1000
    )


    sd = np.std(
        signal_1000
    )


    if sd > 1e-8:

        signal_1000 /= sd


    # ==================================================================================
    # FINAL EDGE QC
    # ==================================================================================

    distance_to_top = path


    distance_to_bottom = (
        H
        -
        1
        -
        path
    )


    distance_to_edge = np.minimum(
        distance_to_top,
        distance_to_bottom
    )


    # Same definition used in previous QC
    edge_margin = max(
        3,
        int(
            round(
                0.02
                *
                H
            )
        )
    )


    top_contact = (
        path
        <=
        edge_margin
    )


    bottom_contact = (
        path
        >=
        H
        -
        1
        -
        edge_margin
    )


    edge_contact = (
        top_contact
        |
        bottom_contact
    )


    longest_run = 0
    current_run = 0


    for contact in edge_contact:

        if contact:

            current_run += 1

            longest_run = max(
                longest_run,
                current_run
            )

        else:

            current_run = 0


    deviation = np.abs(
        path
        -
        guide
    )


    y_indices = np.clip(
        np.round(
            path
        ).astype(
            int
        ),
        0,
        H - 1
    )


    path_scores = score[
        y_indices,
        np.arange(
            W
        )
    ]


    metrics = {

        "n_anchors":
            int(
                len(
                    anchors
                )
            ),

        "crop_width":
            int(
                W
            ),

        "crop_height":
            int(
                H
            ),

        "first_anchor_fraction":
            float(
                first_fraction
            ),

        "last_anchor_fraction":
            float(
                last_fraction
            ),

        "annotation_span_fraction":
            float(
                span_fraction
            ),

        "reconstruction_span_fraction":
            1.0,

        "snap_radius_px":
            int(
                snap_radius
            ),

        "median_snap_distance_px":
            float(
                np.median(
                    deviation
                )
            ),

        "p95_snap_distance_px":
            float(
                np.percentile(
                    deviation,
                    95
                )
            ),

        "minimum_distance_to_edge_px":
            float(
                np.min(
                    distance_to_edge
                )
            ),

        "minimum_distance_to_bottom_px":
            float(
                np.min(
                    distance_to_bottom
                )
            ),

        "top_edge_contact_fraction":
            float(
                np.mean(
                    top_contact
                )
            ),

        "bottom_edge_contact_fraction":
            float(
                np.mean(
                    bottom_contact
                )
            ),

        "any_edge_contact_fraction":
            float(
                np.mean(
                    edge_contact
                )
            ),

        "longest_edge_contact_run_px":
            int(
                longest_run
            ),

        "mean_path_score":
            float(
                np.mean(
                    path_scores
                )
            ),

        "median_path_score":
            float(
                np.median(
                    path_scores
                )
            ),
    }


    return {

        "x":
            np.arange(
                W,
                dtype=np.float32
            ),

        "y":
            path.astype(
                np.float32
            ),

        "guide":
            guide.astype(
                np.float32
            ),

        "anchors":
            anchors.astype(
                np.float32
            ),

        "signal":
            signal_1000.astype(
                np.float32
            ),

        "metrics":
            metrics,
    }


# ======================================================================================
# 10. LOAD ULTRA-DEEP SOURCE CROP
# ======================================================================================

source_path = resolve_source_image(
    geometry[
        "LBBB"
    ][
        "image_path"
    ]
)


source_gray = load_gray(
    source_path
)


box = [
    int(
        round(
            value
        )
    )
    for value
    in ann[
        "original_box"
    ]
]


x0, y0, x1, y1 = (
    box
)


crop = source_gray[
    y0:y1,
    x0:x1
]


anchors = browser_points_to_crop(
    ann
)


print(
    "\nFinal ultra-deep crop shape:",
    crop.shape
)


print(
    "Unique manual anchors:",
    len(
        anchors
    )
)


# ======================================================================================
# 11. RECONSTRUCT LBBB V2
# ======================================================================================

result = reconstruct_final_v2(
    crop=crop,
    anchors=anchors
)


# Replace ONLY LBBB V2.
X_final[
    LBBB_IDX,
    V2_IDX,
    :,
    0
] = result[
    "signal"
]


# ======================================================================================
# 12. PROVE ONLY ONE LEAD CHANGED
# ======================================================================================

changed = []


for ei, ecg in enumerate(
    ECG_ORDER
):

    for li, lead in enumerate(
        LEAD_ORDER
    ):

        max_diff = float(
            np.max(
                np.abs(
                    X_final[
                        ei,
                        li,
                        :,
                        0
                    ]
                    -
                    X_v8[
                        ei,
                        li,
                        :,
                        0
                    ]
                )
            )
        )


        if max_diff > 1e-7:

            changed.append(
                {
                    "ECG":
                        ecg,

                    "lead":
                        lead,

                    "max_abs_change":
                        max_diff
                }
            )


changed_df = pd.DataFrame(
    changed
)


print(
    "\n" +
    "=" * 100
)

print(
    "CHANGED LEADS VS V8"
)

print(
    "=" * 100
)


print(
    changed_df.to_string(
        index=False
    )
)


actual_changed = {
    (
        row[
            "ECG"
        ],
        row[
            "lead"
        ]
    )
    for row
    in changed
}


expected_changed = {
    (
        "LBBB",
        "V2"
    )
}


if actual_changed != expected_changed:

    raise RuntimeError(
        "\nUnexpected changes detected.\n"
        f"Expected: {expected_changed}\n"
        f"Actual:   {actual_changed}"
    )


print(
    "\n✓ Exactly ONE signal changed: LBBB V2."
)

print(
    "✓ The other 119 ECG-lead waveforms are identical to V8."
)


# ======================================================================================
# 13. FINAL QC TABLE
# ======================================================================================

metrics_df = pd.DataFrame(
    [
        {
            "ECG":
                "LBBB",

            "lead":
                "V2",

            **result[
                "metrics"
            ]
        }
    ]
)


metrics_df[
    "final_edge_flag"
] = (

    (
        metrics_df[
            "any_edge_contact_fraction"
        ]
        >
        0.01
    )

    |

    (
        metrics_df[
            "minimum_distance_to_edge_px"
        ]
        <
        3
    )

    |

    (
        metrics_df[
            "longest_edge_contact_run_px"
        ]
        >
        5
    )
)


metrics_df.to_csv(

    OUT_DIR
    /
    "final_LBBB_V2_metrics.csv",

    index=False
)


print(
    "\n" +
    "=" * 100
)

print(
    "FINAL LBBB V2 EDGE QC"
)

print(
    "=" * 100
)


print(
    metrics_df[
        [
            "lead",
            "n_anchors",
            "crop_height",
            "first_anchor_fraction",
            "last_anchor_fraction",
            "annotation_span_fraction",
            "minimum_distance_to_edge_px",
            "minimum_distance_to_bottom_px",
            "top_edge_contact_fraction",
            "bottom_edge_contact_fraction",
            "any_edge_contact_fraction",
            "longest_edge_contact_run_px",
            "final_edge_flag",
        ]
    ]
    .to_string(
        index=False
    )
)


# ======================================================================================
# 14. FINAL V2 OVERLAY
# ======================================================================================

fig, axes = plt.subplots(

    1,
    2,

    figsize=(
        18,
        6
    ),

    constrained_layout=True
)


# --------------------------------------------------------------------------------------
# Source + path
# --------------------------------------------------------------------------------------

ax = axes[
    0
]


ax.imshow(
    crop,
    cmap="gray",
    origin="upper",
    aspect="auto"
)


ax.plot(
    result[
        "x"
    ],
    result[
        "guide"
    ],
    linestyle="--",
    linewidth=1.0,
    label="manual full-width guide"
)


ax.plot(
    result[
        "x"
    ],
    result[
        "y"
    ],
    color="red",
    linewidth=1.6,
    label="final path"
)


ax.scatter(
    result[
        "anchors"
    ][
        :,
        0
    ],
    result[
        "anchors"
    ][
        :,
        1
    ],
    s=20,
    facecolors="none",
    edgecolors="yellow",
    linewidths=1,
    label="manual anchors"
)


ax.set_xlim(
    0,
    crop.shape[
        1
    ]
    -
    1
)


ax.set_ylim(
    crop.shape[
        0
    ]
    -
    1,
    0
)


ax.set_title(
    "LBBB V2: FINAL ultra-deep source + path"
)


ax.set_xticks([])
ax.set_yticks([])


ax.legend(
    fontsize=8,
    loc="upper right"
)


# --------------------------------------------------------------------------------------
# Final pseudo-waveform
# --------------------------------------------------------------------------------------

ax = axes[
    1
]


ax.plot(
    X_final[
        LBBB_IDX,
        V2_IDX,
        :,
        0
    ],
    linewidth=1.0
)


ax.axhline(
    0,
    linewidth=0.4,
    alpha=0.3
)


ax.set_xlim(
    0,
    999
)


ax.set_title(
    "LBBB V2: FINAL full-duration pseudo-waveform"
)


ax.set_xticks([])
ax.set_yticks([])


fig.suptitle(
    "LBBB V2 — ABSOLUTE FINAL DIGITISATION QC",
    fontsize=18
)


fig.savefig(

    OUT_DIR
    /
    "FINAL_OVERLAY_LBBB_V2.png",

    dpi=220,

    bbox_inches="tight"
)


plt.close(
    fig
)


# ======================================================================================
# 15. FINAL LBBB 12-LEAD SHEET
# ======================================================================================

DISPLAY_GRID = [

    ["I",   "aVR", "V1", "V4"],

    ["II",  "aVL", "V2", "V5"],

    ["III", "aVF", "V3", "V6"],
]


fig, axes = plt.subplots(

    3,
    4,

    figsize=(
        18,
        10
    ),

    constrained_layout=True
)


for r in range(
    3
):

    for c in range(
        4
    ):

        lead = DISPLAY_GRID[
            r
        ][
            c
        ]


        li = LEAD_ORDER.index(
            lead
        )


        sig = X_final[
            LBBB_IDX,
            li,
            :,
            0
        ]


        ax = axes[
            r,
            c
        ]


        ax.plot(
            sig,
            linewidth=1
        )


        ax.axhline(
            0,
            linewidth=0.4,
            alpha=0.3
        )


        if lead == "V2":

            suffix = (
                " [V9 FINAL]"
            )

        elif lead in [
            "V1",
            "V3"
        ]:

            suffix = (
                " [V8 FINAL]"
            )

        else:

            suffix = (
                " [RETAINED]"
            )


        ax.set_title(
            lead
            +
            suffix
        )


        ax.set_xlim(
            0,
            999
        )


        ax.set_xticks([])
        ax.set_yticks([])


fig.suptitle(
    "LBBB — FINAL V9 12-lead candidate",
    fontsize=18
)


fig.savefig(

    OUT_DIR
    /
    "FINAL_SIGNAL_LBBB.png",

    dpi=200,

    bbox_inches="tight"
)


plt.close(
    fig
)


# ======================================================================================
# 16. SAVE FINAL DATASET
# ======================================================================================

if not np.all(
    np.isfinite(
        X_final
    )
):

    raise RuntimeError(
        "Non-finite values detected in final dataset."
    )


FINAL_PATH = (
    OUT_DIR
    /
    "X_gaze_final.npy"
)


np.save(
    FINAL_PATH,
    X_final
)


# ======================================================================================
# 17. HASHES
# ======================================================================================

def sha256_file(
    path
):

    h = hashlib.sha256()


    with open(
        path,
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024
                *
                1024
            )


            if not block:

                break


            h.update(
                block
            )


    return h.hexdigest()


parent_hash = sha256_file(
    PARENT_PATH
)


final_hash = sha256_file(
    FINAL_PATH
)


# ======================================================================================
# 18. METADATA
# ======================================================================================

metadata = {

    "created":
        datetime.now().isoformat(),

    "parent_dataset":
        str(
            PARENT_PATH
        ),

    "parent_sha256":
        parent_hash,

    "final_annotation_json":
        str(
            ANNOTATION_PATH
        ),

    "final_dataset":
        str(
            FINAL_PATH
        ),

    "final_sha256":
        final_hash,

    "shape":
        list(
            X_final.shape
        ),

    "ecg_order":
        ECG_ORDER,

    "lead_order":
        LEAD_ORDER,

    "final_replaced_signal":
        "LBBB V2",

    "number_signals_changed_from_v8":
        1,

    "method": (
        "Final replacement of LBBB V2 using an ultra-deep asymmetric "
        "source-image crop. Dense manually placed waveform anchors defined "
        "a shape-preserving PCHIP trajectory across the diagnostic segment, "
        "with local image snapping restricted to at most six vertical pixels. "
        "The complete crop width was mapped to 1000 samples to preserve "
        "horizontal segment duration. All other 119 ECG-lead pseudo-waveforms "
        "were retained exactly from the V8 parent dataset."
    ),

    "model_training_performed":
        False,
}


with open(

    OUT_DIR
    /
    "metadata.json",

    "w"

) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ======================================================================================
# 19. FINAL REPORT
# ======================================================================================

print(
    "\n" +
    "=" * 100
)

print(
    "V9 FINAL ONE-LEAD PATCH COMPLETE"
)

print(
    "=" * 100
)


print(
    "\nCandidate final dataset:"
)

print(
    FINAL_PATH
)


print(
    "\nShape:"
)

print(
    X_final.shape
)


print(
    "\nParent V8 SHA256:"
)

print(
    parent_hash
)


print(
    "\nCandidate V9 SHA256:"
)

print(
    final_hash
)


print(
    """
NOW SEND ME ONLY TWO THINGS:

1. The image:

       final_digitisation_v9/FINAL_OVERLAY_LBBB_V2.png

2. The printed table:

       FINAL LBBB V2 EDGE QC

If the waveform is visibly complete and the numerical QC confirms that
the deep QRS no longer reaches the crop boundary, we freeze:

       final_digitisation_v9/X_gaze_final.npy

No further ECG digitisation iterations after that.
"""
)
