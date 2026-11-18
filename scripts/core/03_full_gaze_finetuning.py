# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 3
# Stage: final models from all 10 gaze ECGs
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 3 — TRAIN FINAL FULL-DATA GAZE MODELS
#
# Uses ALL 10 gaze ECGs.
#
# Produces:
#
#   definitive_clean_gaze/final_models/
#       clean_consultant.weights.h5
#       clean_all_clinicians.weights.h5
#
# IMPORTANT:
#   - same frozen V9 input
#   - same frozen gaze targets
#   - same RMS add_3 attention readout
#   - only conv2d_10 trainable
#   - BatchNorm frozen
#   - training=False for every forward pass
#   - gaze-only MSE
#   - Adam 1e-4
#   - 25 epochs
#   - 40 noisy copies per ECG
#   - sigma 0.03
#   - batch size 16
#
# Both populations receive IDENTICAL:
#   - starting weights
#   - augmented ECGs
#   - epoch/batch ordering
#
# Only the gaze target differs.
#
# These models are for subsequent downstream diagnostic evaluation.
# ======================================================================================

import gc
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ======================================================================================
# 1. PARAMETERS — MUST MATCH STEP 2
# ======================================================================================

SEED = 2026

AUGMENTATIONS_PER_ECG = 40
NOISE_SD = 0.03

BATCH_SIZE = 16
EPOCHS = 25

LEARNING_RATE = 1e-4
RMS_EPSILON = 1e-8


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


POPULATIONS = [
    "consultant",
    "all_clinicians",
]


# ======================================================================================
# 2. PATHS
# ======================================================================================

ROOT = Path(
    "definitive_clean_gaze"
)


X_PATH = Path(
    "final_digitisation_v9/X_gaze_final.npy"
)


TARGET_PATH = (
    ROOT
    /
    "clean_gaze_targets.npz"
)


STEP1_MANIFEST_PATH = (
    ROOT
    /
    "step1_manifest.json"
)


STEP2_MANIFEST_PATH = (
    ROOT
    /
    "step2_cv"
    /
    "step2_manifest.json"
)


FINAL_DIR = (
    ROOT
    /
    "final_models"
)


FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for path in [
    X_PATH,
    TARGET_PATH,
    STEP1_MANIFEST_PATH,
    STEP2_MANIFEST_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ======================================================================================
# 3. HASH HELPER
# ======================================================================================

def sha256_file(path):

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

            h.update(
                block
            )

    return h.hexdigest()


# ======================================================================================
# 4. LOAD MANIFESTS + VERIFY FROZEN INPUTS
# ======================================================================================

with open(
    STEP1_MANIFEST_PATH,
    "r"
) as f:

    step1_manifest = json.load(
        f
    )


with open(
    STEP2_MANIFEST_PATH,
    "r"
) as f:

    step2_manifest = json.load(
        f
    )


BASE_MODEL_PATH = Path(
    step1_manifest[
        "base_model"
    ]
)


if not BASE_MODEL_PATH.exists():

    raise FileNotFoundError(
        BASE_MODEL_PATH
    )


x_hash = sha256_file(
    X_PATH
)


target_hash = sha256_file(
    TARGET_PATH
)


base_hash = sha256_file(
    BASE_MODEL_PATH
)


print("=" * 100)
print("FROZEN INPUT VERIFICATION")
print("=" * 100)


print(
    "V9 ECG:",
    x_hash
)


print(
    "Targets:",
    target_hash
)


print(
    "Base model:",
    base_hash
)


if (
    x_hash
    !=
    step1_manifest[
        "frozen_ecg_sha256"
    ]
):

    raise RuntimeError(
        "V9 ECG hash mismatch."
    )


if (
    target_hash
    !=
    step1_manifest[
        "clean_gaze_targets_sha256"
    ]
):

    raise RuntimeError(
        "Gaze target hash mismatch."
    )


if (
    base_hash
    !=
    step1_manifest[
        "base_model_sha256"
    ]
):

    raise RuntimeError(
        "Base model hash mismatch."
    )


print(
    "\n✓ Frozen Step-1 inputs verified."
)


# ======================================================================================
# 5. VERIFY STEP-2 TRAINING DEFINITION
# ======================================================================================

required_step2 = {

    "attention_layer":
        "add_3",

    "trainable_layer":
        "conv2d_10",

    "epochs":
        25,

    "augmentations_per_training_ECG":
        40,

    "gaussian_noise_sigma":
        0.03,

    "batch_size":
        16,
}


for key, expected in required_step2.items():

    actual = step2_manifest[
        key
    ]

    if actual != expected:

        raise RuntimeError(
            f"Step-2 definition mismatch for {key}: "
            f"{actual} != {expected}"
        )


print(
    "✓ Step-3 configuration matches frozen Step-2 definition."
)


# ======================================================================================
# 6. LOAD ARRAYS
# ======================================================================================

X = np.load(
    X_PATH
).astype(
    np.float32
)


target_data = np.load(
    TARGET_PATH
)


Y = {

    "consultant":
        target_data[
            "consultant"
        ].astype(
            np.float32
        ),

    "all_clinicians":
        target_data[
            "all_clinicians"
        ].astype(
            np.float32
        ),
}


if X.shape != (
    10,
    12,
    1000,
    1
):

    raise RuntimeError(
        f"Bad X shape: {X.shape}"
    )


for population in POPULATIONS:

    if Y[
        population
    ].shape != (
        10,
        12
    ):

        raise RuntimeError(
            f"Bad target shape: {population}"
        )


# ======================================================================================
# 7. FROZEN RMS ATTENTION READOUT
# ======================================================================================

def attention_distribution_from_add3(
    add3_tensor
):

    rms = tf.sqrt(

        tf.reduce_mean(
            tf.square(
                add3_tensor
            ),
            axis=[
                2,
                3
            ]
        )

        +

        RMS_EPSILON
    )


    return (
        rms
        /
        (
            tf.reduce_sum(
                rms,
                axis=1,
                keepdims=True
            )
            +
            RMS_EPSILON
        )
    )


# ======================================================================================
# 8. LOAD FRESH BASE MODEL
# ======================================================================================

def load_fresh_model():

    tf.keras.backend.clear_session()

    tf.random.set_seed(
        SEED
    )


    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )


    # Freeze everything.
    for layer in model.layers:

        layer.trainable = False


    conv10 = model.get_layer(
        "conv2d_10"
    )


    conv10.trainable = True


    add3 = model.get_layer(
        "add_3"
    )


    rep_model = tf.keras.Model(
        inputs=model.input,
        outputs=add3.output
    )


    train_vars = list(
        conv10.trainable_variables
    )


    if len(
        train_vars
    ) == 0:

        raise RuntimeError(
            "conv2d_10 has no trainable variables."
        )


    return (
        model,
        rep_model,
        train_vars
    )


# ======================================================================================
# 9. WEIGHT AUDIT HELPERS
# ======================================================================================

def snapshot_weights(
    model
):

    return {

        layer.name: [
            np.array(
                w,
                copy=True
            )
            for w
            in layer.get_weights()
        ]

        for layer
        in model.layers
    }


def changed_layers(
    before,
    model
):

    rows = []


    for layer in model.layers:

        old = before[
            layer.name
        ]


        new = layer.get_weights()


        max_change = 0.0


        for a, b in zip(
            old,
            new
        ):

            if a.shape != b.shape:

                raise RuntimeError(
                    f"Shape mismatch: {layer.name}"
                )


            if a.size:

                max_change = max(

                    max_change,

                    float(
                        np.max(
                            np.abs(
                                a - b
                            )
                        )
                    )
                )


        rows.append(
            {
                "layer":
                    layer.name,

                "max_abs_change":
                    max_change,
            }
        )


    return pd.DataFrame(
        rows
    )


# ======================================================================================
# 10. CREATE IDENTICAL FULL-DATA AUGMENTATION ONCE
# ======================================================================================

augmentation_rng = np.random.default_rng(
    SEED + 500000
)


X_train_base = np.repeat(

    X,

    AUGMENTATIONS_PER_ECG,

    axis=0

).astype(
    np.float32
)


noise = augmentation_rng.normal(

    loc=0.0,

    scale=NOISE_SD,

    size=X_train_base.shape

).astype(
    np.float32
)


X_train_augmented = (
    X_train_base
    +
    noise
).astype(
    np.float32
)


if X_train_augmented.shape != (
    400,
    12,
    1000,
    1
):

    raise RuntimeError(
        f"Unexpected augmented X shape: {X_train_augmented.shape}"
    )


print("\n" + "=" * 100)
print("FULL-DATA TRAINING SET")
print("=" * 100)

print(
    "Training examples:",
    X_train_augmented.shape[0]
)

print(
    "40 augmentations × 10 ECGs = 400 confirmed."
)


# ======================================================================================
# 11. IDENTICAL EPOCH/BATCH ORDER FOR BOTH POPULATIONS
# ======================================================================================

permutation_rng = np.random.default_rng(
    SEED + 600000
)


epoch_permutations = [

    permutation_rng.permutation(
        400
    )

    for _
    in range(
        EPOCHS
    )
]


# ======================================================================================
# 12. TRAIN FINAL MODELS
# ======================================================================================

history_rows = []

weight_audit_rows = []

final_alignment_rows = []

final_weight_hashes = {}


for population in POPULATIONS:

    print(
        "\n\n" +
        "#" * 100
    )

    print(
        f"FINAL MODEL — {population.upper()}"
    )

    print(
        "#" * 100
    )


    (
        model,
        rep_model,
        train_vars
    ) = load_fresh_model()


    before = snapshot_weights(
        model
    )


    # ------------------------------------------------------------------
    # Target augmentation
    # ------------------------------------------------------------------

    Y_augmented = np.repeat(

        Y[
            population
        ],

        AUGMENTATIONS_PER_ECG,

        axis=0

    ).astype(
        np.float32
    )


    if Y_augmented.shape != (
        400,
        12
    ):

        raise RuntimeError(
            f"Unexpected Y shape for {population}: "
            f"{Y_augmented.shape}"
        )


    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )


    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        permutation = epoch_permutations[
            epoch
        ]


        losses = []


        for start in range(
            0,
            400,
            BATCH_SIZE
        ):

            idx = permutation[
                start:
                start + BATCH_SIZE
            ]


            xb = tf.convert_to_tensor(
                X_train_augmented[
                    idx
                ],
                dtype=tf.float32
            )


            yb = tf.convert_to_tensor(
                Y_augmented[
                    idx
                ],
                dtype=tf.float32
            )


            with tf.GradientTape() as tape:

                add3 = rep_model(
                    xb,
                    training=False
                )


                pred = (
                    attention_distribution_from_add3(
                        add3
                    )
                )


                loss = tf.reduce_mean(
                    tf.square(
                        pred - yb
                    )
                )


            gradients = tape.gradient(
                loss,
                train_vars
            )


            if any(
                g is None
                for g
                in gradients
            ):

                raise RuntimeError(
                    "None gradient detected."
                )


            for gradient in gradients:

                if not np.all(
                    np.isfinite(
                        gradient.numpy()
                    )
                ):

                    raise RuntimeError(
                        "Non-finite gradient detected."
                    )


            optimizer.apply_gradients(
                zip(
                    gradients,
                    train_vars
                )
            )


            losses.append(
                float(
                    loss.numpy()
                )
            )


        mean_loss = float(
            np.mean(
                losses
            )
        )


        history_rows.append(
            {
                "population":
                    population,

                "epoch":
                    epoch + 1,

                "loss":
                    mean_loss,
            }
        )


        print(
            f"epoch {epoch + 1:02d}/{EPOCHS}: "
            f"loss={mean_loss:.8f}"
        )


    # ------------------------------------------------------------------
    # Weight-change audit
    # ------------------------------------------------------------------

    audit = changed_layers(
        before,
        model
    )


    changed = set(
        audit.loc[
            audit[
                "max_abs_change"
            ]
            >
            1e-12,
            "layer"
        ]
    )


    if changed != {
        "conv2d_10"
    }:

        raise RuntimeError(
            f"{population}: unexpected changed layers: {changed}"
        )


    print(
        "\n✓ ONLY conv2d_10 changed."
    )


    audit[
        "population"
    ] = population


    weight_audit_rows.extend(
        audit.to_dict(
            "records"
        )
    )


    # ------------------------------------------------------------------
    # Training-set alignment QC
    #
    # This is IN-SAMPLE and must not be interpreted as validation.
    # ------------------------------------------------------------------

    final_attention = (
        attention_distribution_from_add3(
            rep_model(
                X,
                training=False
            )
        )
        .numpy()
    )


    for i, ecg in enumerate(
        ECG_ORDER
    ):

        target = Y[
            population
        ][i]


        prediction = final_attention[
            i
        ]


        cosine = float(
            np.dot(
                target,
                prediction
            )
            /
            (
                np.linalg.norm(
                    target
                )
                *
                np.linalg.norm(
                    prediction
                )
            )
        )


        mse = float(
            np.mean(
                np.square(
                    target
                    -
                    prediction
                )
            )
        )


        final_alignment_rows.append(
            {
                "population":
                    population,

                "ECG":
                    ecg,

                "cosine":
                    cosine,

                "MSE":
                    mse,

                "interpretation":
                    "in-sample training QC only",
            }
        )


    # ------------------------------------------------------------------
    # Save final weights
    # ------------------------------------------------------------------

    if population == "consultant":

        weight_path = (
            FINAL_DIR
            /
            "clean_consultant.weights.h5"
        )

    else:

        weight_path = (
            FINAL_DIR
            /
            "clean_all_clinicians.weights.h5"
        )


    model.save_weights(
        weight_path
    )


    final_weight_hashes[
        population
    ] = sha256_file(
        weight_path
    )


    print(
        "\nSaved:"
    )

    print(
        weight_path
    )


    print(
        "SHA256:"
    )

    print(
        final_weight_hashes[
            population
        ]
    )


    gc.collect()


# ======================================================================================
# 13. SAVE AUDITS
# ======================================================================================

history_df = pd.DataFrame(
    history_rows
)


weight_audit_df = pd.DataFrame(
    weight_audit_rows
)


alignment_df = pd.DataFrame(
    final_alignment_rows
)


history_df.to_csv(

    FINAL_DIR
    /
    "full_data_training_history.csv",

    index=False
)


weight_audit_df.to_csv(

    FINAL_DIR
    /
    "full_data_weight_audit.csv",

    index=False
)


alignment_df.to_csv(

    FINAL_DIR
    /
    "full_data_alignment_QC.csv",

    index=False
)


# ======================================================================================
# 14. PRINT TRAINING-QC SUMMARY
# ======================================================================================

print(
    "\n\n" +
    "=" * 100
)

print(
    "FINAL FULL-DATA MODEL QC"
)

print(
    "=" * 100
)


qc_summary = (

    alignment_df

    .groupby(
        "population"
    )

    .agg(
        mean_cosine=(
            "cosine",
            "mean"
        ),

        min_cosine=(
            "cosine",
            "min"
        ),

        mean_MSE=(
            "MSE",
            "mean"
        ),
    )

    .reset_index()
)


print(
    qc_summary
    .round(
        7
    )
    .to_string(
        index=False
    )
)


print(
    "\nNOTE: These are IN-SAMPLE full-data training QC values."
)

print(
    "The valid generalisation estimate remains the 5-fold held-out results from Step 2."
)


# ======================================================================================
# 15. CHANGED-LAYER SUMMARY
# ======================================================================================

changed_summary = (

    weight_audit_df[
        weight_audit_df[
            "max_abs_change"
        ]
        >
        1e-12
    ]

    [
        [
            "population",
            "layer",
            "max_abs_change"
        ]
    ]
)


print(
    "\n" +
    "=" * 100
)

print(
    "FINAL MODEL LAYERS THAT CHANGED"
)

print(
    "=" * 100
)


print(
    changed_summary.to_string(
        index=False
    )
)


# ======================================================================================
# 16. FINAL-MODEL MANIFEST
# ======================================================================================

manifest = {

    "experiment":
        "Definitive clean gaze-aligned ECG experiment",

    "stage":
        "Step 3: final full-data gaze models",

    "training_ECG_count":
        10,

    "attention_readout":
        (
            "RMS activation energy from add_3 across "
            "temporal and channel axes, normalised across 12 leads."
        ),

    "trainable_layer":
        "conv2d_10",

    "loss":
        "gaze-only mean squared error",

    "epochs":
        EPOCHS,

    "augmentations_per_ECG":
        AUGMENTATIONS_PER_ECG,

    "gaussian_noise_sigma":
        NOISE_SD,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "same_augmented_inputs_between_populations":
        True,

    "same_batch_order_between_populations":
        True,

    "consultant_weights":
        str(
            FINAL_DIR
            /
            "clean_consultant.weights.h5"
        ),

    "consultant_weights_sha256":
        final_weight_hashes[
            "consultant"
        ],

    "all_clinician_weights":
        str(
            FINAL_DIR
            /
            "clean_all_clinicians.weights.h5"
        ),

    "all_clinician_weights_sha256":
        final_weight_hashes[
            "all_clinicians"
        ],

    "frozen_V9_sha256":
        x_hash,

    "frozen_target_sha256":
        target_hash,

    "base_model_sha256":
        base_hash,

    "generalisation_results_source":
        (
            "Step 2 five-fold held-out-ECG evaluation; "
            "full-data alignment values are training QC only."
        ),
}


with open(

    FINAL_DIR
    /
    "final_model_manifest.json",

    "w"

) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ======================================================================================
# 17. FINISH
# ======================================================================================

print(
    "\n\n" +
    "=" * 100
)

print(
    "STEP 3 COMPLETE"
)

print(
    "=" * 100
)


print(
    """
Please paste back only:

1. FINAL FULL-DATA MODEL QC

2. FINAL MODEL LAYERS THAT CHANGED

3. The SHA256 values printed for:
       clean_consultant.weights.h5
       clean_all_clinicians.weights.h5

After that we freeze the two final gaze models and move to
downstream diagnostic evaluation.
"""
)
