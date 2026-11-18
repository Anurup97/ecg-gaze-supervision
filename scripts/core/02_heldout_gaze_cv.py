# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 1
# Stage: five-fold held-out gaze cross-validation
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# DEFINITIVE CLEAN GAZE EXPERIMENT
# STEP 2 — 5-FOLD HELD-OUT-ECG GAZE ALIGNMENT
#
# FROZEN:
#   ECG input: final_digitisation_v9/X_gaze_final.npy
#   Targets: definitive_clean_gaze/clean_gaze_targets.npz
#
# Attention:
#   add_3 -> RMS over (time, channels) -> normalise across 12 leads
#
# Training:
#   5 folds, 2 held-out ECGs/fold
#   only conv2d_10 trainable
#   BatchNorm frozen
#   training=False throughout forward passes
#   gaze-only MSE
#   Adam 1e-4
#   25 epochs
#   40 noisy copies/training ECG
#   Gaussian noise SD = 0.03
#   batch size = 16
#
# Consultant and all-clinician arms use identical:
#   folds
#   starting weights
#   augmented X
#   batch order
#
# Only the gaze target differs.
# ======================================================================================

import gc
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import KFold
from scipy.stats import wilcoxon


# ======================================================================================
# 1. PARAMETERS
# ======================================================================================

SEED = 2026
N_SPLITS = 5

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
# 2. REPRODUCIBILITY
# ======================================================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("TensorFlow deterministic operations enabled.")
except Exception as e:
    print("Could not explicitly enable deterministic operations:", repr(e))


# ======================================================================================
# 3. PATHS
# ======================================================================================

ROOT = Path("definitive_clean_gaze")

STEP1_MANIFEST = ROOT / "step1_manifest.json"
TARGET_PATH = ROOT / "clean_gaze_targets.npz"

X_PATH = Path(
    "final_digitisation_v9/X_gaze_final.npy"
)

STEP2_DIR = ROOT / "step2_cv"
STEP2_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR = STEP2_DIR / "fold_weights"
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for p in [
    STEP1_MANIFEST,
    TARGET_PATH,
    X_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


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


# ======================================================================================
# 5. VERIFY FROZEN FILES
# ======================================================================================

with open(
    STEP1_MANIFEST,
    "r"
) as f:

    step1_manifest = json.load(f)


BASE_MODEL_PATH = Path(
    step1_manifest["base_model"]
)


if not BASE_MODEL_PATH.exists():
    raise FileNotFoundError(
        BASE_MODEL_PATH
    )


current_x_hash = sha256_file(
    X_PATH
)

current_target_hash = sha256_file(
    TARGET_PATH
)

current_base_hash = sha256_file(
    BASE_MODEL_PATH
)


print("=" * 100)
print("FROZEN INPUT VERIFICATION")
print("=" * 100)

print("V9 ECG hash:")
print(current_x_hash)

print("\nTarget hash:")
print(current_target_hash)

print("\nBase-model hash:")
print(current_base_hash)


if (
    current_x_hash
    !=
    step1_manifest["frozen_ecg_sha256"]
):
    raise RuntimeError(
        "Frozen V9 ECG input hash no longer matches Step 1."
    )


if (
    current_target_hash
    !=
    step1_manifest["clean_gaze_targets_sha256"]
):
    raise RuntimeError(
        "Gaze-target hash no longer matches Step 1."
    )


if (
    current_base_hash
    !=
    step1_manifest["base_model_sha256"]
):
    raise RuntimeError(
        "Base-model hash no longer matches Step 1."
    )


print(
    "\n✓ All frozen hashes match Step 1."
)


# ======================================================================================
# 6. LOAD DATA
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


saved_ecg_order = [
    str(x)
    for x in target_data[
        "ecg_order"
    ]
]


saved_lead_order = [
    str(x)
    for x in target_data[
        "lead_order"
    ]
]


if saved_ecg_order != ECG_ORDER:
    raise RuntimeError(
        "Saved ECG order mismatch."
    )


if saved_lead_order != LEAD_ORDER:
    raise RuntimeError(
        "Saved lead order mismatch."
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


for population in POPULATIONS:

    if Y[population].shape != (
        10,
        12
    ):
        raise RuntimeError(
            f"Unexpected target shape for {population}: "
            f"{Y[population].shape}"
        )


print(
    "\n✓ Frozen arrays loaded."
)


# ======================================================================================
# 7. FROZEN ATTENTION READOUT
# ======================================================================================

def attention_distribution_from_add3(
    add3_tensor
):

    # add_3:
    # (batch, 12, 55, 128)

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


    denominator = tf.reduce_sum(
        rms,
        axis=1,
        keepdims=True
    )


    return (
        rms
        /
        (
            denominator
            +
            RMS_EPSILON
        )
    )


# ======================================================================================
# 8. METRICS
# ======================================================================================

def cosine_metric(p, q):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )


    denominator = (
        np.linalg.norm(p)
        *
        np.linalg.norm(q)
    )


    if denominator <= 0:
        return np.nan


    return float(
        np.dot(p, q)
        /
        denominator
    )


def mse_metric(p, q):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )


    return float(
        np.mean(
            np.square(
                p - q
            )
        )
    )


def jsd_metric(
    p,
    q,
    eps=1e-12
):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )


    p = np.clip(
        p,
        eps,
        None
    )

    q = np.clip(
        q,
        eps,
        None
    )


    p /= p.sum()
    q /= q.sum()


    m = 0.5 * (
        p + q
    )


    kl_pm = np.sum(
        p
        *
        np.log(
            p / m
        )
    )


    kl_qm = np.sum(
        q
        *
        np.log(
            q / m
        )
    )


    return float(
        0.5
        *
        (
            kl_pm
            +
            kl_qm
        )
    )


# ======================================================================================
# 9. LOAD A FRESH MODEL FOR EACH ARM/FOLD
# ======================================================================================

def load_fresh_representation_model():

    tf.keras.backend.clear_session()

    tf.random.set_seed(
        SEED
    )


    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )


    # Freeze entire network.
    for layer in model.layers:
        layer.trainable = False


    # Only this layer may change.
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


    if len(train_vars) == 0:
        raise RuntimeError(
            "conv2d_10 has no trainable variables."
        )


    return (
        model,
        rep_model,
        conv10,
        train_vars
    )


# ======================================================================================
# 10. PREDICT LEAD ATTENTION
# ======================================================================================

def predict_attention(
    rep_model,
    x
):

    activations = rep_model(
        x,
        training=False
    )


    attention = (
        attention_distribution_from_add3(
            activations
        )
        .numpy()
    )


    if not np.all(
        np.isfinite(
            attention
        )
    ):
        raise RuntimeError(
            "Non-finite attention values."
        )


    if not np.allclose(
        attention.sum(
            axis=1
        ),
        1.0,
        atol=1e-5
    ):
        raise RuntimeError(
            "Attention distributions do not sum to 1."
        )


    if np.any(
        attention < 0
    ):
        raise RuntimeError(
            "Negative attention detected."
        )


    return attention


# ======================================================================================
# 11. SNAPSHOT / COMPARE WEIGHTS
# ======================================================================================

def snapshot_layer_weights(
    model
):

    snapshot = {}


    for layer in model.layers:

        snapshot[
            layer.name
        ] = [
            np.array(
                weight,
                copy=True
            )
            for weight in layer.get_weights()
        ]


    return snapshot


def compare_layer_weights(
    before,
    model
):

    rows = []


    for layer in model.layers:

        old_weights = before[
            layer.name
        ]

        new_weights = layer.get_weights()


        if len(old_weights) != len(new_weights):
            raise RuntimeError(
                f"Weight-count mismatch: {layer.name}"
            )


        max_difference = 0.0


        for old, new in zip(
            old_weights,
            new_weights
        ):

            if old.shape != new.shape:
                raise RuntimeError(
                    f"Weight-shape mismatch: {layer.name}"
                )


            if old.size > 0:

                difference = float(
                    np.max(
                        np.abs(
                            old - new
                        )
                    )
                )


                max_difference = max(
                    max_difference,
                    difference
                )


        rows.append(
            {
                "layer":
                    layer.name,

                "n_weight_tensors":
                    len(
                        new_weights
                    ),

                "max_abs_change":
                    max_difference,
            }
        )


    return pd.DataFrame(
        rows
    )


# ======================================================================================
# 12. FREEZE CROSS-VALIDATION SPLITS
# ======================================================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)


folds = []


for fold_number, split in enumerate(
    kf.split(
        np.arange(
            len(
                ECG_ORDER
            )
        )
    ),
    start=1
):

    train_indices, test_indices = split


    folds.append(
        {
            "fold":
                fold_number,

            "train_indices":
                np.asarray(
                    train_indices,
                    dtype=int
                ),

            "test_indices":
                np.asarray(
                    test_indices,
                    dtype=int
                ),
        }
    )


print(
    "\n" +
    "=" * 100
)

print(
    "FROZEN 5-FOLD SPLIT"
)

print(
    "=" * 100
)


fold_manifest_rows = []


for fold in folds:

    train_names = [
        ECG_ORDER[i]
        for i in fold[
            "train_indices"
        ]
    ]


    test_names = [
        ECG_ORDER[i]
        for i in fold[
            "test_indices"
        ]
    ]


    print(
        f"\nFold {fold['fold']}"
    )

    print(
        "  held out:",
        test_names
    )

    print(
        "  training:",
        train_names
    )


    fold_manifest_rows.append(
        {
            "fold":
                fold[
                    "fold"
                ],

            "held_out_ECGs":
                " | ".join(
                    test_names
                ),

            "training_ECGs":
                " | ".join(
                    train_names
                ),
        }
    )


pd.DataFrame(
    fold_manifest_rows
).to_csv(
    STEP2_DIR
    /
    "fold_manifest.csv",
    index=False
)


# ======================================================================================
# 13. TRAIN ONE POPULATION × FOLD
# ======================================================================================

def train_one_arm(
    population,
    fold_number,
    train_indices,
    test_indices,
    X_train_augmented,
    epoch_permutations
):

    (
        model,
        rep_model,
        conv10,
        train_vars
    ) = load_fresh_representation_model()


    before_weights = snapshot_layer_weights(
        model
    )


    print(
        "\nTrainable conv2d_10 variables:"
    )

    for variable in train_vars:

        print(
            "   ",
            variable.name,
            tuple(
                variable.shape
            )
        )


    # ------------------------------------------------------------------
    # Baseline held-out attention
    # ------------------------------------------------------------------

    X_test = X[
        test_indices
    ]


    Y_test = Y[
        population
    ][
        test_indices
    ]


    baseline_attention = predict_attention(
        rep_model,
        X_test
    )


    # ------------------------------------------------------------------
    # Training targets
    # ------------------------------------------------------------------

    Y_train_augmented = np.repeat(
        Y[
            population
        ][
            train_indices
        ],
        AUGMENTATIONS_PER_ECG,
        axis=0
    ).astype(
        np.float32
    )


    if (
        len(
            X_train_augmented
        )
        !=
        len(
            Y_train_augmented
        )
    ):
        raise RuntimeError(
            "Augmented X/Y size mismatch."
        )


    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )


    epoch_losses = []


    # ------------------------------------------------------------------
    # Deterministic custom loop
    # ------------------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        permutation = epoch_permutations[
            epoch
        ]


        batch_losses = []


        for start in range(
            0,
            len(
                permutation
            ),
            BATCH_SIZE
        ):

            batch_idx = permutation[
                start:
                start + BATCH_SIZE
            ]


            xb = tf.convert_to_tensor(
                X_train_augmented[
                    batch_idx
                ],
                dtype=tf.float32
            )


            yb = tf.convert_to_tensor(
                Y_train_augmented[
                    batch_idx
                ],
                dtype=tf.float32
            )


            with tf.GradientTape() as tape:

                add3 = rep_model(
                    xb,
                    training=False
                )


                predicted_attention = (
                    attention_distribution_from_add3(
                        add3
                    )
                )


                loss = tf.reduce_mean(
                    tf.square(
                        predicted_attention
                        -
                        yb
                    )
                )


            gradients = tape.gradient(
                loss,
                train_vars
            )


            if any(
                gradient is None
                for gradient in gradients
            ):
                raise RuntimeError(
                    "At least one conv2d_10 gradient is None."
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


            batch_losses.append(
                float(
                    loss.numpy()
                )
            )


        mean_epoch_loss = float(
            np.mean(
                batch_losses
            )
        )


        epoch_losses.append(
            mean_epoch_loss
        )


        print(
            f"    epoch {epoch + 1:02d}/{EPOCHS}: "
            f"loss={mean_epoch_loss:.8f}"
        )


    # ------------------------------------------------------------------
    # Held-out attention after training
    # ------------------------------------------------------------------

    trained_attention = predict_attention(
        rep_model,
        X_test
    )


    # ------------------------------------------------------------------
    # Confirm only conv2d_10 changed
    # ------------------------------------------------------------------

    weight_df = compare_layer_weights(
        before_weights,
        model
    )


    changed_layers = set(
        weight_df.loc[
            weight_df[
                "max_abs_change"
            ]
            >
            1e-12,
            "layer"
        ]
    )


    if changed_layers != {
        "conv2d_10"
    }:
        raise RuntimeError(
            "\nUnexpected changed layers.\n"
            "Expected: {'conv2d_10'}\n"
            f"Actual:   {changed_layers}"
        )


    print(
        "\n    ✓ Weight audit: ONLY conv2d_10 changed."
    )


    # ------------------------------------------------------------------
    # Save weights
    # ------------------------------------------------------------------

    weight_path = (
        MODEL_DIR
        /
        f"{population}_fold{fold_number}.weights.h5"
    )


    model.save_weights(
        weight_path
    )


    return {
        "baseline_attention":
            baseline_attention,

        "trained_attention":
            trained_attention,

        "Y_test":
            Y_test,

        "epoch_losses":
            epoch_losses,

        "weight_df":
            weight_df,

        "weight_path":
            weight_path,
    }


# ======================================================================================
# 14. RUN FIVE-FOLD EXPERIMENT
# ======================================================================================

metric_rows = []
prediction_rows = []
history_rows = []
weight_audit_rows = []


for fold in folds:

    fold_number = fold[
        "fold"
    ]


    train_indices = fold[
        "train_indices"
    ]


    test_indices = fold[
        "test_indices"
    ]


    print(
        "\n\n" +
        "#" * 100
    )

    print(
        f"FOLD {fold_number}"
    )

    print(
        "#" * 100
    )


    print(
        "Held-out ECGs:",
        [
            ECG_ORDER[i]
            for i in test_indices
        ]
    )


    # ------------------------------------------------------------------
    # Generate augmented X exactly once per fold
    # ------------------------------------------------------------------

    augmentation_rng = np.random.default_rng(
        SEED
        +
        10000
        *
        fold_number
    )


    X_train_base = np.repeat(
        X[
            train_indices
        ],
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


    expected_training_samples = (
        len(
            train_indices
        )
        *
        AUGMENTATIONS_PER_ECG
    )


    if (
        X_train_augmented.shape[0]
        !=
        expected_training_samples
    ):
        raise RuntimeError(
            "Unexpected augmented training-set size."
        )


    # ------------------------------------------------------------------
    # Same epoch permutations for both populations
    # ------------------------------------------------------------------

    permutation_rng = np.random.default_rng(
        SEED
        +
        20000
        *
        fold_number
    )


    epoch_permutations = [
        permutation_rng.permutation(
            expected_training_samples
        )
        for _ in range(
            EPOCHS
        )
    ]


    # ------------------------------------------------------------------
    # Both populations
    # ------------------------------------------------------------------

    for population in POPULATIONS:

        print(
            "\n" +
            "=" * 100
        )

        print(
            f"FOLD {fold_number} — {population.upper()}"
        )

        print(
            "=" * 100
        )


        result = train_one_arm(
            population=population,
            fold_number=fold_number,
            train_indices=train_indices,
            test_indices=test_indices,
            X_train_augmented=X_train_augmented,
            epoch_permutations=epoch_permutations,
        )


        baseline_attention = result[
            "baseline_attention"
        ]


        trained_attention = result[
            "trained_attention"
        ]


        Y_test = result[
            "Y_test"
        ]


        # --------------------------------------------------------------
        # Training history
        # --------------------------------------------------------------

        for epoch_idx, loss_value in enumerate(
            result[
                "epoch_losses"
            ],
            start=1
        ):

            history_rows.append(
                {
                    "population":
                        population,

                    "fold":
                        fold_number,

                    "epoch":
                        epoch_idx,

                    "loss":
                        loss_value,
                }
            )


        # --------------------------------------------------------------
        # Weight audit
        # --------------------------------------------------------------

        temporary_weight_df = result[
            "weight_df"
        ].copy()


        temporary_weight_df[
            "population"
        ] = population


        temporary_weight_df[
            "fold"
        ] = fold_number


        weight_audit_rows.extend(
            temporary_weight_df.to_dict(
                "records"
            )
        )


        # --------------------------------------------------------------
        # Held-out ECGs
        # --------------------------------------------------------------

        for local_idx, ecg_index in enumerate(
            test_indices
        ):

            ecg = ECG_ORDER[
                ecg_index
            ]


            target = Y_test[
                local_idx
            ]


            base_pred = baseline_attention[
                local_idx
            ]


            trained_pred = trained_attention[
                local_idx
            ]


            base_cos = cosine_metric(
                base_pred,
                target
            )


            trained_cos = cosine_metric(
                trained_pred,
                target
            )


            base_mse = mse_metric(
                base_pred,
                target
            )


            trained_mse = mse_metric(
                trained_pred,
                target
            )


            base_jsd = jsd_metric(
                base_pred,
                target
            )


            trained_jsd = jsd_metric(
                trained_pred,
                target
            )


            metric_rows.append(
                {
                    "population":
                        population,

                    "fold":
                        fold_number,

                    "ECG":
                        ecg,

                    "baseline_cosine":
                        base_cos,

                    "trained_cosine":
                        trained_cos,

                    "delta_cosine":
                        trained_cos
                        -
                        base_cos,

                    "baseline_MSE":
                        base_mse,

                    "trained_MSE":
                        trained_mse,

                    "delta_MSE":
                        trained_mse
                        -
                        base_mse,

                    "baseline_JSD":
                        base_jsd,

                    "trained_JSD":
                        trained_jsd,

                    "delta_JSD":
                        trained_jsd
                        -
                        base_jsd,
                }
            )


            for lead_idx, lead in enumerate(
                LEAD_ORDER
            ):

                prediction_rows.append(
                    {
                        "population":
                            population,

                        "fold":
                            fold_number,

                        "ECG":
                            ecg,

                        "lead":
                            lead,

                        "target":
                            float(
                                target[
                                    lead_idx
                                ]
                            ),

                        "baseline":
                            float(
                                base_pred[
                                    lead_idx
                                ]
                            ),

                        "trained":
                            float(
                                trained_pred[
                                    lead_idx
                                ]
                            ),
                    }
                )


        gc.collect()


# ======================================================================================
# 15. CREATE / SAVE RAW OUTPUT TABLES
# ======================================================================================

metrics_df = pd.DataFrame(
    metric_rows
)


predictions_df = pd.DataFrame(
    prediction_rows
)


history_df = pd.DataFrame(
    history_rows
)


weight_audit_df = pd.DataFrame(
    weight_audit_rows
)


metrics_df.to_csv(
    STEP2_DIR
    /
    "heldout_metrics.csv",
    index=False
)


predictions_df.to_csv(
    STEP2_DIR
    /
    "heldout_attention_distributions.csv",
    index=False
)


history_df.to_csv(
    STEP2_DIR
    /
    "training_history.csv",
    index=False
)


weight_audit_df.to_csv(
    STEP2_DIR
    /
    "weight_change_audit.csv",
    index=False
)


# ======================================================================================
# 16. VERIFY HELD-OUT COVERAGE
# ======================================================================================

for population in POPULATIONS:

    temp = metrics_df[
        metrics_df[
            "population"
        ]
        ==
        population
    ]


    counts = temp[
        "ECG"
    ].value_counts()


    if len(counts) != 10:
        raise RuntimeError(
            f"{population}: not all 10 ECGs were evaluated."
        )


    if not (
        counts == 1
    ).all():
        raise RuntimeError(
            f"{population}: ECG holdout count error."
        )


print(
    "\n✓ Every ECG was held out exactly once for each population."
)


# ======================================================================================
# 17. WILCOXON
# ======================================================================================

def safe_wilcoxon(
    baseline,
    trained
):

    baseline = np.asarray(
        baseline,
        dtype=np.float64
    )


    trained = np.asarray(
        trained,
        dtype=np.float64
    )


    differences = (
        trained
        -
        baseline
    )


    if np.allclose(
        differences,
        0
    ):
        return (
            0.0,
            1.0
        )


    result = wilcoxon(
        trained,
        baseline,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )


    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        )
    )


# ======================================================================================
# 18. SUMMARY STATISTICS
# ======================================================================================

summary_rows = []


for population in POPULATIONS:

    temp = metrics_df[
        metrics_df[
            "population"
        ]
        ==
        population
    ].copy()


    # ------------------------------------------------------------------
    # COSINE — higher is better
    # ------------------------------------------------------------------

    statistic, p_value = safe_wilcoxon(
        temp[
            "baseline_cosine"
        ],
        temp[
            "trained_cosine"
        ]
    )


    summary_rows.append(
        {
            "population":
                population,

            "metric":
                "cosine",

            "direction_better":
                "higher",

            "baseline_mean":
                temp[
                    "baseline_cosine"
                ].mean(),

            "trained_mean":
                temp[
                    "trained_cosine"
                ].mean(),

            "mean_delta":
                temp[
                    "delta_cosine"
                ].mean(),

            "median_delta":
                temp[
                    "delta_cosine"
                ].median(),

            "n_improved":
                int(
                    (
                        temp[
                            "delta_cosine"
                        ]
                        >
                        0
                    ).sum()
                ),

            "wilcoxon_statistic":
                statistic,

            "wilcoxon_p":
                p_value,
        }
    )


    # ------------------------------------------------------------------
    # MSE — lower is better
    # ------------------------------------------------------------------

    statistic, p_value = safe_wilcoxon(
        temp[
            "baseline_MSE"
        ],
        temp[
            "trained_MSE"
        ]
    )


    summary_rows.append(
        {
            "population":
                population,

            "metric":
                "MSE",

            "direction_better":
                "lower",

            "baseline_mean":
                temp[
                    "baseline_MSE"
                ].mean(),

            "trained_mean":
                temp[
                    "trained_MSE"
                ].mean(),

            "mean_delta":
                temp[
                    "delta_MSE"
                ].mean(),

            "median_delta":
                temp[
                    "delta_MSE"
                ].median(),

            "n_improved":
                int(
                    (
                        temp[
                            "delta_MSE"
                        ]
                        <
                        0
                    ).sum()
                ),

            "wilcoxon_statistic":
                statistic,

            "wilcoxon_p":
                p_value,
        }
    )


    # ------------------------------------------------------------------
    # JSD — lower is better
    # ------------------------------------------------------------------

    statistic, p_value = safe_wilcoxon(
        temp[
            "baseline_JSD"
        ],
        temp[
            "trained_JSD"
        ]
    )


    summary_rows.append(
        {
            "population":
                population,

            "metric":
                "JSD",

            "direction_better":
                "lower",

            "baseline_mean":
                temp[
                    "baseline_JSD"
                ].mean(),

            "trained_mean":
                temp[
                    "trained_JSD"
                ].mean(),

            "mean_delta":
                temp[
                    "delta_JSD"
                ].mean(),

            "median_delta":
                temp[
                    "delta_JSD"
                ].median(),

            "n_improved":
                int(
                    (
                        temp[
                            "delta_JSD"
                        ]
                        <
                        0
                    ).sum()
                ),

            "wilcoxon_statistic":
                statistic,

            "wilcoxon_p":
                p_value,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    STEP2_DIR
    /
    "alignment_summary.csv",
    index=False
)


# ======================================================================================
# 19. PRINT HELD-OUT RESULTS
# ======================================================================================

print(
    "\n\n" +
    "=" * 100
)

print(
    "HELD-OUT ECG RESULTS"
)

print(
    "=" * 100
)


print(
    metrics_df[
        [
            "population",
            "fold",
            "ECG",
            "baseline_cosine",
            "trained_cosine",
            "delta_cosine",
            "baseline_MSE",
            "trained_MSE",
            "delta_MSE",
            "baseline_JSD",
            "trained_JSD",
            "delta_JSD",
        ]
    ]
    .round(
        7
    )
    .to_string(
        index=False
    )
)


# ======================================================================================
# 20. PRINT PRIMARY SUMMARY
# ======================================================================================

print(
    "\n\n" +
    "=" * 100
)

print(
    "DEFINITIVE HELD-OUT GAZE-ALIGNMENT SUMMARY"
)

print(
    "=" * 100
)


print(
    summary_df
    .round(
        7
    )
    .to_string(
        index=False
    )
)


# ======================================================================================
# 21. FINAL WEIGHT AUDIT
# ======================================================================================

changed_summary = (
    weight_audit_df[
        weight_audit_df[
            "max_abs_change"
        ]
        >
        1e-12
    ][
        [
            "population",
            "fold",
            "layer",
            "max_abs_change",
        ]
    ]
)


print(
    "\n\n" +
    "=" * 100
)

print(
    "LAYERS THAT CHANGED"
)

print(
    "=" * 100
)


print(
    changed_summary.to_string(
        index=False
    )
)


if set(
    changed_summary[
        "layer"
    ].unique()
) != {
    "conv2d_10"
}:
    raise RuntimeError(
        "Final weight audit failed."
    )


if len(
    changed_summary
) != (
    N_SPLITS
    *
    len(
        POPULATIONS
    )
):
    raise RuntimeError(
        "Expected exactly one changed layer per population × fold."
    )


print(
    "\n✓ Every fold/arm changed ONLY conv2d_10."
)


# ======================================================================================
# 22. SAVE STEP-2 MANIFEST
# ======================================================================================

step2_manifest = {

    "experiment":
        "Definitive clean gaze-aligned ECG experiment",

    "stage":
        "Step 2: five-fold held-out ECG gaze alignment",

    "seed":
        SEED,

    "n_folds":
        N_SPLITS,

    "held_out_ECGs_per_fold":
        2,

    "training_ECGs_per_fold":
        8,

    "attention_readout":
        (
            "RMS activation energy from add_3: "
            "sqrt(mean(add_3^2)) across temporal and channel axes, "
            "then normalised across 12 leads."
        ),

    "attention_layer":
        "add_3",

    "trainable_layer":
        "conv2d_10",

    "batchnorm_mode":
        "frozen",

    "forward_training_mode":
        False,

    "loss":
        "MSE between 12-lead RMS attention distribution and gaze target",

    "optimizer":
        "Adam",

    "learning_rate":
        LEARNING_RATE,

    "epochs":
        EPOCHS,

    "augmentations_per_training_ECG":
        AUGMENTATIONS_PER_ECG,

    "gaussian_noise_sigma":
        NOISE_SD,

    "batch_size":
        BATCH_SIZE,

    "same_folds_between_populations":
        True,

    "same_noisy_training_examples_between_populations":
        True,

    "same_batch_order_between_populations":
        True,

    "populations": {
        "consultant":
            9,

        "all_clinicians":
            20,
    },

    "frozen_V9_sha256":
        current_x_hash,

    "frozen_gaze_target_sha256":
        current_target_hash,

    "base_model_sha256":
        current_base_hash,

    "statistical_unit":
        "held-out ECG",

    "n_statistical_units_per_population":
        10,

    "statistical_test":
        "two-sided paired Wilcoxon signed-rank test",

    "metrics": [
        "cosine similarity",
        "mean squared error",
        "Jensen-Shannon divergence",
    ],
}


with open(
    STEP2_DIR
    /
    "step2_manifest.json",
    "w"
) as f:

    json.dump(
        step2_manifest,
        f,
        indent=2
    )


print(
    "\n\n" +
    "=" * 100
)

print(
    "STEP 2 COMPLETE"
)

print(
    "=" * 100
)


print(
    """
Please paste back:

1. FROZEN 5-FOLD SPLIT

2. DEFINITIVE HELD-OUT GAZE-ALIGNMENT SUMMARY

3. LAYERS THAT CHANGED

You do not need to paste all epoch losses unless there is an error.
"""
)
