# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 4
# Stage: freeze full-data weight hashes
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# FREEZE FINAL MODEL HASHES
# No training.
# ======================================================================================

import hashlib
import json
from pathlib import Path


FINAL_DIR = Path(
    "definitive_clean_gaze/final_models"
)

CONSULTANT_PATH = (
    FINAL_DIR
    /
    "clean_consultant.weights.h5"
)

ALL_PATH = (
    FINAL_DIR
    /
    "clean_all_clinicians.weights.h5"
)


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


consultant_hash = sha256_file(
    CONSULTANT_PATH
)

all_hash = sha256_file(
    ALL_PATH
)


print("=" * 100)
print("FROZEN FINAL GAZE MODELS")
print("=" * 100)

print("\nConsultant:")
print(CONSULTANT_PATH)
print(consultant_hash)

print("\nAll clinicians:")
print(ALL_PATH)
print(all_hash)


# Compare with the manifest generated during Step 3.
manifest_path = (
    FINAL_DIR
    /
    "final_model_manifest.json"
)

with open(
    manifest_path,
    "r"
) as f:

    manifest = json.load(f)


assert (
    consultant_hash
    ==
    manifest["consultant_weights_sha256"]
)

assert (
    all_hash
    ==
    manifest["all_clinician_weights_sha256"]
)


print(
    "\n✓ Both hashes match the Step-3 manifest."
)

print(
    "✓ Final gaze models are now frozen."
)
