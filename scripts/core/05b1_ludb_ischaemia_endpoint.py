# THIS IS AN UNMODIFIED EXTRACTION OF A FINAL RESEARCH NOTEBOOK CODE CELL.
# Original source: latest gaze training 31.08.ipynb ; code cell 11
# Stage: resolve LUDB ischaemia endpoint
# Execute from the project root with the required upstream files present.
# Do not alter experiment definitions when reproducing published results.

# ======================================================================================
# STEP 5B.1 — RESOLVE LUDB ISCHAEMIA ENDPOINT
#
# READ ONLY.
# NO MODEL INFERENCE.
# NO TRAINING.
#
# Goal:
#   Identify the 12 records currently classified as indeterminate and
#   inspect the exact historical code used to produce N=192, positive=43.
# ======================================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================================
# 1. SHOW ALL CURRENTLY INDETERMINATE RECORDS
# ======================================================================================

indeterminate = ludb[
    ludb["target_Ischaemia_related"].isna()
][
    [
        "ID",
        "Ischemia",
    ]
].copy()


print("=" * 100)
print("CURRENTLY INDETERMINATE ISCHAEMIA RECORDS")
print("=" * 100)

print(
    "N =",
    len(indeterminate)
)

print(
    indeterminate.to_string(
        index=False
    )
)


# ======================================================================================
# 2. BREAK EACH ISCHAEMIA FIELD INTO LINES
# ======================================================================================

print("\n" + "=" * 100)
print("INDETERMINATE RECORDS — LINE-BY-LINE")
print("=" * 100)


for _, row in indeterminate.iterrows():

    print(
        f"\nID {row['ID']}"
    )

    print(
        "-" * 60
    )

    lines = split_lines(
        row["Ischemia"]
    )

    for line in lines:

        print(
            repr(
                line
            )
        )


# ======================================================================================
# 3. READ HISTORICAL NOTEBOOK AND SEARCH FOR ISCHAEMIA TASK DEFINITION
# ======================================================================================

HISTORICAL_NOTEBOOK = Path(
    "Re-analysis.ipynb"
)


if not HISTORICAL_NOTEBOOK.exists():

    raise FileNotFoundError(
        HISTORICAL_NOTEBOOK
    )


nb = json.loads(
    HISTORICAL_NOTEBOOK.read_text(
        encoding="utf-8"
    )
)


search_terms = [
    "Ischaemia_related",
    "ischemia_related",
    "ischaemia",
    "ischemia",
    "scar",
    "stemi",
    "supp.nstemi",
]


print("\n" + "=" * 100)
print("HISTORICAL ISCHAEMIA-RELATED CODE CELLS")
print("=" * 100)


matched_cells = 0


for cell_index, cell in enumerate(
    nb.get(
        "cells",
        []
    )
):

    if cell.get(
        "cell_type"
    ) != "code":

        continue


    source = "".join(
        cell.get(
            "source",
            []
        )
    )


    source_lower = source.lower()


    if any(
        term.lower() in source_lower
        for term in search_terms
    ):

        matched_cells += 1

        print(
            "\n" +
            "#" * 100
        )

        print(
            f"CODE CELL {cell_index}"
        )

        print(
            "#" * 100
        )

        print(
            source
        )


print(
    "\nMatched code cells:",
    matched_cells
)


# ======================================================================================
# 4. ALSO SEARCH "Test on LUDB all classes.ipynb"
# ======================================================================================

SECOND_NOTEBOOK = Path(
    "Test on LUDB all classes.ipynb"
)


if SECOND_NOTEBOOK.exists():

    nb2 = json.loads(
        SECOND_NOTEBOOK.read_text(
            encoding="utf-8"
        )
    )


    print("\n" + "=" * 100)
    print("SECOND HISTORICAL NOTEBOOK — ISCHAEMIA-RELATED CODE")
    print("=" * 100)


    matched_cells_2 = 0


    for cell_index, cell in enumerate(
        nb2.get(
            "cells",
            []
        )
    ):

        if cell.get(
            "cell_type"
        ) != "code":

            continue


        source = "".join(
            cell.get(
                "source",
                []
            )
        )


        source_lower = source.lower()


        if any(
            term.lower() in source_lower
            for term in search_terms
        ):

            matched_cells_2 += 1

            print(
                "\n" +
                "#" * 100
            )

            print(
                f"CODE CELL {cell_index}"
            )

            print(
                "#" * 100
            )

            print(
                source
            )


    print(
        "\nMatched code cells:",
        matched_cells_2
    )


# ======================================================================================
# 5. SIMPLE CONTENT AUDIT OF THE 12 RECORDS
# ======================================================================================

content_rows = []


for _, row in indeterminate.iterrows():

    text = clean_text(
        row["Ischemia"]
    )


    content_rows.append(
        {
            "ID":
                int(
                    row["ID"]
                ),

            "contains_undefined":
                "undefined ischemia/scar/supp.nstemi"
                in text,

            "contains_stemi_word":
                "stemi"
                in text,

            "contains_supp_nstemi":
                "supp.nstemi"
                in text,

            "n_lines":
                len(
                    split_lines(
                        row["Ischemia"]
                    )
                ),

            "raw_Ischemia":
                row["Ischemia"],
        }
    )


content_df = pd.DataFrame(
    content_rows
)


print("\n" + "=" * 100)
print("INDETERMINATE CONTENT SUMMARY")
print("=" * 100)

print(
    content_df.to_string(
        index=False
    )
)


print(
    """
NO MODEL INFERENCE OR TRAINING PERFORMED.

Please paste back:

1. CURRENTLY INDETERMINATE ISCHAEMIA RECORDS
2. HISTORICAL ISCHAEMIA-RELATED CODE CELLS
3. INDETERMINATE CONTENT SUMMARY
"""
)
