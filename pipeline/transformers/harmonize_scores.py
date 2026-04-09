"""Normalize functional scores and compute consensus labels.

Each assay source has different score directionality:
- Pillar, LabelSEQ: higher = more functional (benign)
- FisSEQ: higher = more dysfunctional (pathogenic) → inverted

This transformer min-max normalizes each source to [0, 1], inverts where
needed, and computes a consensus mean across available sources.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.config import (
    CONFLICT_SCORE_THRESHOLD,
    CONSENSUS_FUNC_THRESHOLD,
    CONSENSUS_LOF_THRESHOLD,
    MIN_FISSEQ_CELLS,
    MIN_LABELSEQ_BARCODES,
    SCORE_ORIENTATION,
)

logger = logging.getLogger(__name__)

_SCORE_COLS = {
    "pillar": "functional_score_pillar",
    "labelseq": "functional_score_labelseq",
    "fisseq": "functional_score_fisseq",
    "vampseq": "functional_score_vampseq",
}


def harmonize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize per-source scores and compute consensus functional score/label."""

    # Min-max normalize each source and apply orientation correction
    norm_cols: list[str] = []
    for source, col in _SCORE_COLS.items():
        if col not in df.columns or df[col].isna().all():
            continue

        vals = pd.to_numeric(df[col], errors="coerce")
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            normed = pd.Series(0.5, index=df.index)
        else:
            normed = (vals - vmin) / (vmax - vmin)

        # Invert if higher = dysfunctional
        orient = SCORE_ORIENTATION.get(source)
        if orient is False:
            normed = 1.0 - normed

        norm_name = f"_norm_{source}"
        df[norm_name] = normed
        norm_cols.append(norm_name)
        logger.info(
            "  Normalized %s: %d non-null values (orientation=%s)",
            col, vals.notna().sum(), orient,
        )

    if not norm_cols:
        logger.warning("Harmonize scores: no functional score columns found")
        df["consensus_functional_score"] = pd.NA
        df["consensus_functional_label"] = pd.NA
        return df

    # Consensus score = mean of available normalized scores
    df["consensus_functional_score"] = df[norm_cols].mean(axis=1, skipna=True)

    # Consensus labels
    df["consensus_functional_label"] = pd.NA
    df.loc[
        df["consensus_functional_score"] >= CONSENSUS_FUNC_THRESHOLD,
        "consensus_functional_label",
    ] = "functional"
    df.loc[
        df["consensus_functional_score"] <= CONSENSUS_LOF_THRESHOLD,
        "consensus_functional_label",
    ] = "loss_of_function"
    df.loc[
        (df["consensus_functional_score"] > CONSENSUS_LOF_THRESHOLD)
        & (df["consensus_functional_score"] < CONSENSUS_FUNC_THRESHOLD),
        "consensus_functional_label",
    ] = "intermediate"

    # QC: conflict flag (multi-source disagreement)
    if len(norm_cols) >= 2:
        score_matrix = df[norm_cols]
        score_range = score_matrix.max(axis=1) - score_matrix.min(axis=1)
        n_sources = score_matrix.notna().sum(axis=1)
        df["conflict_flag"] = (n_sources >= 2) & (score_range > CONFLICT_SCORE_THRESHOLD)
    else:
        df["conflict_flag"] = False

    # QC: low coverage flag
    df["low_coverage_flag"] = False
    if "labelseq_n_barcodes" in df.columns:
        df.loc[
            df["labelseq_n_barcodes"].notna()
            & (df["labelseq_n_barcodes"] < MIN_LABELSEQ_BARCODES),
            "low_coverage_flag",
        ] = True
    for rep_col in ["fisseq_num_cells_rep1", "fisseq_num_cells_rep2"]:
        if rep_col in df.columns:
            total_cells = df[[c for c in df.columns if "num_cells" in c]].sum(axis=1)
            df.loc[
                total_cells.notna() & (total_cells < MIN_FISSEQ_CELLS),
                "low_coverage_flag",
            ] = True
            break

    # Clean up temp columns
    df = df.drop(columns=norm_cols, errors="ignore")

    # Summary
    scores = df["consensus_functional_score"].dropna()
    logger.info(
        "Harmonize scores: %d rows scored, mean=%.3f, "
        "functional=%d, intermediate=%d, LOF=%d",
        len(scores),
        scores.mean() if len(scores) else 0,
        (df["consensus_functional_label"] == "functional").sum(),
        (df["consensus_functional_label"] == "intermediate").sum(),
        (df["consensus_functional_label"] == "loss_of_function").sum(),
    )
    return df
