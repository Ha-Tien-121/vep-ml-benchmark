"""Add human-readable protein substitution columns.

Derives two columns from the existing aa_ref, aa_pos, aa_alt fields:
  - protein_substitution      1-letter notation  e.g. "V600E"
  - protein_substitution_long 3-letter notation  e.g. "Val600Glu"
"""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.config import AA1TO3

logger = logging.getLogger(__name__)


def add_protein_substitution_col(df: pd.DataFrame) -> pd.DataFrame:
    """Populate protein_substitution and protein_substitution_long columns."""
    has_fields = (
        "aa_ref" in df.columns
        and "aa_pos" in df.columns
        and "aa_alt" in df.columns
    )
    if not has_fields:
        logger.warning(
            "add_protein_substitution_col: missing aa_ref/aa_pos/aa_alt — skipping"
        )
        df["protein_substitution"] = pd.NA
        df["protein_substitution_long"] = pd.NA
        return df

    valid = (
        df["aa_ref"].notna()
        & df["aa_pos"].notna()
        & df["aa_alt"].notna()
    )

    aa_ref = df["aa_ref"].astype(str).str.strip().str.upper()
    aa_alt = df["aa_alt"].astype(str).str.strip().str.upper()
    aa_pos = df["aa_pos"]

    # 1-letter: "V600E"
    df["protein_substitution"] = pd.NA
    df.loc[valid, "protein_substitution"] = (
        aa_ref[valid] + aa_pos[valid].astype(str) + aa_alt[valid]
    )

    # 3-letter: "Val600Glu"
    ref3 = aa_ref.map(lambda x: AA1TO3.get(x, x))
    alt3 = aa_alt.map(lambda x: AA1TO3.get(x, x))

    df["protein_substitution_long"] = pd.NA
    df.loc[valid, "protein_substitution_long"] = (
        ref3[valid] + aa_pos[valid].astype(str) + alt3[valid]
    )

    n_filled = valid.sum()
    logger.info(
        "protein_substitution: %d / %d rows filled (e.g. %s / %s)",
        n_filled,
        len(df),
        df["protein_substitution"].dropna().iloc[0] if n_filled else "n/a",
        df["protein_substitution_long"].dropna().iloc[0] if n_filled else "n/a",
    )
    return df
