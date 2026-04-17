"""Merge all source dataframes into one unified benchmark dataframe.

Pillar is the left/backbone.  Other sources are outer-joined first on
``genomic_coord`` (when available) and then on a protein-level compound
key ``(gene, aa_pos, aa_ref, aa_alt)`` as a fallback.  Variants unique
to any assay are preserved via outer join.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_SHARED_IDENTITY = {
    "gene", "aa_pos", "aa_ref", "aa_alt", "hgvs_p", "hgvs_c",
    "chrom", "hg38_pos", "ref_allele", "alt_allele",
    "variant_type_harmonized", "ensembl_transcript_id",
    "refseq_transcript_id",
}

# Internal metadata columns that should never be carried from the right side;
# they are set per-row by each loader and would cause duplicate-column conflicts
# when the same source label is merged more than once.
_RIGHT_EXCLUDE = {"coord_mapping_method", "source_file", "_source"}


def _merge_source(
    left: pd.DataFrame,
    right: pd.DataFrame,
    source_label: str,
) -> pd.DataFrame:
    """Merge a single source into the master dataframe.

    Strategy:
    1. Rows in *right* with ``genomic_coord`` -> outer-merge on ``genomic_coord``
    2. Remaining rows with ``(gene, aa_pos, aa_ref, aa_alt)`` -> merge on
       the protein-level compound key (matches Pillar rows that share the
       same amino-acid change)
    3. Remaining rows -> appended as unmatched
    """
    if right.empty:
        logger.info("  Skipping %s (empty)", source_label)
        return left

    right = right.copy()

    # ── Partition right into coord-based, protein-based, and unmatched ───
    if "genomic_coord" in right.columns:
        has_coord = right["genomic_coord"].notna()
    else:
        has_coord = pd.Series(False, index=right.index)

    has_protein_cols = all(c in right.columns for c in ["gene", "aa_pos", "aa_ref", "aa_alt"])
    if has_protein_cols:
        has_protein = (
            right["gene"].notna()
            & right["aa_pos"].notna()
            & right["aa_ref"].notna()
            & right["aa_alt"].notna()
        )
    else:
        has_protein = pd.Series(False, index=right.index)

    right_coord = right[has_coord]
    right_protein = right[~has_coord & has_protein]
    right_rest = right[~has_coord & ~has_protein]

    logger.info(
        "  %s: %d with genomic_coord, %d protein-level only, %d unmatched",
        source_label, len(right_coord), len(right_protein), len(right_rest),
    )

    # Columns to carry from right: exclude shared-identity fields (handled by
    # backfill) and internal metadata columns that live on the left side already.
    right_only = lambda df_r, merge_keys: [
        c for c in df_r.columns
        if (c not in _SHARED_IDENTITY and c not in _RIGHT_EXCLUDE) or c in merge_keys
    ]

    # ── (1) Merge on genomic_coord ───────────────────────────────────────
    if not right_coord.empty:
        cols = right_only(right_coord, {"genomic_coord"})
        right_dedup = right_coord[cols].drop_duplicates(subset="genomic_coord", keep="first")
        before = len(left)
        left = left.merge(right_dedup, on="genomic_coord", how="outer", suffixes=("", f"_{source_label}"))
        _backfill_identity(left, source_label)
        _update_source_datasets(left, right_dedup["genomic_coord"], source_label)
        logger.info(
            "    genomic_coord merge: %d -> %d rows (delta %+d)",
            before, len(left), len(left) - before,
        )

    # ── (2) Merge on protein-level key ───────────────────────────────────
    if not right_protein.empty:
        protein_keys = ["gene", "aa_pos", "aa_ref", "aa_alt"]
        # Coerce aa_pos to same type for merge
        right_protein = right_protein.copy()
        right_protein["aa_pos"] = pd.to_numeric(right_protein["aa_pos"], errors="coerce").astype("Int64")
        left["aa_pos"] = pd.to_numeric(left["aa_pos"], errors="coerce").astype("Int64")

        cols = right_only(right_protein, set(protein_keys))
        right_dedup = right_protein[cols].drop_duplicates(subset=protein_keys, keep="first")
        before = len(left)
        left = left.merge(right_dedup, on=protein_keys, how="outer", suffixes=("", f"_{source_label}"))
        _backfill_identity(left, source_label)

        # Update source_datasets for protein-matched rows
        came_from_right = left["source_datasets"].isna()
        left.loc[came_from_right, "source_datasets"] = source_label
        # Mark existing rows that matched on protein key
        right_key_set = set(
            right_dedup[protein_keys].apply(lambda r: (r["gene"], r["aa_pos"], r["aa_ref"], r["aa_alt"]), axis=1)
        )
        for idx in left.index:
            if left.at[idx, "source_datasets"] and source_label not in str(left.at[idx, "source_datasets"]):
                row_key = (left.at[idx, "gene"], left.at[idx, "aa_pos"], left.at[idx, "aa_ref"], left.at[idx, "aa_alt"])
                if row_key in right_key_set:
                    left.at[idx, "source_datasets"] = str(left.at[idx, "source_datasets"]) + "|" + source_label

        logger.info(
            "    protein-key merge: %d -> %d rows (delta %+d)",
            before, len(left), len(left) - before,
        )

    # ── (3) Append unmatched ─────────────────────────────────────────────
    if not right_rest.empty:
        right_rest = right_rest.copy()
        right_rest["source_datasets"] = source_label
        left = pd.concat([left, right_rest], ignore_index=True)
        logger.info("    Appended %d unmatched rows from %s", len(right_rest), source_label)

    return left


def _backfill_identity(df: pd.DataFrame, source_label: str) -> None:
    """Back-fill shared identity columns from the right side of a merge."""
    for col in _SHARED_IDENTITY:
        right_col = f"{col}_{source_label}"
        if right_col in df.columns:
            mask = df[col].isna() & df[right_col].notna()
            df.loc[mask, col] = df.loc[mask, right_col]
            df.drop(columns=[right_col], inplace=True)


def _update_source_datasets(
    df: pd.DataFrame,
    right_keys: pd.Series,
    source_label: str,
) -> None:
    """Update source_datasets for newly merged rows."""
    came_from_right = df["source_datasets"].isna()
    df.loc[came_from_right, "source_datasets"] = source_label
    matched_existing = (
        df["source_datasets"].notna()
        & ~came_from_right
        & df["genomic_coord"].isin(right_keys)
    )
    df.loc[matched_existing, "source_datasets"] = (
        df.loc[matched_existing, "source_datasets"] + "|" + source_label
    )


def merge_all(
    pillar: pd.DataFrame,
    labelseq_frames: list[pd.DataFrame],
    fisseq_frames: list[pd.DataFrame],
    fisseq_feature_frames: list[pd.DataFrame] | None = None,
    vampseq_frames: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Outer-merge all source dataframes.

    Merge order: Pillar -> LabelSEQ -> FisSEQ -> FisSEQ features -> VampSEQ.
    Uses genomic_coord as primary key with (gene, aa_pos, aa_ref, aa_alt)
    as protein-level fallback for sources without genomic coordinates.
    """
    master = pillar.copy()
    master["source_datasets"] = "pillar"
    logger.info("Backbone: %d rows from Pillar", len(master))

    for ls in labelseq_frames:
        master = _merge_source(master, ls, "labelseq")

    for fs in fisseq_frames:
        gene_tag = fs["gene"].iloc[0] if "gene" in fs.columns and len(fs) > 0 else "unknown"
        master = _merge_source(master, fs, f"fisseq_{gene_tag}".lower())

    if fisseq_feature_frames:
        fisseq_features_combined = pd.concat(fisseq_feature_frames, ignore_index=True)
        logger.info(
            "  Combined %d FisSEQ feature frames into %d rows",
            len(fisseq_feature_frames), len(fisseq_features_combined),
        )
        master = _merge_source(master, fisseq_features_combined, "fisseq_features")

    if vampseq_frames:
        vampseq_combined = pd.concat(vampseq_frames, ignore_index=True)
        logger.info(
            "  Combined %d VampSEQ frames into %d rows",
            len(vampseq_frames), len(vampseq_combined),
        )
        master = _merge_source(master, vampseq_combined, "vampseq")

    logger.info("Final merged dataframe: %d rows x %d cols", *master.shape)
    return master
