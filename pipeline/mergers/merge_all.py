"""Merge all source dataframes into one unified benchmark dataframe.

Pillar is the left/backbone.  Other sources are outer-joined on
``genomic_coord`` so that variants unique to any assay are preserved.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def merge_all(
    pillar: pd.DataFrame,
    labelseq_frames: list[pd.DataFrame],
    fisseq_frames: list[pd.DataFrame],
    vampseq_frames: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Outer-merge all source dataframes on ``genomic_coord``.

    Merge order: Pillar → LabelSEQ → FisSEQ → VampSEQ (stub).

    For shared columns (``gene``, ``aa_pos``, etc.) Pillar values take
    precedence; other sources fill only where Pillar is null.
    """
    master = pillar.copy()
    master["source_datasets"] = "pillar"
    logger.info("Backbone: %d rows from Pillar", len(master))

    # ── Helper: controlled outer merge ────────────────────────────────────
    def _merge_source(
        left: pd.DataFrame,
        right: pd.DataFrame,
        source_label: str,
    ) -> pd.DataFrame:
        if right.empty or "genomic_coord" not in right.columns:
            logger.info("  Skipping %s (empty or no genomic_coord)", source_label)
            return left

        right_has_coord = right["genomic_coord"].notna()
        right_valid = right[right_has_coord]
        right_no_coord = right[~right_has_coord]
        if not right_valid.empty:
            logger.info(
                "  Merging %s: %d rows with coord, %d without",
                source_label, len(right_valid), len(right_no_coord),
            )
        else:
            logger.info(
                "  %s: 0 rows have genomic_coord — appending as unmatched",
                source_label,
            )

        # Identify columns that exist in both frames
        shared_identity = {"gene", "aa_pos", "aa_ref", "aa_alt", "hgvs_p",
                          "hgvs_c", "chrom", "hg38_pos", "ref_allele",
                          "alt_allele", "variant_type_harmonized",
                          "ensembl_transcript_id", "refseq_transcript_id"}
        # Drop shared identity columns from right to prevent _x/_y duplication;
        # we'll back-fill from right after merge for rows that were null in left.
        right_only_cols = [
            c for c in right_valid.columns
            if c not in shared_identity or c == "genomic_coord"
        ]

        # Deduplicate right on genomic_coord (keep first)
        right_dedup = right_valid[right_only_cols].drop_duplicates(
            subset="genomic_coord", keep="first"
        )

        before = len(left)
        merged = left.merge(right_dedup, on="genomic_coord", how="outer", suffixes=("", f"_{source_label}"))

        # Back-fill identity columns from right where left was null
        for col in shared_identity:
            right_col = f"{col}_{source_label}"
            if right_col in merged.columns:
                mask = merged[col].isna() & merged[right_col].notna()
                merged.loc[mask, col] = merged.loc[mask, right_col]
                merged = merged.drop(columns=[right_col])

        # Update source_datasets
        new_rows = merged.index[before:] if len(merged) > before else pd.Index([])
        # For rows that came from the right frame only
        came_from_right = merged["source_datasets"].isna()
        merged.loc[came_from_right, "source_datasets"] = source_label
        # For rows that existed in left and also matched right
        matched_existing = (
            merged["source_datasets"].notna()
            & ~came_from_right
            & merged["genomic_coord"].isin(right_dedup["genomic_coord"])
        )
        merged.loc[matched_existing, "source_datasets"] = (
            merged.loc[matched_existing, "source_datasets"] + "|" + source_label
        )

        logger.info(
            "  After %s merge: %d rows (was %d, delta %+d)",
            source_label, len(merged), before, len(merged) - before,
        )

        # Append rows without coords as unmatched
        if not right_no_coord.empty:
            no_coord_df = right_no_coord.copy()
            no_coord_df["source_datasets"] = source_label
            merged = pd.concat([merged, no_coord_df], ignore_index=True)
            logger.info(
                "  Appended %d no-coord rows from %s", len(right_no_coord), source_label,
            )

        return merged

    # ── Merge each source ─────────────────────────────────────────────────
    for i, ls in enumerate(labelseq_frames):
        master = _merge_source(master, ls, f"labelseq")

    for i, fs in enumerate(fisseq_frames):
        gene_tag = fs["gene"].iloc[0] if "gene" in fs.columns and len(fs) > 0 else f"fisseq_{i}"
        master = _merge_source(master, fs, f"fisseq_{gene_tag}".lower())

    # TODO: Merge VampSEQ once load_vampseq() is implemented
    if vampseq_frames:
        for i, vs in enumerate(vampseq_frames):
            master = _merge_source(master, vs, "vampseq")

    logger.info(
        "Final merged dataframe: %d rows x %d cols", *master.shape
    )
    return master
