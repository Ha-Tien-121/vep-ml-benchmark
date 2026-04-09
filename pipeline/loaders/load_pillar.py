"""Loader for Pillar (canonical variant backbone) datasets.

Pillar CSVs contain hg38 genomic coordinates, transcript IDs, HGVS notation,
amino-acid changes, DMS functional scores, and pre-computed predictor scores
(AlphaMissense, REVEL, SpliceAI).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import VARIANT_TYPE_MAP

logger = logging.getLogger(__name__)

# Explicit rename map: raw Pillar column → unified schema column
_RENAME: dict[str, str] = {
    "ID": "variant_id",
    "Gene": "gene",
    "HGNC_id": "hgnc_id",
    "Chrom": "chrom",
    "STRAND": "strand",
    "hg19_pos": "hg19_pos",
    "hg38_start": "hg38_pos",
    "hg38_end": "hg38_end",
    "ref_allele": "ref_allele",
    "alt_allele": "alt_allele",
    "auth_transcript_id": "refseq_transcript_id",
    "Ensembl_transcript_ID": "ensembl_transcript_id",
    "transcript_pos": "transcript_pos",
    "transcript_ref": "transcript_ref",
    "transcript_alt": "transcript_alt",
    "aa_pos": "aa_pos",
    "aa_ref": "aa_ref",
    "aa_alt": "aa_alt",
    "hgvs_c": "hgvs_c",
    "hgvs_p": "hgvs_p",
    "consequence": "consequence",
    "simplified_consequence": "simplified_consequence",
    "auth_reported_score": "functional_score_pillar",
    "auth_reported_rep_score": "functional_score_pillar_rep",
    "auth_reported_func_class": "functional_class_pillar",
    "splice_measure": "splice_measure",
    "gnomad_MAF": "gnomad_maf",
    "nucleotide_or_aa": "nucleotide_or_aa",
    "AM_score": "alphamissense_score",
    "AM_class": "alphamissense_label",
    "REVEL": "revel_score",
    "Dataset": "pillar_dataset_id",
    "MaveDB Score Set URN": "mavedb_urn",
    "Flag": "pillar_flag",
}

_SPLICEAI_DS_COLS = [
    "spliceAI_DS_AG",
    "spliceAI_DS_AL",
    "spliceAI_DS_DG",
    "spliceAI_DS_DL",
]


def _take_first_caret(series: pd.Series) -> pd.Series:
    """Extract the first value from caret-separated multi-position fields.

    Many Pillar rows encode amino-acid-level variants with multiple codon
    positions/alleles joined by ``^`` (e.g. ``3476163.0^3476163.0``).
    This helper takes the first token so downstream numeric coercion works.
    """
    return series.astype(str).str.split("^", n=1).str[0].replace({"nan": pd.NA, "": pd.NA, "None": pd.NA})


def load_pillar(filepath: str | Path) -> pd.DataFrame:
    """Read a Pillar CSV, rename to unified schema, and compute derived columns.

    Returns a DataFrame with:
    - Columns renamed per the unified schema
    - ``genomic_coord`` in ``chr{chrom}:{pos}:{ref}:{alt}`` format
    - ``spliceai_max_ds`` from the four SpliceAI delta-score columns
    - ``coord_mapping_method = 'direct'``
    - ``_source = 'pillar'``
    """
    filepath = Path(filepath)
    logger.info("Loading Pillar file: %s", filepath.name)
    df = pd.read_csv(filepath, low_memory=False)
    logger.info("  Raw shape: %d rows x %d cols", *df.shape)

    # SpliceAI max delta score (before rename)
    splice_present = [c for c in _SPLICEAI_DS_COLS if c in df.columns]
    if splice_present:
        df["spliceai_max_ds"] = (
            df[splice_present]
            .apply(pd.to_numeric, errors="coerce")
            .max(axis=1)
        )
    else:
        df["spliceai_max_ds"] = np.nan

    # ── Handle caret-separated multi-position fields ─────────────────
    # AA-level variants can have values like "3476163.0^3476163.0" for
    # hg38_start, "A^ACT^ACT" for ref_allele, etc.  We keep the full
    # raw values as pillar__*_raw and extract the first token for the
    # unified columns.
    caret_cols = ["hg38_start", "hg38_end", "hg19_pos", "ref_allele", "alt_allele"]
    for col in caret_cols:
        if col in df.columns and df[col].astype(str).str.contains(r"\^", na=False).any():
            raw_name = f"{col}_raw"
            df[raw_name] = df[col]  # preserve full value
            df[col] = _take_first_caret(df[col])
            n_split = df[raw_name].astype(str).str.contains(r"\^", na=False).sum()
            logger.info("  %s: extracted first token from %d caret-separated values", col, n_split)

    # Rename core columns
    rename_map = {k: v for k, v in _RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Prefix remaining (un-renamed) raw columns as pillar__*
    renamed_targets = set(rename_map.values())
    already = set(rename_map.keys()) | renamed_targets | {"spliceai_max_ds"}
    for col in list(df.columns):
        if col not in already and col not in renamed_targets:
            df = df.rename(columns={col: f"pillar__{col}"})

    # Coerce types
    df["hg38_pos"] = pd.to_numeric(df["hg38_pos"], errors="coerce").astype("Int64")
    df["hg19_pos"] = pd.to_numeric(df.get("hg19_pos"), errors="coerce").astype("Int64")
    df["aa_pos"] = pd.to_numeric(df["aa_pos"], errors="coerce").astype("Int64")
    df["functional_score_pillar"] = pd.to_numeric(
        df["functional_score_pillar"], errors="coerce"
    )
    df["alphamissense_score"] = pd.to_numeric(
        df.get("alphamissense_score"), errors="coerce"
    )
    df["strand"] = pd.to_numeric(df.get("strand"), errors="coerce").astype("Int64")

    # Genomic coordinate (primary merge key) — only build where all parts exist
    df["chrom"] = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    df["chrom"] = "chr" + df["chrom"]

    has_all = (
        df["hg38_pos"].notna()
        & df["ref_allele"].notna()
        & df["alt_allele"].notna()
    )
    df["genomic_coord"] = pd.NA
    df.loc[has_all, "genomic_coord"] = (
        df.loc[has_all, "chrom"].astype(str)
        + ":" + df.loc[has_all, "hg38_pos"].astype(str)
        + ":" + df.loc[has_all, "ref_allele"].astype(str)
        + ":" + df.loc[has_all, "alt_allele"].astype(str)
    )

    # Harmonise variant type
    if "simplified_consequence" in df.columns:
        df["variant_type_harmonized"] = (
            df["simplified_consequence"]
            .map(VARIANT_TYPE_MAP)
            .fillna("other")
        )
    else:
        df["variant_type_harmonized"] = "other"

    # Provenance
    df["coord_mapping_method"] = "direct"
    df["_source"] = "pillar"
    df["source_file"] = filepath.name

    n_coords = df["genomic_coord"].notna().sum()
    logger.info(
        "  Loaded %d rows; %d with genomic_coord; %d with AM score",
        len(df),
        n_coords,
        df["alphamissense_score"].notna().sum(),
    )
    return df
