"""Loaders for FisSEQ supplementary summary tables and feature matrices.

FisSEQ data uses compact protein-level variant notation (e.g. ``L263Y``) with
no genomic coordinates.  Gene identity is inferred from the filename.

Two table types exist:
- **Supplementary summary tables** (compact, ~13–20 cols): variant, scores, clusters.
- **Feature matrices** (hundreds of morphology columns): kept as summary only.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import FISSEQ_FILE_TO_GENE, VARIANT_TYPE_MAP

logger = logging.getLogger(__name__)

_COMPACT_VARIANT_RE = re.compile(r"^([A-Z*])(\d+)([A-Z*\-])$")


def _parse_compact_variant(variant: str) -> tuple[str, int, str] | None:
    """Parse ``L263Y`` → ``('L', 263, 'Y')``.  Returns None on failure."""
    if not isinstance(variant, str):
        return None
    m = _COMPACT_VARIANT_RE.match(variant.strip())
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None


def _infer_gene(filename: str) -> str:
    gene = FISSEQ_FILE_TO_GENE.get(filename)
    if gene:
        return gene
    lower = filename.lower()
    if "lmna" in lower:
        return "LMNA"
    if "pten" in lower:
        return "PTEN"
    logger.warning("Cannot infer gene from filename '%s'; using UNKNOWN", filename)
    return "UNKNOWN"


# ── Supplementary summary tables ──────────────────────────────────────────

# Supp1 (LMNA) column rename
_SUPP1_RENAME: dict[str, str] = {
    "Morphological Impact Score": "functional_score_fisseq",
    "Cluster": "fisseq_cluster",
    "AUROC Score": "fisseq_auroc",
    "Lamin A Nuclear Intensity median z-score": "fisseq_lmna_intensity_zscore",
    "Lamin A Nuclear Boundary Intensity median z-score": "fisseq_lmna_boundary_zscore",
    "Lamin A Nuclear Granularity 1 median z-score": "fisseq_lmna_granularity_zscore",
    "Nuclear Shape Factor median z-score": "fisseq_lmna_shape_zscore",
    "Num Cells Replicate 1": "fisseq_num_cells_rep1",
    "Num Cells Replicate 2": "fisseq_num_cells_rep2",
    "UMAP1": "fisseq_umap1",
    "UMAP2": "fisseq_umap2",
}

# Supp2 (PTEN) column rename
_SUPP2_RENAME: dict[str, str] = {
    "Morphological Impact Score iPSC": "fisseq_pten_ipsc_morphscore",
    "Morphological Impact Score neuron": "fisseq_pten_neuron_morphscore",
    "AUROC Score iPSC": "fisseq_pten_ipsc_auroc",
    "AUROC Score neuron": "fisseq_pten_neuron_auroc",
    "iPSC PTEN Intensity median z-score": "fisseq_pten_ipsc_intensity_zscore",
    "Neuron PTEN Intensity median z-score": "fisseq_pten_neuron_intensity_zscore",
    "iPSC pAKT Intensity median z-score": "fisseq_pten_ipsc_pakt_zscore",
    "Neuron pAKT Intensity median z-score": "fisseq_pten_neuron_pakt_zscore",
    "iPSC DAPI-PTEN Corr median z-score": "fisseq_pten_ipsc_dapi_corr_zscore",
    "Neuron DAPI-PTEN Corr median z-score": "fisseq_pten_neuron_dapi_corr_zscore",
    "Num Cells iPSC Replicate 1": "fisseq_num_cells_rep1",
    "Num Cells iPSC Replicate 2": "fisseq_num_cells_rep2",
    "Num Cells neuron Replicate 1": "fisseq_num_cells_neuron_rep1",
    "Num Cells neuron Replicate 2": "fisseq_num_cells_neuron_rep2",
    "UMAP1 iPSC": "fisseq_umap1_ipsc",
    "UMAP2 iPSC": "fisseq_umap2_ipsc",
    "UMAP1 neuron": "fisseq_umap1_neuron",
    "UMAP2 neuron": "fisseq_umap2_neuron",
}


def _apply_variant_parsing(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``Variant`` column into ``aa_ref``, ``aa_pos``, ``aa_alt``."""
    parsed = df["Variant"].map(_parse_compact_variant)
    df["aa_ref"] = parsed.map(lambda x: x[0] if x else None)
    df["aa_pos"] = parsed.map(lambda x: x[1] if x else pd.NA).astype("Int64")
    df["aa_alt"] = parsed.map(lambda x: x[2] if x else None)

    n_parsed = parsed.notna().sum()
    n_failed = parsed.isna().sum()
    if n_failed:
        failed_examples = df.loc[parsed.isna(), "Variant"].head(5).tolist()
        logger.warning(
            "  %d / %d variants failed to parse. Examples: %s",
            n_failed, len(df), failed_examples,
        )
    else:
        logger.info("  All %d variants parsed successfully", n_parsed)
    return df


def _build_hgvs_p(row: pd.Series) -> str | None:
    """Build HGVS protein string from parsed variant fields."""
    from pipeline.config import AA1TO3
    ref = row.get("aa_ref")
    alt = row.get("aa_alt")
    pos = row.get("aa_pos")
    if not ref or not alt or pd.isna(pos):
        return None
    ref3 = AA1TO3.get(ref, ref)
    if alt == "-":
        return f"p.{ref3}{int(pos)}del"
    if alt == "*":
        return f"p.{ref3}{int(pos)}Ter"
    alt3 = AA1TO3.get(alt, alt)
    return f"p.{ref3}{int(pos)}{alt3}"


def _harmonize_variant_type(df: pd.DataFrame) -> pd.DataFrame:
    """Create ``variant_type_harmonized`` from ``Variant Type`` and parsed alt."""
    if "Variant Type" in df.columns:
        df["variant_type_harmonized"] = (
            df["Variant Type"].map(VARIANT_TYPE_MAP).fillna("other")
        )
    else:
        df["variant_type_harmonized"] = "other"

    # Override based on parsed alt for special cases
    if "aa_alt" in df.columns:
        df.loc[df["aa_alt"] == "-", "variant_type_harmonized"] = "frameshift"
        df.loc[df["aa_alt"] == "*", "variant_type_harmonized"] = "nonsense"

    return df


def load_fisseq_supp(filepath: str | Path) -> pd.DataFrame:
    """Load a FisSEQ supplementary summary table.

    Detects whether this is a Supp1-style (LMNA, single cell type) or
    Supp2-style (PTEN, dual cell types) table based on column presence.
    """
    filepath = Path(filepath)
    gene = _infer_gene(filepath.name)
    logger.info("Loading FisSEQ supp: %s (gene=%s)", filepath.name, gene)

    df = pd.read_csv(filepath, low_memory=False)
    logger.info("  Raw shape: %d rows x %d cols", *df.shape)

    # Detect table type
    is_supp2 = "Morphological Impact Score iPSC" in df.columns
    rename_map = _SUPP2_RENAME if is_supp2 else _SUPP1_RENAME

    df = _apply_variant_parsing(df)
    df["gene"] = gene
    df["hgvs_p"] = df.apply(_build_hgvs_p, axis=1)

    # For PTEN Supp2: compute average functional score across cell types
    if is_supp2:
        ipsc = pd.to_numeric(df.get("Morphological Impact Score iPSC"), errors="coerce")
        neuron = pd.to_numeric(df.get("Morphological Impact Score neuron"), errors="coerce")
        df["functional_score_fisseq"] = np.nanmean(
            np.column_stack([ipsc.values, neuron.values]), axis=1
        )
        ipsc_auroc = pd.to_numeric(df.get("AUROC Score iPSC"), errors="coerce")
        neuron_auroc = pd.to_numeric(df.get("AUROC Score neuron"), errors="coerce")
        df["fisseq_auroc"] = np.nanmean(
            np.column_stack([ipsc_auroc.values, neuron_auroc.values]), axis=1
        )
        df["fisseq_cluster"] = pd.NA  # no single cluster for dual-cell-type data

    # Apply the rename map
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=actual_rename)

    df = _harmonize_variant_type(df)

    # Coerce numeric types
    for col in ["functional_score_fisseq", "fisseq_auroc", "fisseq_cluster"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
    for col in df.columns:
        if "num_cells" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Drop raw columns that were renamed
    raw_variant_type = "Variant Type"
    if raw_variant_type in df.columns:
        df = df.drop(columns=[raw_variant_type])

    df["coord_mapping_method"] = "protein_lifted"
    df["_source"] = "fisseq"
    df["source_file"] = filepath.name

    return df


# ── Feature matrices ──────────────────────────────────────────────────────

def load_fisseq_features(filepath: str | Path) -> pd.DataFrame:
    """Load a FisSEQ morphology feature matrix.

    These files can be very wide (300+ columns).  We extract only the summary
    columns (Variant, Variant_Class, Morphological Impact Score, cluster info,
    UMAP, cell counts) and discard the raw morphology features to keep the
    master dataframe manageable.
    """
    filepath = Path(filepath)
    gene = _infer_gene(filepath.name)
    logger.info("Loading FisSEQ features: %s (gene=%s)", filepath.name, gene)

    df = pd.read_csv(filepath, low_memory=False, nrows=5)
    all_cols = list(df.columns)
    logger.info("  Detected %d columns", len(all_cols))

    # Identify columns worth keeping
    keep = {"Variant", "Variant_Class", "Morphological Impact Score",
            "Cluster", "Louvain Cluster"}
    keep_cols = [c for c in all_cols if (
        c in keep
        or c.startswith("UMAP")
        or c.startswith("NCells")
    )]
    # Drop unnamed index columns
    keep_cols = [c for c in keep_cols if not c.startswith("Unnamed")]

    logger.info(
        "  Keeping %d summary cols out of %d total (discarding %d feature cols)",
        len(keep_cols), len(all_cols), len(all_cols) - len(keep_cols),
    )

    df = pd.read_csv(filepath, low_memory=False, usecols=keep_cols)
    logger.info("  Loaded %d rows", len(df))

    df = _apply_variant_parsing(df)
    df["gene"] = gene

    # Rename summary columns with fisseq__ prefix
    rename_map: dict[str, str] = {}
    for c in df.columns:
        if c not in ("Variant", "gene", "aa_ref", "aa_pos", "aa_alt"):
            safe = c.lower().replace(" ", "_").replace("-", "_")
            rename_map[c] = f"fisseq_feat__{safe}"
    df = df.rename(columns=rename_map)

    df["_source"] = "fisseq"
    df["source_file"] = filepath.name
    return df
