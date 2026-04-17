"""Loader for VampSEQ (VAMP-seq) data from MaveDB exports.

VampSEQ measures protein abundance as a proxy for stability / function.
Data arrives as MaveDB-format CSVs with columns:
    accession, hgvs_nt, hgvs_splice, hgvs_pro,
    mavedb.post_mapped_hgvs_g, mavedb.post_mapped_hgvs_p,
    mavedb.post_mapped_hgvs_c, mavedb.post_mapped_hgvs_at_assay_level,
    mavedb.post_mapped_vrs_digest, scores.score

The ``mavedb.post_mapped_hgvs_p`` column provides the canonical protein
position (e.g. ``NP_000539.2:p.Ala563Cys``), which we use to extract the
amino acid change on the full-length protein.  Gene identity is resolved
from the RefSeq protein accession.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from pipeline.config import AA3TO1, VAMPSEQ_REFSEQ_TO_GENE

logger = logging.getLogger(__name__)

_HGVS_P_RE = re.compile(
    r"(?:.*:)?p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=|del|dup|fs)"
)


def _parse_hgvs_p(hgvs: str) -> tuple[str, int, str] | None:
    """Parse HGVS protein notation into (aa_ref_1letter, position, aa_alt_1letter).

    Handles mapped form ``NP_xxx:p.Ala103Arg`` and local form ``p.Ala103Arg``.
    For synonymous (``=``): alt is set to the same as ref.
    """
    if not isinstance(hgvs, str):
        return None
    m = _HGVS_P_RE.match(hgvs.strip())
    if not m:
        return None
    ref3, pos_str, alt_raw = m.group(1), m.group(2), m.group(3)
    ref1 = AA3TO1.get(ref3, ref3)
    pos = int(pos_str)
    if alt_raw == "=":
        alt1 = ref1
    elif alt_raw in ("del", "dup", "fs"):
        alt1 = "-"
    else:
        alt1 = AA3TO1.get(alt_raw, alt_raw)
    return ref1, pos, alt1


def _extract_refseq_id(mapped_hgvs: str) -> str | None:
    """Extract the RefSeq accession from ``NP_xxx.y:p.Ala103Arg``."""
    if not isinstance(mapped_hgvs, str) or ":" not in mapped_hgvs:
        return None
    return mapped_hgvs.split(":")[0].strip()


def _gene_from_refseq(refseq_id: str | None) -> str:
    """Resolve gene symbol from a RefSeq protein accession."""
    if not refseq_id:
        return "UNKNOWN"
    return VAMPSEQ_REFSEQ_TO_GENE.get(refseq_id, "UNKNOWN")


def _classify_variant(ref: str | None, alt: str | None) -> str:
    """Assign a harmonized variant type from single-letter AA change."""
    if not ref or not alt:
        return "other"
    if ref == alt:
        return "synonymous"
    if alt == "*":
        return "nonsense"
    if alt == "-":
        return "in_frame_indel"
    return "missense"


def _load_vampseq_abundance(
    df: pd.DataFrame,
    filepath: Path,
    gene_override: str | None,
) -> pd.DataFrame:
    """Handle abundance-score format: ``variant`` (HGVS-p or ENSP:HGVS-p) + ``abundance_avg``."""
    fmt = "raw-abundance" if "aaChanges" in df.columns else "processed-abundance"
    logger.info("  Detected %s format", fmt)

    parsed = df["variant"].map(_parse_hgvs_p)
    df["aa_ref"] = parsed.map(lambda x: x[0] if x else None)
    df["aa_pos"] = parsed.map(lambda x: x[1] if x else pd.NA).astype("Int64")
    df["aa_alt"] = parsed.map(lambda x: x[2] if x else None)

    n_failed = parsed.isna().sum()
    if n_failed:
        logger.warning(
            "  %d / %d variants failed to parse. Examples: %s",
            n_failed, len(df),
            df.loc[parsed.isna(), "variant"].head(5).tolist(),
        )
    else:
        logger.info("  All %d variants parsed successfully", parsed.notna().sum())

    df["gene"] = gene_override if gene_override else "UNKNOWN"

    from pipeline.config import AA1TO3
    def _build_hgvs_p_row(row: pd.Series) -> str | None:
        ref, alt, pos = row.get("aa_ref"), row.get("aa_alt"), row.get("aa_pos")
        if not ref or not alt or pd.isna(pos):
            return None
        ref3 = AA1TO3.get(ref, ref)
        if ref == alt:
            return f"p.{ref3}{int(pos)}="
        if alt == "-":
            return f"p.{ref3}{int(pos)}del"
        if alt == "*":
            return f"p.{ref3}{int(pos)}Ter"
        return f"p.{ref3}{int(pos)}{AA1TO3.get(alt, alt)}"

    df["hgvs_p"] = df.apply(_build_hgvs_p_row, axis=1)

    df["variant_type_harmonized"] = df.apply(
        lambda r: _classify_variant(r.get("aa_ref"), r.get("aa_alt")), axis=1
    )

    df["functional_score_vampseq"] = pd.to_numeric(
        df.get("abundance_avg"), errors="coerce"
    )

    # Rename abundance columns with vampseq_ prefix
    rename_map: dict[str, str] = {}
    for col in ("variant", "abundance_1", "abundance_2", "abundance_3",
                "abundance_avg", "variance", "std_dev",
                "QC_flag", "functional_consequence",
                "aaChanges", "aaChanges_type"):
        if col in df.columns:
            rename_map[col] = f"vampseq_{col}"
    df = df.rename(columns=rename_map)

    df["coord_mapping_method"] = "protein_lifted"
    df["_source"] = "vampseq"
    df["source_file"] = filepath.name

    logger.info(
        "  Loaded %d rows; gene=%s; %d with functional score; types: %s",
        len(df),
        df["gene"].iloc[0] if len(df) else "N/A",
        df["functional_score_vampseq"].notna().sum(),
        df["variant_type_harmonized"].value_counts().to_dict(),
    )
    return df


def load_vampseq(filepath: str | Path, gene_override: str | None = None) -> pd.DataFrame:
    """Load a VampSEQ MaveDB CSV and map to unified schema columns.

    Parameters
    ----------
    filepath
        Path to a VampSEQ MaveDB-format CSV.
    gene_override
        If given, use this gene symbol instead of resolving from RefSeq.

    Returns
    -------
    pd.DataFrame
        Unified schema with ``gene``, ``aa_ref``, ``aa_pos``, ``aa_alt``,
        ``functional_score_vampseq``, ``variant_type_harmonized``, etc.
    """
    filepath = Path(filepath)
    logger.info("Loading VampSEQ: %s", filepath.name)

    df = pd.read_csv(filepath, low_memory=False)
    logger.info("  Raw shape: %d rows x %d cols", *df.shape)

    if df.empty:
        logger.warning("  VampSEQ file is empty: %s", filepath.name)
        return df

    # Detect processed-abundance format (variant + abundance_avg, no MaveDB columns)
    if "abundance_avg" in df.columns and "mavedb.post_mapped_hgvs_p" not in df.columns:
        return _load_vampseq_abundance(df, filepath, gene_override)

    # Prefer the mapped HGVS-p (canonical positions) over the local hgvs_pro
    mapped_col = "mavedb.post_mapped_hgvs_p"
    local_col = "hgvs_pro"
    use_col = mapped_col if mapped_col in df.columns else local_col

    parsed = df[use_col].map(_parse_hgvs_p)
    df["aa_ref"] = parsed.map(lambda x: x[0] if x else None)
    df["aa_pos"] = parsed.map(lambda x: x[1] if x else pd.NA).astype("Int64")
    df["aa_alt"] = parsed.map(lambda x: x[2] if x else None)

    n_parsed = parsed.notna().sum()
    n_failed = parsed.isna().sum()
    if n_failed:
        failed_examples = df.loc[parsed.isna(), use_col].head(5).tolist()
        logger.warning(
            "  %d / %d variants failed to parse. Examples: %s",
            n_failed, len(df), failed_examples,
        )
    else:
        logger.info("  All %d variants parsed successfully", n_parsed)

    # Gene resolution
    if gene_override:
        df["gene"] = gene_override
    elif mapped_col in df.columns:
        refseq_ids = df[mapped_col].map(_extract_refseq_id)
        df["gene"] = refseq_ids.map(_gene_from_refseq)
        n_resolved = (df["gene"] != "UNKNOWN").sum()
        logger.info("  Gene resolved for %d / %d rows", n_resolved, len(df))
    else:
        df["gene"] = "UNKNOWN"
        logger.warning("  No mapped HGVS-p column; gene cannot be resolved")

    # Build HGVS-p from parsed components
    from pipeline.config import AA1TO3
    def _build_hgvs_p(row: pd.Series) -> str | None:
        ref = row.get("aa_ref")
        alt = row.get("aa_alt")
        pos = row.get("aa_pos")
        if not ref or not alt or pd.isna(pos):
            return None
        ref3 = AA1TO3.get(ref, ref)
        if ref == alt:
            return f"p.{ref3}{int(pos)}="
        if alt == "-":
            return f"p.{ref3}{int(pos)}del"
        if alt == "*":
            return f"p.{ref3}{int(pos)}Ter"
        alt3 = AA1TO3.get(alt, alt)
        return f"p.{ref3}{int(pos)}{alt3}"

    df["hgvs_p"] = df.apply(_build_hgvs_p, axis=1)

    # Variant type classification
    df["variant_type_harmonized"] = df.apply(
        lambda r: _classify_variant(r.get("aa_ref"), r.get("aa_alt")),
        axis=1,
    )

    # Functional score
    df["functional_score_vampseq"] = pd.to_numeric(
        df.get("scores.score"), errors="coerce"
    )

    # Preserve MaveDB metadata columns with vampseq_ prefix
    rename_map = {
        "accession": "vampseq_accession",
        "hgvs_pro": "vampseq_hgvs_pro_local",
        "mavedb.post_mapped_hgvs_p": "vampseq_mapped_hgvs_p",
        "mavedb.post_mapped_hgvs_g": "vampseq_mapped_hgvs_g",
        "mavedb.post_mapped_hgvs_c": "vampseq_mapped_hgvs_c",
        "mavedb.post_mapped_vrs_digest": "vampseq_vrs_digest",
        "scores.score": "vampseq_raw_score",
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=actual_rename)

    # Drop columns that are not useful
    for col in ("hgvs_nt", "hgvs_splice",
                "mavedb.post_mapped_hgvs_at_assay_level"):
        if col in df.columns:
            df = df.drop(columns=[col])

    df["coord_mapping_method"] = "protein_lifted"
    df["_source"] = "vampseq"
    df["source_file"] = filepath.name

    logger.info(
        "  Loaded %d rows; %d with gene; %d with functional score; types: %s",
        len(df),
        (df["gene"] != "UNKNOWN").sum(),
        df["functional_score_vampseq"].notna().sum(),
        df["variant_type_harmonized"].value_counts().to_dict(),
    )
    return df
