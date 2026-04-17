"""Input and output schema validation with clear error reporting.

Catches column name drift, missing critical fields, and data quality issues
early — before they propagate silently through the pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Per-source required columns (critical = pipeline fails without them)
_LOADER_REQUIRED: dict[str, dict[str, list[str]]] = {
    "pillar": {
        "critical": ["variant_id", "gene", "hg38_pos", "ref_allele", "alt_allele"],
        "expected": ["aa_pos", "aa_ref", "aa_alt", "hgvs_p", "functional_score_pillar",
                      "simplified_consequence", "alphamissense_score"],
    },
    "fisseq": {
        "critical": ["gene", "aa_ref", "aa_pos", "aa_alt", "functional_score_fisseq"],
        "expected": ["fisseq_auroc", "variant_type_harmonized"],
    },
    "labelseq": {
        "critical": ["aa_ref", "aa_pos", "aa_alt"],
        "expected": ["gene", "functional_score_labelseq", "variant_type_harmonized"],
    },
    "vampseq": {
        "critical": ["aa_ref", "aa_pos", "aa_alt", "functional_score_vampseq"],
        "expected": ["gene", "hgvs_p", "variant_type_harmonized"],
    },
}

_MERGED_REQUIRED = [
    "gene", "genomic_coord", "source_datasets",
]


def validate_loader_output(
    df: pd.DataFrame,
    source: str,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate a freshly-loaded dataframe against expected schema.

    Returns a list of warning/error strings.  If *strict*, raises
    ``ValueError`` on critical column mismatches.
    """
    issues: list[str] = []
    spec = _LOADER_REQUIRED.get(source)
    if not spec:
        logger.debug("No schema spec for source '%s' — skipping validation", source)
        return issues

    for col in spec["critical"]:
        if col not in df.columns:
            msg = f"[{source}] CRITICAL: missing column '{col}'"
            issues.append(msg)
            logger.error(msg)

    for col in spec.get("expected", []):
        if col not in df.columns:
            msg = f"[{source}] WARNING: missing expected column '{col}'"
            issues.append(msg)
            logger.warning(msg)

    # Data quality checks
    if "gene" in df.columns:
        n_null_gene = df["gene"].isna().sum()
        if n_null_gene > 0:
            issues.append(f"[{source}] {n_null_gene} rows have null gene")

    if len(df) == 0:
        issues.append(f"[{source}] dataframe is empty")

    if strict and any("CRITICAL" in i for i in issues):
        raise ValueError(
            f"Schema validation failed for {source}: "
            + "; ".join(i for i in issues if "CRITICAL" in i)
        )

    return issues


def validate_merged_output(df: pd.DataFrame) -> list[str]:
    """Validate the merged benchmark dataframe."""
    issues: list[str] = []

    for col in _MERGED_REQUIRED:
        if col not in df.columns:
            msg = f"[merged] CRITICAL: missing column '{col}'"
            issues.append(msg)
            logger.error(msg)

    # Check for duplicate genomic_coord (excluding NA)
    if "genomic_coord" in df.columns:
        coords = df["genomic_coord"].dropna()
        n_dup = coords.duplicated().sum()
        if n_dup > 0:
            issues.append(f"[merged] {n_dup} duplicate genomic_coord values")
            df["is_duplicate_flag"] = df.duplicated(subset=["genomic_coord"], keep=False) & df["genomic_coord"].notna()
        else:
            df["is_duplicate_flag"] = False

    # Coverage summary (informational, not errors)
    n_total = len(df)
    n_coord = df["genomic_coord"].notna().sum() if "genomic_coord" in df.columns else 0
    logger.info(
        "Merged validation: %d total rows, %d with genomic_coord (%.1f%%)",
        n_total, n_coord, n_coord / n_total * 100 if n_total else 0,
    )

    return issues
