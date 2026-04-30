"""Export the final dataframe and generate summary statistics."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pipeline.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def export_dataframe(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Save the master dataframe as Parquet and CSV, plus a text summary.

    Also produces two purpose-built outputs:
      - protein_modeling_dataframe: AA-level features for protein ML models
      - dna_modeling_dataframe:     genomic features for DNA/sequence ML models
    """
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = out / "benchmark_dataframe.parquet"
    csv_path = out / "benchmark_dataframe.csv"
    summary_path = out / "summary_statistics.txt"

    # Coerce mixed-type columns so pyarrow doesn't choke
    df = _sanitize_for_parquet(df)

    # Save master Parquet
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Saved %s (%d rows x %d cols)", parquet_path.name, *df.shape)

    # Save master CSV
    try:
        df.to_csv(csv_path, index=False)
        logger.info("Saved %s", csv_path.name)
    except PermissionError:
        logger.warning(
            "Cannot write %s (file may be open in another program). "
            "Parquet output is still available.", csv_path.name,
        )

    # Protein modeling dataframe
    _export_subset(df, _protein_cols(df), out, "protein_modeling_dataframe")

    # DNA modeling dataframe
    _export_subset(df, _dna_cols(df), out, "dna_modeling_dataframe")

    # Summary statistics
    lines = _build_summary(df)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", summary_path.name)

    # Print to stdout as well
    print("\n" + "\n".join(lines))


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Fix mixed-type object columns that pyarrow cannot serialize."""
    df = df.copy()

    # Deduplicate column names (keep first occurrence)
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        logger.warning("Dropping %d duplicate column names: %s", len(dupes), dupes[:10])
        df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            numeric = pd.to_numeric(series, errors="coerce")
            non_null_orig = series.notna().sum()
            non_null_num = numeric.notna().sum()
            if non_null_num >= non_null_orig * 0.5 and non_null_num > 0:
                df[col] = numeric
            else:
                df[col] = series.astype(str).replace("nan", pd.NA).replace("None", pd.NA)
    return df


def _export_subset(
    df: pd.DataFrame,
    cols: list[str],
    out: Path,
    name: str,
) -> None:
    """Save a column-subset of the master df as parquet + CSV."""
    subset = df[[c for c in cols if c in df.columns]].copy()
    parquet_path = out / f"{name}.parquet"
    csv_path = out / f"{name}.csv"
    subset.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Saved %s (%d rows x %d cols)", parquet_path.name, *subset.shape)
    try:
        subset.to_csv(csv_path, index=False)
        logger.info("Saved %s", csv_path.name)
    except PermissionError:
        logger.warning("Cannot write %s (file open elsewhere).", csv_path.name)


def _identity_cols() -> list[str]:
    """Columns shared by both protein and DNA modeling dataframes."""
    return [
        "gene", "variant_id",
        "aa_ref", "aa_pos", "aa_alt",
        "hgvs_p", "protein_substitution", "protein_substitution_long",
        "variant_type_harmonized", "source_datasets",
        "consensus_functional_score", "consensus_functional_label",
        "conflict_flag", "low_coverage_flag", "is_duplicate_flag",
    ]


def _protein_cols(df: pd.DataFrame) -> list[str]:
    """Column list for the protein modeling dataframe.

    Priority order puts identity/score/sequence columns first, then appends
    every remaining column from the dataframe so no source data is dropped.
    """
    cols = _identity_cols()

    # All functional scores (raw + named variants)
    cols += [c for c in df.columns if c.startswith("functional_score_")]

    # Normalized scores
    cols += [c for c in df.columns if c.startswith("normalized_score_")]

    # Protein-level model predictors
    cols += ["alphamissense_score", "alphamissense_label", "revel_score"]

    # Protein sequence
    cols += ["sequence_protein", "sequence_id_protein", "has_protein_seq",
             "msa_filepath", "has_msa"]

    # All source supplementary cols (labelseq_, fisseq_, vampseq_)
    cols += [c for c in df.columns if c.startswith("labelseq_")]
    cols += [c for c in df.columns if c.startswith("fisseq_")]
    cols += [c for c in df.columns if c.startswith("vampseq_")]

    # Append every remaining column so no source data is dropped
    cols += list(df.columns)

    return _dedup(cols)


def _dna_cols(df: pd.DataFrame) -> list[str]:
    """Column list for the DNA modeling dataframe.

    Priority order puts identity/coordinate/sequence columns first, then appends
    every remaining column from the dataframe so no source data is dropped.
    """
    cols = _identity_cols()

    # Genomic coordinates and transcript info
    cols += [
        "chrom", "hg38_pos", "ref_allele", "alt_allele", "genomic_coord",
        "hgvs_c", "hg19_pos", "strand",
        "ensembl_transcript_id", "refseq_transcript_id",
        "simplified_consequence", "consequence",
        "transcript_pos", "transcript_ref", "transcript_alt",
        "splice_measure", "gnomad_maf", "nucleotide_or_aa",
        "coord_mapping_method",
    ]

    # Pillar primary scores
    cols += ["functional_score_pillar", "functional_score_pillar_rep",
             "normalized_score_pillar"]

    # SpliceAI scores (precomputed + from Pillar)
    cols += [
        "spliceai_DS_AG", "spliceai_DS_AL", "spliceai_DS_DG", "spliceai_DS_DL",
        "spliceai_max_delta_score", "spliceai_max_ds",
        "pillar__spliceAI_DS_AG", "pillar__spliceAI_DS_AL",
        "pillar__spliceAI_DS_DG", "pillar__spliceAI_DS_DL",
        "pillar__spliceAI_DP_AG", "pillar__spliceAI_DP_AL",
        "pillar__spliceAI_DP_DG", "pillar__spliceAI_DP_DL",
    ]

    # DNA-level model predictors
    cols += ["esm3_score", "esm3_rank", "evo2_score", "evo2_rank"]

    # DNA sequence
    cols += ["sequence_dna", "sequence_id_dna", "has_dna_seq"]

    # Pillar supplementary (ClinVar, gnomAD, MutPred2, ClinGen, etc.)
    cols += [c for c in df.columns if c.startswith("pillar__")]
    cols += [
        "pillar_dataset_id", "pillar_flag", "functional_class_pillar",
        "mavedb_urn", "hgnc_id",
    ]

    # Append every remaining column so no source data is dropped
    cols += list(df.columns)

    return _dedup(cols)


def _dedup(cols: list[str]) -> list[str]:
    """Return cols with duplicates removed, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _build_summary(df: pd.DataFrame) -> list[str]:
    """Build a human-readable summary of the master dataframe."""
    lines: list[str] = [
        "=" * 70,
        "  VEP BENCHMARK DATAFRAME -- SUMMARY STATISTICS",
        "=" * 70,
        "",
    ]

    # Total
    lines.append(f"Total variants:  {len(df)}")

    # Per-gene breakdown
    if "gene" in df.columns:
        gene_counts = df["gene"].value_counts(dropna=False)
        lines.append(f"\nPer-gene breakdown ({len(gene_counts)} genes):")
        for gene, count in gene_counts.items():
            label = str(gene) if pd.notna(gene) else "<missing>"
            lines.append(f"  {label:20s}  {count:>8d}")

    # Source dataset contribution
    if "source_datasets" in df.columns:
        lines.append("\nSource dataset contributions:")
        all_sources: dict[str, int] = {}
        for val in df["source_datasets"].dropna():
            for s in str(val).split("|"):
                s = s.strip()
                all_sources[s] = all_sources.get(s, 0) + 1
        for src, count in sorted(all_sources.items(), key=lambda x: -x[1]):
            lines.append(f"  {src:30s}  {count:>8d}")

    # Coordinate mapping method
    if "coord_mapping_method" in df.columns:
        lines.append("\nCoordinate mapping method:")
        for method, count in df["coord_mapping_method"].value_counts(dropna=False).items():
            lines.append(f"  {str(method):20s}  {count:>8d}")

    # Sequence availability
    for col, label in [
        ("has_protein_seq", "With protein sequence"),
        ("has_dna_seq", "With DNA sequence"),
        ("has_msa", "With MSA"),
    ]:
        if col in df.columns:
            n = df[col].sum()
            lines.append(f"\n{label}: {n} / {len(df)} ({n / len(df) * 100:.1f}%)")

    # Predictor coverage
    lines.append("\nPredictor coverage:")
    for col, label in [
        ("alphamissense_score", "AlphaMissense"),
        ("esm3_score", "ESM3"),
        ("evo2_score", "Evo2"),
    ]:
        if col in df.columns:
            n = df[col].notna().sum()
            lines.append(f"  {label:20s}  {n:>8d} / {len(df)} ({n / len(df) * 100:.1f}%)")

    # QC flags
    lines.append("\nQC flags:")
    for col, label in [
        ("conflict_flag", "Conflict flag"),
        ("is_duplicate_flag", "Duplicate flag"),
        ("low_coverage_flag", "Low coverage flag"),
    ]:
        if col in df.columns:
            n = df[col].sum()
            lines.append(f"  {label:20s}  {n:>8d}")

    # Consensus score distribution
    if "consensus_functional_score" in df.columns:
        scores = df["consensus_functional_score"].dropna()
        if len(scores):
            lines.append(f"\nConsensus functional score (n={len(scores)}):")
            lines.append(f"  Mean:   {scores.mean():.4f}")
            lines.append(f"  Std:    {scores.std():.4f}")
            lines.append(f"  Min:    {scores.min():.4f}")
            lines.append(f"  25%:    {scores.quantile(0.25):.4f}")
            lines.append(f"  50%:    {scores.quantile(0.50):.4f}")
            lines.append(f"  75%:    {scores.quantile(0.75):.4f}")
            lines.append(f"  Max:    {scores.max():.4f}")

    # Consensus label distribution
    if "consensus_functional_label" in df.columns:
        lines.append("\nConsensus functional label:")
        for label, count in df["consensus_functional_label"].value_counts(dropna=False).items():
            lines.append(f"  {str(label):25s}  {count:>8d}")

    # Genes in final dataframe
    if "gene" in df.columns:
        genes = sorted(df["gene"].dropna().unique())
        lines.append(f"\nGenes in final dataframe ({len(genes)}): {genes}")

    lines.append("")
    lines.append("=" * 70)
    return lines
