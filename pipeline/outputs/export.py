"""Export the final dataframe and generate summary statistics."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pipeline.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def export_dataframe(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Save the master dataframe as Parquet and CSV, plus a text summary."""
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = out / "benchmark_dataframe.parquet"
    csv_path = out / "benchmark_dataframe.csv"
    summary_path = out / "summary_statistics.txt"

    # Save Parquet 
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Saved %s (%d rows x %d cols)", parquet_path.name, *df.shape)

    # Save CSV
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path.name)

    # Summary statistics
    lines = _build_summary(df)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", summary_path.name)

    # Print to stdout as well
    print("\n" + "\n".join(lines))


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
