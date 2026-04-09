"""Attach protein and DNA sequences to each variant row.

Uses the gene metadata cache (populated by ``ensure_gene_metadata``) to
look up canonical protein and CDS sequences.  Each row in the same gene
gets the same wild-type sequence.
"""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.config import MSA_DIR
from pipeline.utils.gene_metadata import get_all_metadata

logger = logging.getLogger(__name__)


def add_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Populate sequence columns from gene metadata cache.

    Fills: ``sequence_protein``, ``sequence_id_protein``,
    ``sequence_dna``, ``sequence_id_dna``, ``sequence_type``,
    ``msa_filepath``, ``has_protein_seq``, ``has_dna_seq``, ``has_msa``.
    """
    for col in [
        "sequence_protein", "sequence_id_protein",
        "sequence_dna", "sequence_id_dna",
        "sequence_type", "msa_filepath",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    genes = df["gene"].dropna().unique()
    genes_with_protein = 0
    genes_with_dna = 0

    for gene in genes:
        meta = get_all_metadata(gene)
        mask = df["gene"] == gene

        prot_seq = meta.get("protein_sequence")
        if prot_seq:
            df.loc[mask & df["sequence_protein"].isna(), "sequence_protein"] = prot_seq
            df.loc[mask & df["sequence_id_protein"].isna(), "sequence_id_protein"] = meta.get("protein_seq_id", gene)
            genes_with_protein += 1

        dna_seq = meta.get("dna_sequence")
        if dna_seq:
            df.loc[mask & df["sequence_dna"].isna(), "sequence_dna"] = dna_seq
            df.loc[mask & df["sequence_id_dna"].isna(), "sequence_id_dna"] = meta.get("dna_seq_id", gene)
            genes_with_dna += 1

        df.loc[mask, "sequence_type"] = "protein"

        # Check for MSA file
        for ext in (".a3m", ".sto", ".aln"):
            msa_path = MSA_DIR / f"{gene}{ext}"
            if msa_path.exists():
                df.loc[mask & df["msa_filepath"].isna(), "msa_filepath"] = str(msa_path)
                break

    # Boolean flags
    df["has_protein_seq"] = df["sequence_protein"].notna()
    df["has_dna_seq"] = df["sequence_dna"].notna()
    df["has_msa"] = df["msa_filepath"].notna()

    logger.info(
        "Sequences: %d / %d genes have protein seq, %d / %d have DNA seq",
        genes_with_protein, len(genes), genes_with_dna, len(genes),
    )
    logger.info(
        "  Rows with protein: %d, DNA: %d, MSA: %d",
        df["has_protein_seq"].sum(), df["has_dna_seq"].sum(), df["has_msa"].sum(),
    )
    return df
