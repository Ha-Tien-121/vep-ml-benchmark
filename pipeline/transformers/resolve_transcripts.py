"""Back-fill missing transcript IDs from gene metadata cache."""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.utils.gene_metadata import get_canonical_transcript, get_refseq_id

logger = logging.getLogger(__name__)


def resolve_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    """Fill ``ensembl_transcript_id`` and ``refseq_transcript_id`` from gene metadata.

    Only fills where the value is currently null; never overwrites existing data.
    """
    if "ensembl_transcript_id" not in df.columns:
        df["ensembl_transcript_id"] = pd.NA
    if "refseq_transcript_id" not in df.columns:
        df["refseq_transcript_id"] = pd.NA

    genes = df["gene"].dropna().unique()
    filled_ens = 0
    filled_ref = 0

    for gene in genes:
        gene_mask = df["gene"] == gene

        # Ensembl transcript
        missing_ens = gene_mask & df["ensembl_transcript_id"].isna()
        if missing_ens.any():
            tx = get_canonical_transcript(gene)
            if tx:
                df.loc[missing_ens, "ensembl_transcript_id"] = tx
                filled_ens += missing_ens.sum()

        # RefSeq transcript
        missing_ref = gene_mask & df["refseq_transcript_id"].isna()
        if missing_ref.any():
            refseq = get_refseq_id(gene)
            if refseq:
                df.loc[missing_ref, "refseq_transcript_id"] = refseq
                filled_ref += missing_ref.sum()

    logger.info(
        "Transcript resolution: filled %d Ensembl IDs, %d RefSeq IDs across %d genes",
        filled_ens, filled_ref, len(genes),
    )
    return df
