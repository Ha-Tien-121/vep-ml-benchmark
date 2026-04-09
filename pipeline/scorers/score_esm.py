"""ESM scorer — protein variant effect prediction via log-likelihood ratios.

Uses ESM C 300M by default (laptop-friendly, ~600 MB, runs on CPU).
For each variant, computes:

    LLR = log P(alt_aa | context) - log P(wt_aa | context)

A more negative LLR indicates a more damaging substitution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import ESM_DEFAULT_MODEL, SCORER_CACHE_DIR
from pipeline.scorers.base import check_device, load_parquet_cache, save_parquet_cache

logger = logging.getLogger(__name__)

# Mapping from 1-letter amino acid codes to ESM token indices.
# Populated lazily once the tokenizer is loaded.
_AA_TO_TOKEN: dict[str, int] = {}


def _load_model(model_name: str, device: str):
    """Load an ESM model and return (model, tokenizer).

    Supports:
    - ``esmc_300m``: ESM C 300M (default, laptop-friendly)
    - ``esmc_600m``: ESM C 600M (better accuracy, more memory)
    - ``esm3_small``: ESM3-small-open (1.4B params, needs 16 GB VRAM)
    """
    import torch

    if model_name.startswith("esmc"):
        from esm.models.esmc import ESMC

        logger.info("Loading ESM C model: %s on %s", model_name, device)
        model = ESMC.from_pretrained(model_name).to(device).eval()
        return model, None  # ESMC uses its own tokenization

    # ESM3 path
    from esm.models.esm3 import ESM3

    logger.info("Loading ESM3 model: %s on %s", model_name, device)
    model = ESM3.from_pretrained(model_name).to(device).eval()
    return model, None


def _get_protein_sequence(df: pd.DataFrame, gene: str) -> str | None:
    """Get the wild-type protein sequence for a gene from the dataframe."""
    gene_rows = df[df["gene"] == gene]
    if "sequence_protein" in gene_rows.columns:
        seqs = gene_rows["sequence_protein"].dropna().unique()
        if len(seqs) > 0:
            return str(seqs[0])
    return None


def _fetch_protein_sequence(gene: str) -> str | None:
    """Fetch protein sequence from Ensembl REST API as fallback."""
    from pipeline.config import CANONICAL_TRANSCRIPTS

    transcript = CANONICAL_TRANSCRIPTS.get(gene)
    if not transcript:
        return None

    # Strip version suffix for Ensembl lookup
    transcript_base = transcript.split(".")[0]

    try:
        import requests

        url = f"https://rest.ensembl.org/sequence/id/{transcript_base}"
        resp = requests.get(
            url,
            headers={"Content-Type": "application/json"},
            params={"type": "protein"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        seq = data.get("seq")
        if seq:
            logger.info("Fetched protein sequence for %s (%d aa)", gene, len(seq))
            return seq
    except Exception:
        logger.warning("Failed to fetch protein sequence for %s from Ensembl", gene)

    return None


def _score_variants_esmc(
    model,
    wt_sequence: str,
    variants: pd.DataFrame,
    device: str,
) -> dict[int, float]:
    """Score variants using ESM C masked marginal approach.

    One forward pass on the wild-type sequence gives per-position
    log-probabilities.  The LLR for each variant is extracted from that.
    """
    import torch
    from esm.sdk.api import ESMProtein
    from esm.tokenization import EsmSequenceTokenizer

    results: dict[int, float] = {}

    tokenizer = EsmSequenceTokenizer()

    # Tokenize WT sequence
    protein = ESMProtein(sequence=wt_sequence)
    input_ids = tokenizer.encode(wt_sequence, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        output = model(input_tensor)
        # output.sequence_logits: (1, seq_len, vocab_size)
        logits = output.sequence_logits[0]  # (seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

    # Build AA → token index mapping
    aa_to_idx = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id is not None:
            aa_to_idx[aa] = token_id

    for idx, row in variants.iterrows():
        aa_pos = row.get("aa_pos")
        aa_ref = str(row.get("aa_ref", ""))
        aa_alt = str(row.get("aa_alt", ""))

        if pd.isna(aa_pos) or aa_ref not in aa_to_idx or aa_alt not in aa_to_idx:
            continue

        # Position in the tokenized sequence (1-indexed aa_pos, +1 for BOS token)
        token_pos = int(aa_pos)  # BOS token shifts everything by 1, but aa_pos is 1-indexed

        if token_pos < 0 or token_pos >= log_probs.shape[0]:
            continue

        ref_logprob = log_probs[token_pos, aa_to_idx[aa_ref]].item()
        alt_logprob = log_probs[token_pos, aa_to_idx[aa_alt]].item()
        llr = alt_logprob - ref_logprob
        results[idx] = llr

    return results


def score_esm(
    df: pd.DataFrame,
    model_name: str = ESM_DEFAULT_MODEL,
    device: str | None = None,
) -> pd.DataFrame:
    """Fill ``esm3_score`` and ``esm3_rank`` using ESM log-likelihood ratios.

    Rows that already have a non-null ``esm3_score`` are skipped.
    """
    if "esm3_score" not in df.columns:
        df["esm3_score"] = pd.NA
    if "esm3_rank" not in df.columns:
        df["esm3_rank"] = pd.NA

    needs_score = df["esm3_score"].isna()
    n_need = needs_score.sum()
    if n_need == 0:
        logger.info("ESM: all rows already scored — skipping")
        return df

    logger.info("ESM: %d / %d rows need scoring", n_need, len(df))

    # Check cache
    cache_path = SCORER_CACHE_DIR / f"esm_scores_{model_name}.parquet"
    cached = load_parquet_cache(cache_path)
    if cached is not None and "variant_id" in cached.columns and "variant_id" in df.columns:
        cache_map = dict(zip(cached["variant_id"], cached["esm3_score"]))
        for idx in df.index[needs_score]:
            vid = df.at[idx, "variant_id"]
            if vid in cache_map:
                df.at[idx, "esm3_score"] = cache_map[vid]
        needs_score = df["esm3_score"].isna()
        n_need = needs_score.sum()
        logger.info("ESM: %d rows remain after cache lookup", n_need)
        if n_need == 0:
            df["esm3_rank"] = df["esm3_score"].rank(pct=True)
            return df

    # Load model
    if device is None:
        device = check_device()

    try:
        model, _ = _load_model(model_name, device)
    except Exception:
        logger.exception("ESM: failed to load model %s — skipping scorer", model_name)
        return df

    # Score by gene (one WT forward pass per gene)
    genes = df.loc[needs_score, "gene"].dropna().unique()
    total_scored = 0

    for gene in genes:
        gene_mask = needs_score & (df["gene"] == gene)
        gene_variants = df.loc[gene_mask]

        if len(gene_variants) == 0:
            continue

        # Get WT protein sequence
        wt_seq = _get_protein_sequence(df, gene)
        if wt_seq is None:
            wt_seq = _fetch_protein_sequence(gene)
        if wt_seq is None:
            logger.warning("ESM: no protein sequence for %s — skipping %d variants", gene, len(gene_variants))
            continue

        logger.info("ESM: scoring %d variants for %s (%d aa)", len(gene_variants), gene, len(wt_seq))

        try:
            if model_name.startswith("esmc"):
                scores = _score_variants_esmc(model, wt_seq, gene_variants, device)
            else:
                # ESM3 scoring would go here — same approach, different API
                logger.warning("ESM3 scoring not yet implemented — use esmc_300m or esmc_600m")
                scores = {}
        except Exception:
            logger.exception("ESM: scoring failed for %s", gene)
            scores = {}

        for idx, llr in scores.items():
            df.at[idx, "esm3_score"] = llr
        total_scored += len(scores)
        logger.info("ESM: scored %d / %d variants for %s", len(scores), len(gene_variants), gene)

    # Compute percentile ranks
    scored_mask = df["esm3_score"].notna()
    if scored_mask.any():
        df.loc[scored_mask, "esm3_rank"] = df.loc[scored_mask, "esm3_score"].rank(pct=True)

    # Save cache
    scored_df = df.loc[scored_mask, ["variant_id", "esm3_score"]].dropna(subset=["variant_id"])
    if not scored_df.empty:
        save_parquet_cache(scored_df, cache_path)

    total = df["esm3_score"].notna().sum()
    logger.info(
        "ESM: scoring complete — %d / %d rows have scores (%.1f%%)",
        total, len(df), total / len(df) * 100,
    )
    return df
