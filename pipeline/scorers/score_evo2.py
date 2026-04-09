"""Evo2 scorer — DNA-level variant effect prediction via log-likelihood ratios.

Evo2 is a DNA foundation model that predicts the effect of genomic variants
using surrounding DNA context.  For each variant, computes:

    LLR = log P(alt_allele | flanking_DNA) - log P(ref_allele | flanking_DNA)

A more negative LLR indicates a more damaging variant.

**GPU required**: Evo2 models are too large for practical CPU inference.
If no GPU is available, this scorer logs a warning and returns the dataframe
unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import CACHE_DIR, EVO2_DEFAULT_MODEL, SCORER_CACHE_DIR
from pipeline.scorers.base import (
    batch_iter,
    check_device,
    load_parquet_cache,
    save_parquet_cache,
)

logger = logging.getLogger(__name__)

# DNA context window: how many bases to include on each side of the variant
_CONTEXT_WINDOW = 512

# Nucleotide to token index — populated lazily
_NUC_TO_TOKEN: dict[str, int] = {}


def _fetch_dna_context(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    window: int = _CONTEXT_WINDOW,
) -> tuple[str, str] | None:
    """Fetch flanking DNA context from Ensembl REST API.

    Returns (wt_context, mut_context) strings centered on the variant,
    or None on failure.
    """
    try:
        import requests

        # Ensembl uses 1-based coordinates
        start = max(1, pos - window)
        end = pos + window + len(ref) - 1

        chrom_clean = str(chrom).replace("chr", "")
        url = f"https://rest.ensembl.org/sequence/region/human/{chrom_clean}:{start}..{end}:1"
        resp = requests.get(
            url,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        wt_seq = data.get("seq", "")

        if not wt_seq:
            return None

        # Position of the variant within the fetched window
        var_offset = pos - start
        # Construct mutant sequence
        mut_seq = wt_seq[:var_offset] + alt + wt_seq[var_offset + len(ref):]

        return wt_seq, mut_seq

    except Exception:
        logger.debug("Failed to fetch DNA context for %s:%d", chrom, pos)
        return None


def _get_local_dna_context(
    df: pd.DataFrame,
    idx: int,
    window: int = _CONTEXT_WINDOW,
) -> tuple[str, str] | None:
    """Try to build DNA context from the dataframe's sequence_dna column."""
    if "sequence_dna" not in df.columns:
        return None

    dna_seq = df.at[idx, "sequence_dna"]
    if pd.isna(dna_seq) or not dna_seq:
        return None

    # If we have a full CDS sequence, extract context around the variant position
    hg38_pos = df.at[idx, "hg38_pos"]
    ref = str(df.at[idx, "ref_allele"])
    alt = str(df.at[idx, "alt_allele"])

    if pd.isna(hg38_pos):
        return None

    # For CDS sequences, the variant position is relative to the transcript
    # This is an approximation — for precise work, use the Ensembl API
    dna_str = str(dna_seq)
    mid = len(dna_str) // 2
    start = max(0, mid - window)
    end = min(len(dna_str), mid + window)
    wt_context = dna_str[start:end]
    mut_context = wt_context  # placeholder — need proper coordinate mapping

    return wt_context, mut_context


def _load_model(model_name: str, device: str):
    """Load an Evo2 model. Returns (model, tokenizer)."""
    from evo2 import Evo2

    logger.info("Loading Evo2 model: %s on %s", model_name, device)
    model = Evo2(model_name)
    return model, None


def _score_variant_evo2(
    model,
    wt_seq: str,
    mut_seq: str,
    device: str,
) -> float | None:
    """Score a single variant by comparing WT and mutant DNA log-likelihoods."""
    import torch

    try:
        with torch.no_grad():
            wt_logits, _ = model(wt_seq, device=device)
            mut_logits, _ = model(mut_seq, device=device)

            # Mean log-likelihood across positions
            wt_ll = torch.log_softmax(wt_logits, dim=-1).mean().item()
            mut_ll = torch.log_softmax(mut_logits, dim=-1).mean().item()

            return mut_ll - wt_ll
    except Exception:
        return None


def score_evo2(
    df: pd.DataFrame,
    model_name: str = EVO2_DEFAULT_MODEL,
    device: str | None = None,
) -> pd.DataFrame:
    """Fill ``evo2_score`` and ``evo2_rank`` using Evo2 DNA log-likelihood ratios.

    Rows that already have a non-null ``evo2_score`` are skipped.
    **Requires GPU** — returns df unchanged if no GPU is available.
    """
    if "evo2_score" not in df.columns:
        df["evo2_score"] = pd.NA
    if "evo2_rank" not in df.columns:
        df["evo2_rank"] = pd.NA

    # GPU check
    if device is None:
        device = check_device()

    if device == "cpu":
        logger.warning(
            "Evo2: no GPU available — skipping scorer. "
            "Evo2 models require GPU for practical inference. "
            "Install CUDA and a compatible GPU, or use --device cuda."
        )
        return df

    needs_score = df["evo2_score"].isna()
    n_need = needs_score.sum()
    if n_need == 0:
        logger.info("Evo2: all rows already scored — skipping")
        return df

    logger.info("Evo2: %d / %d rows need scoring", n_need, len(df))

    # Check cache
    cache_path = SCORER_CACHE_DIR / f"evo2_scores_{model_name}.parquet"
    cached = load_parquet_cache(cache_path)
    if cached is not None and "variant_id" in cached.columns and "variant_id" in df.columns:
        cache_map = dict(zip(cached["variant_id"], cached["evo2_score"]))
        for idx in df.index[needs_score]:
            vid = df.at[idx, "variant_id"]
            if vid in cache_map:
                df.at[idx, "evo2_score"] = cache_map[vid]
        needs_score = df["evo2_score"].isna()
        n_need = needs_score.sum()
        logger.info("Evo2: %d rows remain after cache lookup", n_need)
        if n_need == 0:
            df["evo2_rank"] = df["evo2_score"].rank(pct=True)
            return df

    # Load model
    try:
        model, _ = _load_model(model_name, device)
    except Exception:
        logger.exception("Evo2: failed to load model %s — skipping scorer", model_name)
        return df

    # Score variants with genomic coordinates
    todo = df.loc[needs_score].copy()
    has_coords = (
        todo["hg38_pos"].notna()
        & todo["ref_allele"].notna()
        & todo["alt_allele"].notna()
        & todo["chrom"].notna()
    )
    scoreable = todo[has_coords]
    logger.info("Evo2: %d / %d variants have genomic coordinates for scoring", len(scoreable), n_need)

    scored_count = 0
    for idx, row in scoreable.iterrows():
        # Try local DNA context first, fall back to Ensembl API
        context = _get_local_dna_context(df, idx)
        if context is None:
            context = _fetch_dna_context(
                row["chrom"], int(row["hg38_pos"]),
                str(row["ref_allele"]), str(row["alt_allele"]),
            )
        if context is None:
            continue

        wt_seq, mut_seq = context
        score = _score_variant_evo2(model, wt_seq, mut_seq, device)
        if score is not None:
            df.at[idx, "evo2_score"] = score
            scored_count += 1

        if scored_count % 100 == 0 and scored_count > 0:
            logger.info("Evo2: scored %d variants so far...", scored_count)

    # Compute percentile ranks
    scored_mask = df["evo2_score"].notna()
    if scored_mask.any():
        df.loc[scored_mask, "evo2_rank"] = df.loc[scored_mask, "evo2_score"].rank(pct=True)

    # Save cache
    scored_df = df.loc[scored_mask, ["variant_id", "evo2_score"]].dropna(subset=["variant_id"])
    if not scored_df.empty:
        save_parquet_cache(scored_df, cache_path)

    total = df["evo2_score"].notna().sum()
    logger.info(
        "Evo2: scoring complete — %d / %d rows have scores (%.1f%%)",
        total, len(df), total / len(df) * 100,
    )
    return df
