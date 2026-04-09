"""Format finetuning data for ESM protein language models.

Produces parquet files with protein sequences, variant info, and functional
score labels suitable for supervised finetuning of ESM C / ESM3 models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ESM_COLUMNS = [
    "variant_id",
    "gene",
    "sequence_protein",
    "mutant_sequence",
    "aa_pos",
    "aa_ref",
    "aa_alt",
    "hgvs_p",
    "variant_type_harmonized",
    "label_continuous",
    "label_categorical",
    "split",
]


def _build_mutant_sequence(row: pd.Series) -> str | None:
    """Apply the amino acid substitution to the wild-type protein sequence."""
    seq = row.get("sequence_protein")
    pos = row.get("aa_pos")
    alt = row.get("aa_alt")
    ref = row.get("aa_ref")

    if pd.isna(seq) or pd.isna(pos) or pd.isna(alt):
        return None

    seq = str(seq)
    pos = int(pos)

    # aa_pos is 1-indexed
    idx = pos - 1
    if idx < 0 or idx >= len(seq):
        return None

    # Verify ref matches (optional sanity check)
    if ref and not pd.isna(ref) and seq[idx] != str(ref):
        logger.debug(
            "Ref mismatch at pos %d: expected %s, got %s",
            pos, ref, seq[idx],
        )

    return seq[:idx] + str(alt) + seq[idx + 1:]


def format_esm_finetuning(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Export ESM-formatted finetuning data to parquet files.

    Filters to missense variants with protein sequences and functional labels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        # Filter: need protein sequence, missense only, has label
        mask = (
            split_df["sequence_protein"].notna()
            & (split_df["variant_type_harmonized"] == "missense")
            & split_df["consensus_functional_score"].notna()
            & split_df["aa_pos"].notna()
            & split_df["aa_ref"].notna()
            & split_df["aa_alt"].notna()
        )
        filtered = split_df[mask].copy()

        if filtered.empty:
            logger.warning("ESM finetuning: no eligible rows for %s split", split_name)
            continue

        # Build mutant sequences
        filtered["mutant_sequence"] = filtered.apply(_build_mutant_sequence, axis=1)
        filtered = filtered[filtered["mutant_sequence"].notna()]

        # Rename label columns
        filtered["label_continuous"] = filtered["consensus_functional_score"]
        filtered["label_categorical"] = filtered["consensus_functional_label"]

        # Select and export
        available_cols = [c for c in _ESM_COLUMNS if c in filtered.columns]
        out = filtered[available_cols].reset_index(drop=True)

        out_path = output_dir / f"{split_name}.parquet"
        out.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("ESM finetuning: %s → %d rows → %s", split_name, len(out), out_path.name)

    logger.info("ESM finetuning export complete → %s", output_dir)
