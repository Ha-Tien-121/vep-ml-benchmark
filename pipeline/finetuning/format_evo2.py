"""Format finetuning data for Evo2 DNA foundation models.

Produces parquet files with genomic coordinates, DNA context, and functional
score labels suitable for supervised finetuning of Evo2 models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_EVO2_COLUMNS = [
    "variant_id",
    "gene",
    "chrom",
    "hg38_pos",
    "ref_allele",
    "alt_allele",
    "genomic_coord",
    "sequence_dna",
    "variant_type_harmonized",
    "label_continuous",
    "label_categorical",
    "split",
]


def format_evo2_finetuning(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Export Evo2-formatted finetuning data to parquet files.

    Filters to variants with genomic coordinates and functional labels.
    DNA sequences are included when available in the ``sequence_dna`` column.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        # Filter: need genomic coordinates and a functional label
        mask = (
            split_df["hg38_pos"].notna()
            & split_df["ref_allele"].notna()
            & split_df["alt_allele"].notna()
            & split_df["chrom"].notna()
            & split_df["consensus_functional_score"].notna()
        )
        filtered = split_df[mask].copy()

        if filtered.empty:
            logger.warning("Evo2 finetuning: no eligible rows for %s split", split_name)
            continue

        # Rename label columns
        filtered["label_continuous"] = filtered["consensus_functional_score"]
        filtered["label_categorical"] = filtered["consensus_functional_label"]

        # Select and export
        available_cols = [c for c in _EVO2_COLUMNS if c in filtered.columns]
        out = filtered[available_cols].reset_index(drop=True)

        out_path = output_dir / f"{split_name}.parquet"
        out.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Evo2 finetuning: %s → %d rows → %s", split_name, len(out), out_path.name)

    logger.info("Evo2 finetuning export complete → %s", output_dir)
