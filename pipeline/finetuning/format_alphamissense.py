"""Format finetuning data for AlphaMissense-based classifiers.

Since AlphaMissense has no public weights, this formats data for training
a lightweight downstream classifier on top of AlphaMissense precomputed
features + variant metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_AM_COLUMNS = [
    "variant_id",
    "gene",
    "aa_pos",
    "aa_ref",
    "aa_alt",
    "hgvs_p",
    "variant_type_harmonized",
    "alphamissense_score",
    "alphamissense_label",
    "revel_score",
    "spliceai_max_ds",
    "label_continuous",
    "label_categorical",
    "split",
]


def format_am_finetuning(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Export AlphaMissense-formatted finetuning data to parquet files.

    Includes AlphaMissense precomputed features alongside other predictor
    scores (REVEL, SpliceAI) and functional labels for training downstream
    classifiers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        # Filter: need at least a functional label
        mask = split_df["consensus_functional_score"].notna()
        filtered = split_df[mask].copy()

        if filtered.empty:
            logger.warning("AM finetuning: no eligible rows for %s split", split_name)
            continue

        # Rename label columns
        filtered["label_continuous"] = filtered["consensus_functional_score"]
        filtered["label_categorical"] = filtered["consensus_functional_label"]

        # Select and export
        available_cols = [c for c in _AM_COLUMNS if c in filtered.columns]
        out = filtered[available_cols].reset_index(drop=True)

        out_path = output_dir / f"{split_name}.parquet"
        out.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("AM finetuning: %s → %d rows → %s", split_name, len(out), out_path.name)

    logger.info("AlphaMissense finetuning export complete → %s", output_dir)
