"""Stratified train / val / test splitting for finetuning."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import FINETUNING_SEED, FINETUNING_SPLIT_RATIOS

logger = logging.getLogger(__name__)


def create_splits(
    df: pd.DataFrame,
    seed: int = FINETUNING_SEED,
) -> dict[str, pd.DataFrame]:
    """Split the benchmark dataframe into train / val / test sets.

    Only rows with a non-null ``consensus_functional_score`` are included
    (these are the labeled examples suitable for supervised finetuning).

    Stratification is by ``gene`` + ``consensus_functional_label`` to ensure
    each split has proportional representation of genes and functional classes.

    Returns ``{"train": df, "val": df, "test": df}`` with a ``split`` column
    added to each.
    """
    train_ratio = FINETUNING_SPLIT_RATIOS["train"]
    val_ratio = FINETUNING_SPLIT_RATIOS["val"]
    test_ratio = FINETUNING_SPLIT_RATIOS["test"]

    # Filter to labeled rows
    if "consensus_functional_score" not in df.columns:
        # Fall back to any available functional score
        score_cols = [c for c in df.columns if c.startswith("functional_score_")]
        if score_cols:
            logger.info("No consensus_functional_score — computing from %s", score_cols)
            df["consensus_functional_score"] = df[score_cols].mean(axis=1, skipna=True)
            df["consensus_functional_label"] = pd.cut(
                df["consensus_functional_score"],
                bins=[-float("inf"), 0.3, 0.7, float("inf")],
                labels=["loss_of_function", "intermediate", "functional"],
            )
        else:
            logger.warning("No functional score columns found — returning empty splits")
            empty = pd.DataFrame(columns=list(df.columns) + ["split"])
            return {"train": empty, "val": empty.copy(), "test": empty.copy()}

    labeled = df[df["consensus_functional_score"].notna()].copy()
    if labeled.empty:
        logger.warning("No rows with consensus_functional_score — returning empty splits")
        empty = pd.DataFrame(columns=list(df.columns) + ["split"])
        return {"train": empty, "val": empty.copy(), "test": empty.copy()}

    logger.info("Finetuning splits: %d labeled rows out of %d total", len(labeled), len(df))

    # Build stratification key
    labeled["_strat_key"] = (
        labeled["gene"].fillna("UNKNOWN").astype(str)
        + "_"
        + labeled["consensus_functional_label"].fillna("unlabeled").astype(str)
    )

    # Collapse rare strata (< 3 samples) into "other" to avoid split failures
    strat_counts = labeled["_strat_key"].value_counts()
    rare = strat_counts[strat_counts < 3].index
    if len(rare) > 0:
        labeled.loc[labeled["_strat_key"].isin(rare), "_strat_key"] = "other"
        logger.info("Collapsed %d rare strata into 'other'", len(rare))

    # First split: train vs held-out (val + test)
    held_out_ratio = val_ratio + test_ratio
    try:
        df_train, df_held = train_test_split(
            labeled,
            test_size=held_out_ratio,
            stratify=labeled["_strat_key"],
            random_state=seed,
        )
    except ValueError:
        # Stratification failed (too few samples) — fall back to random split
        logger.warning("Stratified split failed — falling back to random split")
        df_train, df_held = train_test_split(
            labeled, test_size=held_out_ratio, random_state=seed,
        )

    # Second split: val vs test (50/50 of held-out)
    val_fraction = val_ratio / held_out_ratio
    try:
        df_val, df_test = train_test_split(
            df_held,
            test_size=1.0 - val_fraction,
            stratify=df_held["_strat_key"],
            random_state=seed,
        )
    except ValueError:
        df_val, df_test = train_test_split(
            df_held, test_size=1.0 - val_fraction, random_state=seed,
        )

    # Add split label and clean up
    df_train = df_train.drop(columns=["_strat_key"]).copy()
    df_val = df_val.drop(columns=["_strat_key"]).copy()
    df_test = df_test.drop(columns=["_strat_key"]).copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    logger.info(
        "Splits: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(df_train), len(df_train) / len(labeled) * 100,
        len(df_val), len(df_val) / len(labeled) * 100,
        len(df_test), len(df_test) / len(labeled) * 100,
    )

    return {"train": df_train, "val": df_val, "test": df_test}
