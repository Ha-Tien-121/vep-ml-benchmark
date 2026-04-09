"""Shared utilities for scorer modules."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def check_device() -> str:
    """Return the best available PyTorch device string."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        logger.warning("PyTorch not installed — falling back to CPU")
    return "cpu"


def load_parquet_cache(path: Path) -> pd.DataFrame | None:
    """Load a parquet cache file, returning None if it doesn't exist."""
    if path.exists():
        logger.info("Loading cached scores from %s", path.name)
        return pd.read_parquet(path)
    return None


def save_parquet_cache(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe as a parquet cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Cached %d scores to %s", len(df), path.name)


def batch_iter(iterable: list, size: int) -> Iterator[list]:
    """Yield successive chunks of *size* from *iterable*."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]
