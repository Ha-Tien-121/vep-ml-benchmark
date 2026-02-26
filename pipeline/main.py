"""
Run the pipeline with the following steps:
    - load_fisseq_supp
    - load_labelseq
    - load_pillar
    - merge_all
    - export_dataframe
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from pipeline.config import DATA_DIR, FISSEQ_SUPP_FILES, LABELSEQ_FILES, OUTPUT_DIR, PILLAR_FILES
from pipeline.loaders.load_fisseq import load_fisseq_supp
from pipeline.loaders.load_labelseq import load_labelseq
from pipeline.loaders.load_pillar import load_pillar
from pipeline.mergers.merge_all import merge_all
from pipeline.outputs.export import export_dataframe

logger = logging.getLogger("pipeline")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)


def run_pipeline(data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Run loader -> merge -> export using only selected functions."""
    logger.info("=" * 70)
    logger.info("STEP 1: Loading Pillar/FisSEQ/LabelSEQ")
    logger.info("=" * 70)

    # Pillar backbone
    pillar_frames: list[pd.DataFrame] = []
    for fname in PILLAR_FILES:
        path = data_dir / fname
        if path.exists():
            pillar_frames.append(load_pillar(path))
        else:
            logger.warning("Pillar file missing: %s", path)
    if not pillar_frames:
        raise FileNotFoundError("No Pillar files found; cannot run merge test.")
    pillar = pd.concat(pillar_frames, ignore_index=True)

    # FisSEQ supplementary tables
    fisseq_frames: list[pd.DataFrame] = []
    for fname in FISSEQ_SUPP_FILES:
        path = data_dir / fname
        if path.exists():
            fisseq_frames.append(load_fisseq_supp(path))
        else:
            logger.warning("FisSEQ file missing: %s", path)

    # LabelSEQ tables
    labelseq_frames: list[pd.DataFrame] = []
    for fname in LABELSEQ_FILES:
        path = data_dir / fname
        if path.exists():
            labelseq_frames.append(load_labelseq(path))
        else:
            logger.warning("LabelSEQ file missing: %s", path)

    logger.info("=" * 70)
    logger.info("STEP 2: Merging datasets")
    logger.info("=" * 70)
    master = merge_all(
        pillar=pillar,
        labelseq_frames=labelseq_frames,
        fisseq_frames=fisseq_frames,
        vampseq_frames=None,
    )

    logger.info("=" * 70)
    logger.info("STEP 3: Exporting")
    logger.info("=" * 70)
    export_dataframe(master, output_dir=output_dir)

    logger.info("Minimal functionality test complete. Rows=%d Cols=%d", *master.shape)
    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal runner: loaders + merge + export")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Input data directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)
    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
