"""
VEP Benchmark Pipeline
======================

Automated, reproducible pipeline for variant effect prediction benchmarking.

Steps:
    1. Discover & load datasets (Pillar, FisSEQ, LabelSEQ)
    2. Validate loaded data
    3. Merge all sources (genomic-coord + protein-level fallback)
    4. Transform (resolve transcripts, harmonize scores, add sequences)
    5. Score (AlphaMissense, ESM, Evo2)
    6. Export benchmark dataframe + reproducibility manifest
    7. Generate finetuning splits (optional)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from pipeline.config import (
    AUTO_DISCOVER_DATASETS,
    DATA_DIR,
    FISSEQ_FEATURE_FILES,
    FISSEQ_SUPP_FILES,
    FINETUNING_OUTPUT_DIR,
    LABELSEQ_FILES,
    OUTPUT_DIR,
    PILLAR_FILES,
    VAMPSEQ_FILES,
)

logger = logging.getLogger("pipeline")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)


def _load_datasets_explicit(data_dir: Path):
    """Load datasets from hardcoded file lists in config.py."""
    from pipeline.loaders.load_fisseq import load_fisseq_supp
    from pipeline.loaders.load_fisseq import load_fisseq_features
    from pipeline.loaders.load_labelseq import load_labelseq
    from pipeline.loaders.load_pillar import load_pillar
    from pipeline.loaders.load_vampseq import load_vampseq

    pillar_frames: list[pd.DataFrame] = []
    for fname in PILLAR_FILES:
        path = data_dir / fname
        if path.exists():
            pillar_frames.append(load_pillar(path))
        else:
            logger.warning("Pillar file missing: %s", path)

    fisseq_frames: list[pd.DataFrame] = []
    for fname in FISSEQ_SUPP_FILES:
        path = data_dir / fname
        if path.exists():
            fisseq_frames.append(load_fisseq_supp(path))
        else:
            logger.warning("FisSEQ file missing: %s", path)

    fisseq_feature_frames: list[pd.DataFrame] = []
    for fname in FISSEQ_FEATURE_FILES:
        path = data_dir / fname
        if path.exists():
            fisseq_feature_frames.append(load_fisseq_features(path))
        else:
            logger.warning("FisSEQ feature file missing: %s", path)

    labelseq_frames: list[pd.DataFrame] = []
    for fname in LABELSEQ_FILES:
        path = data_dir / fname
        if path.exists():
            labelseq_frames.append(load_labelseq(path))
        else:
            logger.warning("LabelSEQ file missing: %s", path)

    vampseq_frames: list[pd.DataFrame] = []
    for fname in VAMPSEQ_FILES:
        path = data_dir / fname
        if path.exists():
            vampseq_frames.append(load_vampseq(path))
        else:
            logger.warning("VampSEQ file missing: %s", path)

    return pillar_frames, fisseq_frames, fisseq_feature_frames, labelseq_frames, vampseq_frames


def _load_datasets_discovered(data_dir: Path):
    """Load datasets via auto-discovery (column-signature detection)."""
    from pipeline.loaders.load_fisseq import load_fisseq_supp
    from pipeline.loaders.load_fisseq import load_fisseq_features
    from pipeline.loaders.load_labelseq import load_labelseq
    from pipeline.loaders.load_pillar import load_pillar
    from pipeline.loaders.load_vampseq import load_vampseq
    from pipeline.utils.discover import discover_datasets

    datasets = discover_datasets(data_dir)

    pillar_frames = [load_pillar(e.path) for e in datasets.pillar]
    fisseq_frames = [load_fisseq_supp(e.path) for e in datasets.fisseq_supp]
    fisseq_feature_frames = [load_fisseq_features(e.path) for e in datasets.fisseq_features]
    labelseq_frames = [load_labelseq(e.path) for e in datasets.labelseq]
    vampseq_frames = [
        load_vampseq(e.path, gene_override=e.gene) for e in datasets.vampseq
    ]

    return pillar_frames, fisseq_frames, fisseq_feature_frames, labelseq_frames, vampseq_frames


def run_pipeline(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    skip_scoring: bool = False,
    skip_finetuning: bool = False,
    skip_transforms: bool = False,
    scorers: list[str] | None = None,
    esm_model: str | None = None,
    evo2_model: str | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Run the full pipeline: load → validate → merge → transform → score → export → finetune."""

    # ── STEP 0: Reproducibility manifest ─────────────────────────────
    from pipeline.utils.manifest import generate_pipeline_manifest

    logger.info("=" * 70)
    logger.info("STEP 0: Generating reproducibility manifest")
    logger.info("=" * 70)
    generate_pipeline_manifest(data_dir, output_dir)

    # ── STEP 1: Load datasets ────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 1: Loading datasets")
    logger.info("=" * 70)

    if AUTO_DISCOVER_DATASETS:
        (
            pillar_frames,
            fisseq_frames,
            fisseq_feature_frames,
            labelseq_frames,
            vampseq_frames,
        ) = _load_datasets_discovered(data_dir)
    else:
        (
            pillar_frames,
            fisseq_frames,
            fisseq_feature_frames,
            labelseq_frames,
            vampseq_frames,
        ) = _load_datasets_explicit(data_dir)

    if not pillar_frames:
        raise FileNotFoundError("No Pillar files found — cannot build benchmark dataframe.")

    # ── STEP 2: Validate loaded data ─────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 2: Validating loaded data")
    logger.info("=" * 70)

    from pipeline.validators import validate_loader_output

    for df in pillar_frames:
        validate_loader_output(df, "pillar")
    for df in fisseq_frames:
        validate_loader_output(df, "fisseq")
    for df in fisseq_feature_frames:
        validate_loader_output(df, "fisseq_features")
    for df in labelseq_frames:
        validate_loader_output(df, "labelseq")
    for df in vampseq_frames:
        validate_loader_output(df, "vampseq")

    pillar = pd.concat(pillar_frames, ignore_index=True)

    # ── STEP 3: Merge all sources ────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 3: Merging datasets")
    logger.info("=" * 70)

    from pipeline.mergers.merge_all import merge_all
    from pipeline.validators import validate_merged_output

    master = merge_all(
        pillar=pillar,
        labelseq_frames=labelseq_frames,
        fisseq_frames=fisseq_frames,
        fisseq_feature_frames=fisseq_feature_frames or None,
        vampseq_frames=vampseq_frames or None,
    )
    validate_merged_output(master)

    # ── STEP 4: Transform ────────────────────────────────────────────
    if not skip_transforms:
        logger.info("=" * 70)
        logger.info("STEP 4: Transforming (transcripts, scores, sequences)")
        logger.info("=" * 70)

        from pipeline.utils.gene_metadata import ensure_gene_metadata

        # Pre-fetch metadata for all genes in the dataset
        genes = master["gene"].dropna().unique().tolist()
        ensure_gene_metadata(genes)

        from pipeline.transformers import (
            add_sequences,
            harmonize_scores,
            resolve_transcripts,
        )

        master = resolve_transcripts(master)
        master = harmonize_scores(master)
        master = add_sequences(master)
    else:
        logger.info("STEP 4: Transforms — SKIPPED (--skip-transforms)")

    # ── STEP 5: Scoring ──────────────────────────────────────────────
    if not skip_scoring:
        logger.info("=" * 70)
        logger.info("STEP 5: Scoring (AlphaMissense / ESM / Evo2)")
        logger.info("=" * 70)

        active_scorers = scorers or ["alphamissense", "esm", "evo2"]

        if "alphamissense" in active_scorers:
            from pipeline.scorers.score_alphamissense import score_alphamissense
            master = score_alphamissense(master)

        if "esm" in active_scorers:
            from pipeline.scorers.score_esm import score_esm
            from pipeline.config import ESM_DEFAULT_MODEL
            master = score_esm(
                master,
                model_name=esm_model or ESM_DEFAULT_MODEL,
                device=device,
            )

        if "evo2" in active_scorers:
            from pipeline.scorers.score_evo2 import score_evo2
            from pipeline.config import EVO2_DEFAULT_MODEL
            master = score_evo2(
                master,
                model_name=evo2_model or EVO2_DEFAULT_MODEL,
                device=device,
            )
    else:
        logger.info("STEP 5: Scoring — SKIPPED (--skip-scoring)")

    # ── STEP 6: Export ───────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 6: Exporting")
    logger.info("=" * 70)

    from pipeline.outputs.export import export_dataframe
    export_dataframe(master, output_dir=output_dir)

    # ── STEP 7: Finetuning ───────────────────────────────────────────
    if not skip_finetuning:
        logger.info("=" * 70)
        logger.info("STEP 7: Finetuning splits + formatting")
        logger.info("=" * 70)

        from pipeline.finetuning import (
            create_splits,
            format_am_finetuning,
            format_esm_finetuning,
            format_evo2_finetuning,
        )

        ft_dir = FINETUNING_OUTPUT_DIR
        splits = create_splits(master)
        format_esm_finetuning(splits, ft_dir / "esm")
        format_evo2_finetuning(splits, ft_dir / "evo2")
        format_am_finetuning(splits, ft_dir / "alphamissense")
    else:
        logger.info("STEP 7: Finetuning — SKIPPED (--skip-finetuning)")

    logger.info("=" * 70)
    logger.info("Pipeline complete. Rows=%d Cols=%d", *master.shape)
    logger.info("=" * 70)
    return master


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VEP Benchmark Pipeline: load → validate → merge → transform → score → export → finetune"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Input data directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    # Pipeline stage toggles
    parser.add_argument("--skip-scoring", action="store_true", help="Skip model scoring stage")
    parser.add_argument("--skip-finetuning", action="store_true", help="Skip finetuning stage")
    parser.add_argument("--skip-transforms", action="store_true", help="Skip transform stage (transcripts, scores, sequences)")

    # Scoring options
    parser.add_argument(
        "--scorers",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of scorers to run (default: alphamissense,esm,evo2)",
    )
    parser.add_argument("--esm-model", default=None, help="ESM model variant (default: esmc_300m)")
    parser.add_argument("--evo2-model", default=None, help="Evo2 model variant (default: evo2_7b)")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Force compute device (default: auto-detect)",
    )

    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        skip_scoring=args.skip_scoring,
        skip_finetuning=args.skip_finetuning,
        skip_transforms=args.skip_transforms,
        scorers=args.scorers,
        esm_model=args.esm_model,
        evo2_model=args.evo2_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
