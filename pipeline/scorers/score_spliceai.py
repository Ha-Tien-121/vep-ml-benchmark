"""SpliceAI scorer — streaming lookup of precomputed splice-disruption delta scores.

Strategy
--------
Rather than indexing all 438 M rows into SQLite (slow, huge RAM/disk), this
module streams the gzipped VCF *once* and collects only the rows that match
variants already present in the master DataFrame.  Results are cached to a
small parquet file so subsequent pipeline runs are instant.

Four delta scores are extracted per variant:

    DS_AG  acceptor gain
    DS_AL  acceptor loss
    DS_DG  donor gain
    DS_DL  donor loss

plus ``spliceai_max_delta_score = max(DS_AG, DS_AL, DS_DG, DS_DL)``.

When the annotation VCF is absent the scorer logs a warning and returns the
DataFrame unchanged (all five columns will be null).
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import pandas as pd

from pipeline.config import (
    SCORER_CACHE_DIR,
    SPLICEAI_FILE,
)

logger = logging.getLogger(__name__)

SPLICEAI_COLS = [
    "spliceai_DS_AG",
    "spliceai_DS_AL",
    "spliceai_DS_DG",
    "spliceai_DS_DL",
    "spliceai_max_delta_score",
]

# ── File open helper ──────────────────────────────────────────────────────────


def _open_text(path: Path):
    """Open a plain or gzip-compressed text file for line-by-line reading."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


# ── Parquet cache helpers ─────────────────────────────────────────────────────

_CACHE_COLS = ["chrom", "pos", "ref", "alt", "ds_ag", "ds_al", "ds_dg", "ds_dl"]


def _cache_path() -> Path:
    SCORER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SCORER_CACHE_DIR / "spliceai_scores.parquet"


def _load_cache() -> pd.DataFrame:
    """Load cached SpliceAI scores or return empty DataFrame."""
    p = _cache_path()
    if p.exists():
        try:
            return pd.read_parquet(p)
        except (OSError, ValueError) as exc:
            logger.warning("SpliceAI: could not read cache (%s) — starting fresh", exc)
    return pd.DataFrame(columns=_CACHE_COLS)


def _save_cache(df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(), index=False)


# ── VCF streamer ─────────────────────────────────────────────────────────────


def _stream_vcf(
    src: Path,
    lookup: set[tuple[str, int, str, str]],
) -> dict[tuple[str, int, str, str], tuple[float, float, float, float]]:
    """Stream through the SpliceAI VCF and collect scores for target variants.

    Parameters
    ----------
    lookup:
        Set of ``(chrom, pos, ref, alt)`` tuples to find.  ``chrom`` must be
        without the ``chr`` prefix and ``pos`` must be an integer.

    Returns
    -------
    dict mapping each found ``(chrom, pos, ref, alt)`` to
    ``(max_DS_AG, max_DS_AL, max_DS_DG, max_DS_DL)`` taking the maximum
    across multi-gene annotations for the same coordinate.
    """
    results: dict[tuple, tuple] = {}
    remaining = set(lookup)

    # Build a two-level index: (chrom, pos) -> set[(ref, alt)]
    # so we can skip irrelevant lines with a single dict lookup.
    pos_index: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for chrom, pos, ref, alt in lookup:
        pos_index.setdefault((chrom, pos), set()).add((ref, alt))

    n_target = len(lookup)
    lines_read = 0

    with _open_text(src) as fh:
        for line in fh:
            if line.startswith("#"):
                continue

            lines_read += 1
            if lines_read % 5_000_000 == 0:
                pct = 100 * len(results) / n_target if n_target else 100
                logger.info(
                    "  SpliceAI stream: %dM lines read — %d / %d variants found (%.0f%%)",
                    lines_read // 1_000_000,
                    len(results),
                    n_target,
                    pct,
                )
                if not remaining:
                    logger.info("  SpliceAI stream: all target variants found — stopping early")
                    break

            parts = line.rstrip("\n").split("\t", 8)
            if len(parts) < 8:
                continue

            chrom = parts[0].replace("chr", "")
            try:
                pos = int(parts[1])
            except ValueError:
                continue

            # Fast skip: is this position in our target set?
            if (chrom, pos) not in pos_index:
                continue

            ref = parts[3]
            alts = parts[4].split(",")
            info = parts[7]

            # Parse SpliceAI= block from INFO
            sa_start = info.find("SpliceAI=")
            if sa_start == -1:
                continue
            sa_value = info[sa_start + 9:]
            sa_end = sa_value.find(";")
            if sa_end != -1:
                sa_value = sa_value[:sa_end]

            # Each comma-separated annotation: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|...
            # Accumulate MAX scores across multi-gene annotations per alt allele.
            per_alt: dict[str, list[float]] = {}

            for annotation in sa_value.split(","):
                fields = annotation.split("|")
                if len(fields) < 6:
                    continue
                allele = fields[0]
                try:
                    ds_ag = float(fields[2])
                    ds_al = float(fields[3])
                    ds_dg = float(fields[4])
                    ds_dl = float(fields[5])
                except ValueError:
                    continue

                matched = alts if allele == "." else [a for a in alts if a == allele]
                if not matched:
                    matched = alts

                for alt in matched:
                    if alt not in per_alt:
                        per_alt[alt] = [ds_ag, ds_al, ds_dg, ds_dl]
                    else:
                        cur = per_alt[alt]
                        per_alt[alt] = [
                            max(cur[0], ds_ag),
                            max(cur[1], ds_al),
                            max(cur[2], ds_dg),
                            max(cur[3], ds_dl),
                        ]

            for alt, scores in per_alt.items():
                key = (chrom, pos, ref, alt)
                if key in remaining:
                    results[key] = tuple(scores)
                    remaining.discard(key)

    logger.info(
        "SpliceAI stream complete: %dM lines read — %d / %d variants found",
        lines_read // 1_000_000,
        len(results),
        n_target,
    )
    return results


# ── Main scorer ───────────────────────────────────────────────────────────────


def score_spliceai(df: pd.DataFrame) -> pd.DataFrame:
    """Fill SpliceAI delta-score columns from the precomputed annotation VCF.

    Adds five columns: spliceai_DS_AG, spliceai_DS_AL, spliceai_DS_DG,
    spliceai_DS_DL, spliceai_max_delta_score.  Rows where all five are already
    non-null are skipped.

    On first run this streams the full VCF (≈27 GB) once and caches results to
    a small parquet file.  Subsequent runs read from cache in seconds.
    """
    # Ensure output columns exist
    for col in SPLICEAI_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    needs_score = df[SPLICEAI_COLS].isna().any(axis=1)
    n_need = needs_score.sum()
    if n_need == 0:
        logger.info("SpliceAI: all rows already scored — skipping")
        return df

    logger.info("SpliceAI: %d / %d rows need scoring", n_need, len(df))

    # ── Build variant lookup set from rows that have full genomic coords ──
    todo = df.loc[needs_score].copy()
    has_coords = (
        todo["chrom"].notna()
        & todo["hg38_pos"].notna()
        & todo["ref_allele"].notna()
        & todo["alt_allele"].notna()
    )
    todo = todo.loc[has_coords]

    if todo.empty:
        logger.info("SpliceAI: no rows have full genomic coordinates — skipping")
        return df

    # Normalise chrom (strip 'chr') and cast pos to int for lookup
    todo = todo.copy()
    todo["_chrom"] = todo["chrom"].astype(str).str.replace("^chr", "", regex=True)
    todo["_pos"]   = pd.to_numeric(todo["hg38_pos"], errors="coerce").dropna().astype(int)
    todo = todo.dropna(subset=["_pos"])
    todo["_pos"]   = todo["_pos"].astype(int)

    # Unique variants we need to look up
    lookup: set[tuple[str, int, str, str]] = set(
        zip(
            todo["_chrom"].tolist(),
            todo["_pos"].tolist(),
            todo["ref_allele"].astype(str).tolist(),
            todo["alt_allele"].astype(str).tolist(),
        )
    )
    logger.info("SpliceAI: %d unique genomic coordinates to look up", len(lookup))

    # ── Check parquet cache ───────────────────────────────────────────────
    cache_df = _load_cache()
    score_map: dict[tuple, tuple] = {}

    if not cache_df.empty:
        cached_keys = set(
            zip(
                cache_df["chrom"].astype(str).tolist(),
                cache_df["pos"].astype(int).tolist(),
                cache_df["ref"].astype(str).tolist(),
                cache_df["alt"].astype(str).tolist(),
            )
        )
        for row in cache_df.itertuples(index=False):
            score_map[(str(row.chrom), int(row.pos), str(row.ref), str(row.alt))] = (
                row.ds_ag, row.ds_al, row.ds_dg, row.ds_dl
            )

        uncached = lookup - cached_keys
        logger.info(
            "SpliceAI cache: %d / %d variants already cached, %d need VCF scan",
            len(lookup) - len(uncached),
            len(lookup),
            len(uncached),
        )
    else:
        uncached = lookup

    # ── Stream VCF for any uncached variants ─────────────────────────────
    if uncached:
        if SPLICEAI_FILE is None or not Path(SPLICEAI_FILE).exists():
            logger.warning(
                "SpliceAI annotation file not found: %s\n"
                "  Download from Ensembl/BaseSpace and set SPLICEAI_FILE in config.py.\n"
                "  Continuing without SpliceAI scores.",
                SPLICEAI_FILE,
            )
            return df

        logger.info(
            "SpliceAI: streaming VCF for %d uncached variants — this may take several minutes",
            len(uncached),
        )
        new_hits = _stream_vcf(Path(SPLICEAI_FILE), uncached)
        score_map.update(new_hits)

        # Append new hits to cache
        if new_hits:
            new_rows = pd.DataFrame(
                [
                    {
                        "chrom": k[0],
                        "pos":   k[1],
                        "ref":   k[2],
                        "alt":   k[3],
                        "ds_ag": v[0],
                        "ds_al": v[1],
                        "ds_dg": v[2],
                        "ds_dl": v[3],
                    }
                    for k, v in new_hits.items()
                ]
            )
            updated_cache = pd.concat([cache_df, new_rows], ignore_index=True)
            _save_cache(updated_cache)
            logger.info("SpliceAI: cache updated with %d new entries", len(new_hits))

    # ── Apply scores to DataFrame ─────────────────────────────────────────
    n_hits = 0
    for idx in todo.index:
        key = (
            str(todo.at[idx, "_chrom"]),
            int(todo.at[idx, "_pos"]),
            str(df.at[idx, "ref_allele"]),
            str(df.at[idx, "alt_allele"]),
        )
        if key in score_map:
            ag, al, dg, dl = score_map[key]
            df.at[idx, "spliceai_DS_AG"] = ag
            df.at[idx, "spliceai_DS_AL"] = al
            df.at[idx, "spliceai_DS_DG"] = dg
            df.at[idx, "spliceai_DS_DL"] = dl
            vals = [v for v in (ag, al, dg, dl) if v is not None and v == v]
            df.at[idx, "spliceai_max_delta_score"] = max(vals) if vals else pd.NA
            n_hits += 1

    total_scored = df["spliceai_DS_AG"].notna().sum()
    logger.info(
        "SpliceAI: scoring complete — %d / %d rows have scores (%.1f%%)",
        total_scored,
        len(df),
        total_scored / len(df) * 100,
    )
    return df
