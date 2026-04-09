"""AlphaMissense scorer — lookup precomputed pathogenicity scores.

DeepMind does not publish AlphaMissense weights.  Instead, precomputed
scores for ~71 M human missense variants are available from Zenodo.  This
module downloads the TSV once, converts it to an indexed SQLite database
for fast lookups, and fills ``alphamissense_score`` / ``alphamissense_label``
in the benchmark dataframe.
"""

from __future__ import annotations

import gzip
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from pipeline.config import (
    ALPHAMISSENSE_DB_PATH,
    ALPHAMISSENSE_ZENODO_URL,
    CACHE_DIR,
    SCORER_CACHE_DIR,
)
from pipeline.scorers.base import batch_iter, load_parquet_cache, save_parquet_cache

logger = logging.getLogger(__name__)

_TSV_GZ_PATH = CACHE_DIR / "AlphaMissense_hg38.tsv.gz"


# ── Download ─────────────────────────────────────────────────────────


def _download_zenodo_tsv(dest: Path) -> None:
    """Stream-download the AlphaMissense TSV from Zenodo (~4 GB)."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading AlphaMissense TSV from Zenodo (this may take a while)...")
    logger.info("  URL: %s", ALPHAMISSENSE_ZENODO_URL)

    resp = requests.get(ALPHAMISSENSE_ZENODO_URL, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                logger.info("  %.1f%% (%d / %d MB)", pct, downloaded // 1e6, total // 1e6)

    logger.info("Download complete: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


# ── SQLite conversion ────────────────────────────────────────────────


def _build_sqlite_db(tsv_gz_path: Path, db_path: Path) -> None:
    """Convert the gzipped TSV to an indexed SQLite database.

    Reads the TSV in chunks to keep memory usage low.  Creates an index
    on (chrom, pos, ref, alt) for fast coordinate lookups and a second
    index on (uniprot_id, protein_variant) for protein-level fallback.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building SQLite database from %s (one-time operation)...", tsv_gz_path.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS variants (
            chrom      TEXT,
            pos        INTEGER,
            ref        TEXT,
            alt        TEXT,
            genome     TEXT,
            uniprot_id TEXT,
            transcript_id TEXT,
            protein_variant TEXT,
            am_pathogenicity REAL,
            am_class   TEXT
        )
        """
    )
    conn.execute("DELETE FROM variants")  # clear if rebuilding

    rows_inserted = 0
    chunk_size = 500_000

    with gzip.open(tsv_gz_path, "rt") as fh:
        reader = pd.read_csv(
            fh,
            sep="\t",
            comment="#",
            chunksize=chunk_size,
            dtype={"#CHROM": str, "POS": int, "REF": str, "ALT": str},
        )
        for chunk in reader:
            # Normalise column names
            chunk = chunk.rename(columns={"#CHROM": "chrom", "POS": "pos", "REF": "ref", "ALT": "alt"})
            cols = [
                "chrom", "pos", "ref", "alt", "genome",
                "uniprot_id", "transcript_id", "protein_variant",
                "am_pathogenicity", "am_class",
            ]
            existing = [c for c in cols if c in chunk.columns]
            chunk[existing].to_sql("variants", conn, if_exists="append", index=False)
            rows_inserted += len(chunk)
            logger.info("  Inserted %d rows so far...", rows_inserted)

    logger.info("Creating indexes...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_coord ON variants (chrom, pos, ref, alt)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_protein ON variants (uniprot_id, protein_variant)")
    conn.commit()
    conn.close()

    logger.info("SQLite DB ready: %s (%d rows, %.1f MB)", db_path.name, rows_inserted, db_path.stat().st_size / 1e6)


def _ensure_db() -> Path:
    """Ensure the SQLite database exists, downloading + building if needed."""
    if ALPHAMISSENSE_DB_PATH.exists():
        return ALPHAMISSENSE_DB_PATH
    if not _TSV_GZ_PATH.exists():
        _download_zenodo_tsv(_TSV_GZ_PATH)
    _build_sqlite_db(_TSV_GZ_PATH, ALPHAMISSENSE_DB_PATH)
    return ALPHAMISSENSE_DB_PATH


# ── Lookup ───────────────────────────────────────────────────────────


def _lookup_by_coord(
    conn: sqlite3.Connection,
    rows: list[dict],
) -> dict[int, tuple[float, str]]:
    """Batch-lookup scores by genomic coordinate.

    Returns {dataframe_index: (am_pathogenicity, am_class)}.
    """
    results: dict[int, tuple[float, str]] = {}
    for batch in batch_iter(rows, 1000):
        placeholders = ",".join(["(?,?,?,?)"] * len(batch))
        params = []
        idx_map = []
        for row in batch:
            chrom = str(row["chrom"]).replace("chr", "")
            params.extend([chrom, int(row["hg38_pos"]), str(row["ref_allele"]), str(row["alt_allele"])])
            idx_map.append(row["_idx"])

        query = f"""
            SELECT chrom, pos, ref, alt, am_pathogenicity, am_class
            FROM variants
            WHERE (chrom, pos, ref, alt) IN (VALUES {placeholders})
        """
        cursor = conn.execute(query, params)
        found = {}
        for r in cursor.fetchall():
            key = (str(r[0]), int(r[1]), str(r[2]), str(r[3]))
            found[key] = (r[4], r[5])

        for row in batch:
            chrom = str(row["chrom"]).replace("chr", "")
            key = (chrom, int(row["hg38_pos"]), str(row["ref_allele"]), str(row["alt_allele"]))
            if key in found:
                results[row["_idx"]] = found[key]

    return results


def _lookup_by_protein(
    conn: sqlite3.Connection,
    rows: list[dict],
) -> dict[int, tuple[float, str]]:
    """Fallback: lookup by protein variant notation (e.g. 'L263Y')."""
    from pipeline.config import AA1TO3

    results: dict[int, tuple[float, str]] = {}
    for row in rows:
        aa_ref = row.get("aa_ref")
        aa_pos = row.get("aa_pos")
        aa_alt = row.get("aa_alt")
        if not aa_ref or not aa_alt or pd.isna(aa_pos):
            continue

        # AlphaMissense uses 3-letter notation: e.g. "L263Y" → check both formats
        ref3 = AA1TO3.get(str(aa_ref), str(aa_ref))
        alt3 = AA1TO3.get(str(aa_alt), str(aa_alt))
        protein_var = f"{ref3}{int(aa_pos)}{alt3}"
        protein_var_1letter = f"{aa_ref}{int(aa_pos)}{aa_alt}"

        cursor = conn.execute(
            "SELECT am_pathogenicity, am_class FROM variants "
            "WHERE protein_variant IN (?, ?) LIMIT 1",
            (protein_var, protein_var_1letter),
        )
        hit = cursor.fetchone()
        if hit:
            results[row["_idx"]] = (hit[0], hit[1])

    return results


# ── Main scorer ──────────────────────────────────────────────────────


def score_alphamissense(df: pd.DataFrame) -> pd.DataFrame:
    """Fill ``alphamissense_score`` and ``alphamissense_label`` from precomputed DB.

    Rows that already have a non-null ``alphamissense_score`` are skipped.
    """
    # Ensure columns exist
    if "alphamissense_score" not in df.columns:
        df["alphamissense_score"] = pd.NA
    if "alphamissense_label" not in df.columns:
        df["alphamissense_label"] = pd.NA

    needs_score = df["alphamissense_score"].isna()
    n_need = needs_score.sum()
    if n_need == 0:
        logger.info("AlphaMissense: all rows already scored — skipping")
        return df

    logger.info("AlphaMissense: %d / %d rows need scoring", n_need, len(df))

    # Check cache first
    cache_path = SCORER_CACHE_DIR / "am_scores.parquet"
    cached = load_parquet_cache(cache_path)
    if cached is not None and "genomic_coord" in cached.columns:
        cache_map = dict(
            zip(cached["genomic_coord"], zip(cached["am_pathogenicity"], cached["am_class"]))
        )
        for idx in df.index[needs_score]:
            coord = df.at[idx, "genomic_coord"] if "genomic_coord" in df.columns else None
            if coord and coord in cache_map:
                score, label = cache_map[coord]
                df.at[idx, "alphamissense_score"] = score
                df.at[idx, "alphamissense_label"] = label
        needs_score = df["alphamissense_score"].isna()
        n_need = needs_score.sum()
        logger.info("AlphaMissense: %d rows remain after cache lookup", n_need)
        if n_need == 0:
            return df

    # Build / locate SQLite DB
    try:
        db_path = _ensure_db()
    except Exception:
        logger.exception("AlphaMissense: failed to prepare database — skipping scorer")
        return df

    conn = sqlite3.connect(str(db_path))

    # Prepare rows needing scores
    todo = df.loc[needs_score].copy()
    todo["_idx"] = todo.index
    todo_dicts = todo.to_dict("records")

    # Genomic-coordinate lookup
    coord_rows = [
        r for r in todo_dicts
        if pd.notna(r.get("hg38_pos")) and pd.notna(r.get("ref_allele"))
    ]
    coord_hits = _lookup_by_coord(conn, coord_rows) if coord_rows else {}
    logger.info("AlphaMissense: %d hits by genomic coordinate", len(coord_hits))

    for idx, (score, label) in coord_hits.items():
        df.at[idx, "alphamissense_score"] = score
        df.at[idx, "alphamissense_label"] = label

    # Protein-level fallback for remaining
    still_need = [r for r in todo_dicts if r["_idx"] not in coord_hits]
    protein_hits = _lookup_by_protein(conn, still_need) if still_need else {}
    logger.info("AlphaMissense: %d hits by protein variant (fallback)", len(protein_hits))

    for idx, (score, label) in protein_hits.items():
        df.at[idx, "alphamissense_score"] = score
        df.at[idx, "alphamissense_label"] = label

    conn.close()

    # Update cache
    scored = df[df["alphamissense_score"].notna()][
        ["genomic_coord", "alphamissense_score", "alphamissense_label"]
    ].copy()
    scored = scored.rename(columns={
        "alphamissense_score": "am_pathogenicity",
        "alphamissense_label": "am_class",
    })
    save_parquet_cache(scored, cache_path)

    total_scored = df["alphamissense_score"].notna().sum()
    logger.info(
        "AlphaMissense: scoring complete — %d / %d rows have scores (%.1f%%)",
        total_scored, len(df), total_scored / len(df) * 100,
    )
    return df
