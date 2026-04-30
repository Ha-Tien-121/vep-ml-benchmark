"""SpliceAI scorer — lookup precomputed splice-disruption delta scores.

Illumina distributes precomputed SpliceAI scores for all SNVs in hg38 via
BaseSpace (manual download required).  This module builds a SQLite index
from the downloaded VCF or TSV once, then looks up four delta scores per
variant:

    DS_AG  acceptor gain
    DS_AL  acceptor loss
    DS_DG  donor gain
    DS_DL  donor loss

plus a summary ``spliceai_max_delta_score = max(DS_AG, DS_AL, DS_DG, DS_DL)``.

When the annotation file is absent the scorer logs a warning and returns the
dataframe unchanged (all five columns will be null).  Existing non-null values
are never overwritten.
"""

from __future__ import annotations

import contextlib
import gzip
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from pipeline.config import (
    SCORER_CACHE_DIR,
    SPLICEAI_DB_PATH,
    SPLICEAI_FILE,
)
from pipeline.scorers.base import batch_iter, load_parquet_cache, save_parquet_cache

logger = logging.getLogger(__name__)

SPLICEAI_COLS = [
    "spliceai_DS_AG",
    "spliceai_DS_AL",
    "spliceai_DS_DG",
    "spliceai_DS_DL",
    "spliceai_max_delta_score",
]


# ── Format detection ─────────────────────────────────────────────────


def _detect_format(path: Path) -> str:
    """Return 'vcf' or 'tsv' based on file extension (ignoring .gz suffix)."""
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".vcf"):
        return "vcf"
    if name.endswith((".tsv", ".txt", ".csv")):
        return "tsv"
    raise ValueError(f"Unrecognised SpliceAI file extension: {path.name}")


# ── File open helper ─────────────────────────────────────────────────


@contextlib.contextmanager
def _open_text(path: Path):
    """Open a plain or gzip-compressed text file for reading."""
    if path.name.lower().endswith(".gz"):
        fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        fh = open(path, "rt", encoding="utf-8", errors="replace")
    try:
        yield fh
    finally:
        fh.close()


# ── SQLite builders ──────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS variants (
    chrom TEXT,
    pos   INTEGER,
    ref   TEXT,
    alt   TEXT,
    ds_ag REAL,
    ds_al REAL,
    ds_dg REAL,
    ds_dl REAL
)
"""
_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_coord ON variants(chrom, pos, ref, alt)"


def _build_sqlite_db_from_tsv(src: Path, db_path: Path) -> None:
    """Build SQLite DB from a TSV/CSV SpliceAI annotation file."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building SpliceAI SQLite DB from TSV: %s", src.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE)
    conn.execute("DELETE FROM variants")

    # Column aliases accepted from various TSV exports
    _CHROM_ALIASES = {"#chrom", "chrom", "chr", "chromosome"}
    _POS_ALIASES   = {"pos", "position", "start"}
    _REF_ALIASES   = {"ref", "ref_allele"}
    _ALT_ALIASES   = {"alt", "alt_allele"}
    _DS_AG_ALIASES = {"ds_ag", "spliceai_ds_ag"}
    _DS_AL_ALIASES = {"ds_al", "spliceai_ds_al"}
    _DS_DG_ALIASES = {"ds_dg", "spliceai_ds_dg"}
    _DS_DL_ALIASES = {"ds_dl", "spliceai_ds_dl"}

    rows_inserted = 0
    with _open_text(src) as fh:
        for chunk in pd.read_csv(fh, sep="\t", comment="#", chunksize=500_000, low_memory=False):
            lower_map = {c: c.lower() for c in chunk.columns}
            chunk = chunk.rename(columns=lower_map)
            cols = set(chunk.columns)

            chrom_col = next((c for c in chunk.columns if c in _CHROM_ALIASES), None)
            pos_col   = next((c for c in chunk.columns if c in _POS_ALIASES), None)
            ref_col   = next((c for c in chunk.columns if c in _REF_ALIASES), None)
            alt_col   = next((c for c in chunk.columns if c in _ALT_ALIASES), None)
            ag_col    = next((c for c in chunk.columns if c in _DS_AG_ALIASES), None)
            al_col    = next((c for c in chunk.columns if c in _DS_AL_ALIASES), None)
            dg_col    = next((c for c in chunk.columns if c in _DS_DG_ALIASES), None)
            dl_col    = next((c for c in chunk.columns if c in _DS_DL_ALIASES), None)

            if not all([chrom_col, pos_col, ref_col, alt_col]):
                logger.warning("SpliceAI TSV: could not identify required columns in %s", cols)
                continue

            sub = pd.DataFrame({
                "chrom": chunk[chrom_col].astype(str).str.replace("^chr", "", regex=True),
                "pos":   pd.to_numeric(chunk[pos_col], errors="coerce"),
                "ref":   chunk[ref_col].astype(str),
                "alt":   chunk[alt_col].astype(str),
                "ds_ag": pd.to_numeric(chunk[ag_col], errors="coerce") if ag_col else None,
                "ds_al": pd.to_numeric(chunk[al_col], errors="coerce") if al_col else None,
                "ds_dg": pd.to_numeric(chunk[dg_col], errors="coerce") if dg_col else None,
                "ds_dl": pd.to_numeric(chunk[dl_col], errors="coerce") if dl_col else None,
            }).dropna(subset=["pos"])

            sub.to_sql("variants", conn, if_exists="append", index=False)
            rows_inserted += len(sub)
            logger.info("  SpliceAI TSV: inserted %d rows so far...", rows_inserted)

    conn.execute(_CREATE_INDEX)
    conn.commit()
    conn.close()
    logger.info("SpliceAI SQLite DB ready: %s (%d rows)", db_path.name, rows_inserted)


def _build_sqlite_db_from_vcf(src: Path, db_path: Path) -> None:
    """Build SQLite DB from a SpliceAI precomputed VCF file.

    Parses SpliceAI= annotations from the INFO field.  Each VCF row may
    carry multiple gene/allele annotations separated by commas; one DB row
    is inserted per matching annotation.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building SpliceAI SQLite DB from VCF: %s", src.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE)
    conn.execute("DELETE FROM variants")

    batch: list[tuple] = []
    rows_inserted = 0

    def _flush(b: list[tuple]) -> None:
        conn.executemany(
            "INSERT INTO variants(chrom,pos,ref,alt,ds_ag,ds_al,ds_dg,ds_dl) VALUES(?,?,?,?,?,?,?,?)",
            b,
        )
        conn.commit()

    with _open_text(src) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue

            chrom = parts[0].replace("chr", "")
            pos   = int(parts[1])
            ref   = parts[3]
            alts  = parts[4].split(",")
            info  = parts[7]

            # Locate SpliceAI= block inside INFO
            sa_start = info.find("SpliceAI=")
            if sa_start == -1:
                continue
            sa_value = info[sa_start + len("SpliceAI="):]
            # Value ends at next semicolon or end-of-string
            sa_end = sa_value.find(";")
            if sa_end != -1:
                sa_value = sa_value[:sa_end]

            # Each comma-separated annotation: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|...
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

                # Match allele against ALT column (multi-allelic aware)
                matched_alts = alts if allele == "." else [a for a in alts if a == allele]
                if not matched_alts:
                    matched_alts = alts  # fallback: apply to all alts

                for alt in matched_alts:
                    batch.append((chrom, pos, ref, alt, ds_ag, ds_al, ds_dg, ds_dl))

                if len(batch) >= 50_000:
                    _flush(batch)
                    rows_inserted += len(batch)
                    logger.info("  SpliceAI VCF: inserted %d rows so far...", rows_inserted)
                    batch = []

    if batch:
        _flush(batch)
        rows_inserted += len(batch)

    conn.execute(_CREATE_INDEX)
    conn.commit()
    conn.close()
    logger.info("SpliceAI SQLite DB ready: %s (%d rows)", db_path.name, rows_inserted)


# ── DB ensure ────────────────────────────────────────────────────────


def _ensure_db() -> Path | None:
    """Return path to the SpliceAI SQLite DB, building it if needed.

    Returns None if the source annotation file is not available.
    """
    if SPLICEAI_DB_PATH.exists():
        return SPLICEAI_DB_PATH
    if SPLICEAI_FILE is None or not Path(SPLICEAI_FILE).exists():
        logger.warning(
            "SpliceAI annotation file not found: %s\n"
            "  Download from Illumina BaseSpace and place at that path, "
            "or set SPLICEAI_FILE in config.py.\n"
            "  Continuing without SpliceAI scores.",
            SPLICEAI_FILE,
        )
        return None
    fmt = _detect_format(Path(SPLICEAI_FILE))
    if fmt == "vcf":
        _build_sqlite_db_from_vcf(Path(SPLICEAI_FILE), SPLICEAI_DB_PATH)
    else:
        _build_sqlite_db_from_tsv(Path(SPLICEAI_FILE), SPLICEAI_DB_PATH)
    return SPLICEAI_DB_PATH


# ── Lookup ───────────────────────────────────────────────────────────


def _lookup_by_coord(
    conn: sqlite3.Connection,
    rows: list[dict],
) -> dict[int, tuple[float, float, float, float]]:
    """Batch-lookup SpliceAI scores by genomic coordinate.

    Uses MAX() aggregation per coordinate to handle multi-gene VCF rows,
    returning the worst-case delta score across all gene annotations.

    Returns {dataframe_index: (ds_ag, ds_al, ds_dg, ds_dl)}.
    """
    results: dict[int, tuple[float, float, float, float]] = {}

    for batch in batch_iter(rows, 1000):
        placeholders = ",".join(["(?,?,?,?)"] * len(batch))
        params: list = []
        idx_map: list[tuple] = []

        for row in batch:
            chrom = str(row["chrom"]).replace("chr", "")
            params.extend([chrom, int(row["hg38_pos"]), str(row["ref_allele"]), str(row["alt_allele"])])
            idx_map.append((row["_idx"], chrom, int(row["hg38_pos"]), str(row["ref_allele"]), str(row["alt_allele"])))

        query = f"""
            SELECT chrom, pos, ref, alt,
                   MAX(ds_ag), MAX(ds_al), MAX(ds_dg), MAX(ds_dl)
            FROM variants
            WHERE (chrom, pos, ref, alt) IN (VALUES {placeholders})
            GROUP BY chrom, pos, ref, alt
        """
        cursor = conn.execute(query, params)
        found: dict[tuple, tuple] = {}
        for r in cursor.fetchall():
            key = (str(r[0]), int(r[1]), str(r[2]), str(r[3]))
            found[key] = (r[4], r[5], r[6], r[7])

        for df_idx, chrom, pos, ref, alt in idx_map:
            key = (chrom, pos, ref, alt)
            if key in found:
                results[df_idx] = found[key]

    return results


# ── Main scorer ──────────────────────────────────────────────────────


def score_spliceai(df: pd.DataFrame) -> pd.DataFrame:
    """Fill SpliceAI delta-score columns from precomputed annotation DB.

    Adds five columns: spliceai_DS_AG, spliceai_DS_AL, spliceai_DS_DG,
    spliceai_DS_DL, spliceai_max_delta_score.  Rows where all five are
    already non-null are skipped.
    """
    # Ensure columns exist
    for col in SPLICEAI_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    needs_score = df[SPLICEAI_COLS].isna().any(axis=1)
    n_need = needs_score.sum()
    if n_need == 0:
        logger.info("SpliceAI: all rows already scored — skipping")
        return df

    logger.info("SpliceAI: %d / %d rows need scoring", n_need, len(df))

    # Parquet cache lookup
    cache_path = SCORER_CACHE_DIR / "spliceai_scores.parquet"
    cached = load_parquet_cache(cache_path)
    if cached is not None and "genomic_coord" in cached.columns:
        cache_map = dict(
            zip(
                cached["genomic_coord"],
                zip(
                    cached["ds_ag"], cached["ds_al"],
                    cached["ds_dg"], cached["ds_dl"],
                    cached["max_delta_score"],
                ),
            )
        )
        for idx in df.index[needs_score]:
            coord = df.at[idx, "genomic_coord"] if "genomic_coord" in df.columns else None
            if coord and coord in cache_map:
                ag, al, dg, dl, mx = cache_map[coord]
                df.at[idx, "spliceai_DS_AG"]           = ag
                df.at[idx, "spliceai_DS_AL"]           = al
                df.at[idx, "spliceai_DS_DG"]           = dg
                df.at[idx, "spliceai_DS_DL"]           = dl
                df.at[idx, "spliceai_max_delta_score"] = mx
        needs_score = df[SPLICEAI_COLS].isna().any(axis=1)
        n_need = needs_score.sum()
        logger.info("SpliceAI: %d rows remain after cache lookup", n_need)
        if n_need == 0:
            return df

    # Ensure DB exists
    db_path = _ensure_db()
    if db_path is None:
        return df

    conn = sqlite3.connect(str(db_path))

    # Build lookup records
    todo = df.loc[needs_score].copy()
    todo["_idx"] = todo.index
    coord_rows = [
        r for r in todo.to_dict("records")
        if pd.notna(r.get("chrom")) and pd.notna(r.get("hg38_pos"))
        and pd.notna(r.get("ref_allele")) and pd.notna(r.get("alt_allele"))
    ]

    if not coord_rows:
        logger.info("SpliceAI: no rows have full genomic coordinates — skipping DB lookup")
        conn.close()
        return df

    hits = _lookup_by_coord(conn, coord_rows)
    conn.close()
    logger.info("SpliceAI: %d / %d coordinate hits", len(hits), len(coord_rows))

    for idx, (ag, al, dg, dl) in hits.items():
        df.at[idx, "spliceai_DS_AG"] = ag
        df.at[idx, "spliceai_DS_AL"] = al
        df.at[idx, "spliceai_DS_DG"] = dg
        df.at[idx, "spliceai_DS_DL"] = dl
        vals = [v for v in (ag, al, dg, dl) if v is not None and not (isinstance(v, float) and v != v)]
        df.at[idx, "spliceai_max_delta_score"] = max(vals) if vals else pd.NA

    # Save to parquet cache
    scored_mask = df["spliceai_DS_AG"].notna()
    if scored_mask.any():
        cache_df = df.loc[scored_mask, [
            "genomic_coord",
            "spliceai_DS_AG", "spliceai_DS_AL",
            "spliceai_DS_DG", "spliceai_DS_DL",
            "spliceai_max_delta_score",
        ]].copy().rename(columns={
            "spliceai_DS_AG": "ds_ag",
            "spliceai_DS_AL": "ds_al",
            "spliceai_DS_DG": "ds_dg",
            "spliceai_DS_DL": "ds_dl",
            "spliceai_max_delta_score": "max_delta_score",
        })
        save_parquet_cache(cache_df, cache_path)

    total_scored = df["spliceai_DS_AG"].notna().sum()
    logger.info(
        "SpliceAI: scoring complete — %d / %d rows have scores (%.1f%%)",
        total_scored, len(df), total_scored / len(df) * 100,
    )
    return df
