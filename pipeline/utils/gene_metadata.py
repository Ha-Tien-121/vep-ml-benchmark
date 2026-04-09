"""Auto-fetch and cache canonical transcript / sequence metadata for any gene.

Replaces the hardcoded 3-gene dictionaries in config.py with a dynamic
lookup that supports all 40+ genes in the Pillar data.  Results are cached
to ``data/cache/gene_metadata.json`` for offline reproducibility.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pipeline.config import (
    CACHE_DIR,
    CANONICAL_REFSEQ,
    CANONICAL_TRANSCRIPTS,
    DNA_FASTA_DIR,
    PROTEIN_FASTA_DIR,
)

logger = logging.getLogger(__name__)

_CACHE_PATH = CACHE_DIR / "gene_metadata.json"
_ENSEMBL_BASE = "https://rest.ensembl.org"
_REQUEST_DELAY = 0.1  # seconds between requests (stay under 15 req/s)

# In-memory cache: gene -> { "ensembl_transcript": ..., "refseq_id": ..., "protein_sequence": ..., "dna_sequence": ... }
_cache: dict[str, dict] = {}


def _load_disk_cache() -> None:
    """Load the JSON cache from disk into memory."""
    global _cache
    if _cache:
        return
    if _CACHE_PATH.exists():
        try:
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            logger.info("Loaded gene metadata cache: %d genes", len(_cache))
        except Exception:
            logger.warning("Failed to read gene metadata cache — starting fresh")
            _cache = {}
    # Seed with hardcoded overrides from config
    for gene, tx in CANONICAL_TRANSCRIPTS.items():
        _cache.setdefault(gene, {})["ensembl_transcript"] = tx
    for gene, refseq in CANONICAL_REFSEQ.items():
        _cache.setdefault(gene, {})["refseq_id"] = refseq


def _save_disk_cache() -> None:
    """Persist the in-memory cache to disk."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_cache, indent=2), encoding="utf-8")


def _ensembl_get(endpoint: str, params: dict | None = None) -> dict | None:
    """GET a JSON endpoint from Ensembl REST API with rate-limiting."""
    import requests

    url = f"{_ENSEMBL_BASE}{endpoint}"
    headers = {"Content-Type": "application/json"}
    try:
        time.sleep(_REQUEST_DELAY)
        resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
        if resp.status_code == 429:
            retry = float(resp.headers.get("Retry-After", 2))
            logger.warning("Ensembl rate limit — waiting %.1fs", retry)
            time.sleep(retry)
            resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.debug("Ensembl request failed: %s", endpoint)
        return None


def _fetch_gene_info(gene: str) -> dict:
    """Fetch canonical transcript, RefSeq ID, and sequences from Ensembl."""
    info: dict = {}

    # 1. Lookup gene → canonical transcript
    data = _ensembl_get(f"/lookup/symbol/homo_sapiens/{gene}", {"expand": "1"})
    if data and "Transcript" in data:
        for tx in data["Transcript"]:
            if tx.get("is_canonical"):
                info["ensembl_transcript"] = tx["id"]
                if tx.get("version"):
                    info["ensembl_transcript"] = f"{tx['id']}.{tx['version']}"
                break
        if "ensembl_transcript" not in info and data["Transcript"]:
            # Fall back to first transcript
            tx = data["Transcript"][0]
            info["ensembl_transcript"] = tx["id"]

    # 2. Fetch RefSeq ID via xrefs
    tx_base = info.get("ensembl_transcript", "").split(".")[0]
    if tx_base:
        xrefs = _ensembl_get(f"/xrefs/id/{tx_base}", {"external_db": "RefSeq_mRNA"})
        if xrefs:
            for xref in xrefs:
                if xref.get("dbname") == "RefSeq_mRNA":
                    info["refseq_id"] = xref["display_id"]
                    break

    # 3. Fetch protein sequence
    if tx_base:
        seq_data = _ensembl_get(f"/sequence/id/{tx_base}", {"type": "protein"})
        if seq_data and seq_data.get("seq"):
            info["protein_sequence"] = seq_data["seq"]
            info["protein_seq_id"] = seq_data.get("id", tx_base)

    # 4. Fetch CDS (DNA) sequence
    if tx_base:
        seq_data = _ensembl_get(f"/sequence/id/{tx_base}", {"type": "cds"})
        if seq_data and seq_data.get("seq"):
            info["dna_sequence"] = seq_data["seq"]
            info["dna_seq_id"] = seq_data.get("id", tx_base)

    return info


def _try_local_fasta(gene: str, seq_type: str) -> str | None:
    """Check for a local FASTA file for this gene."""
    fasta_dir = PROTEIN_FASTA_DIR if seq_type == "protein" else DNA_FASTA_DIR
    for ext in (".fasta", ".fa", ".faa"):
        path = fasta_dir / f"{gene}{ext}"
        if path.exists():
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            seq_lines = [l for l in lines if not l.startswith(">")]
            return "".join(seq_lines)
    return None


def ensure_gene_metadata(genes: list[str]) -> None:
    """Fetch and cache metadata for all genes not already in the cache.

    Call this once at pipeline startup with the list of genes from the data.
    """
    _load_disk_cache()

    missing = [g for g in genes if g not in _cache or "protein_sequence" not in _cache.get(g, {})]
    if not missing:
        logger.info("Gene metadata: all %d genes already cached", len(genes))
        return

    logger.info("Gene metadata: fetching %d / %d genes from Ensembl...", len(missing), len(genes))
    for i, gene in enumerate(missing):
        # Try local FASTA first
        local_prot = _try_local_fasta(gene, "protein")
        local_dna = _try_local_fasta(gene, "dna")

        if gene in _cache and _cache[gene].get("protein_sequence") and local_prot is None:
            continue  # already complete

        info = _fetch_gene_info(gene)
        if local_prot:
            info["protein_sequence"] = local_prot
        if local_dna:
            info["dna_sequence"] = local_dna

        _cache.setdefault(gene, {}).update(info)
        if (i + 1) % 10 == 0:
            logger.info("  Fetched %d / %d genes", i + 1, len(missing))

    _save_disk_cache()
    logger.info("Gene metadata: %d genes now cached", len(_cache))


def get_canonical_transcript(gene: str) -> str | None:
    """Return the Ensembl canonical transcript ID for a gene."""
    _load_disk_cache()
    return _cache.get(gene, {}).get("ensembl_transcript")


def get_refseq_id(gene: str) -> str | None:
    """Return the RefSeq transcript ID for a gene."""
    _load_disk_cache()
    return _cache.get(gene, {}).get("refseq_id")


def get_protein_sequence(gene: str) -> str | None:
    """Return the canonical protein sequence for a gene."""
    _load_disk_cache()
    return _cache.get(gene, {}).get("protein_sequence")


def get_dna_sequence(gene: str) -> str | None:
    """Return the CDS DNA sequence for a gene."""
    _load_disk_cache()
    return _cache.get(gene, {}).get("dna_sequence")


def get_all_metadata(gene: str) -> dict:
    """Return all cached metadata for a gene."""
    _load_disk_cache()
    return _cache.get(gene, {})
