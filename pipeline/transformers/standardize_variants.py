"""Lift protein-level variants to genomic coordinates.

FisSEQ / LabelSEQ data only have (gene, aa_pos, aa_ref, aa_alt).  This
transformer maps them to genomic coordinates using Ensembl's variant
recoder API, enabling the outer-join merge on ``genomic_coord``.

For Pillar rows that already have ``genomic_coord``, this is a no-op.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from pipeline.config import CACHE_DIR
from pipeline.scorers.base import batch_iter
from pipeline.utils.gene_metadata import get_canonical_transcript

logger = logging.getLogger(__name__)

_CACHE_PATH = CACHE_DIR / "protein_to_genomic.parquet"
_ENSEMBL_BASE = "https://rest.ensembl.org"

# Amino acid 1-letter → 3-letter (for HGVS notation)
_AA1TO3: dict[str, str] = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    "*": "Ter", "X": "Xaa", "-": "del",
}


def _build_hgvs_p(gene: str, aa_ref: str, aa_pos: int, aa_alt: str) -> str | None:
    """Build HGVS protein string for Ensembl lookup."""
    tx = get_canonical_transcript(gene)
    if not tx:
        return None
    tx_base = tx.split(".")[0]

    ref3 = _AA1TO3.get(str(aa_ref), str(aa_ref))
    if str(aa_alt) == "-":
        return f"{tx_base}:p.{ref3}{int(aa_pos)}del"
    if str(aa_alt) in ("*", "X"):
        return f"{tx_base}:p.{ref3}{int(aa_pos)}Ter"
    alt3 = _AA1TO3.get(str(aa_alt), str(aa_alt))
    if ref3 == alt3:
        return f"{tx_base}:p.{ref3}{int(aa_pos)}="  # synonymous
    return f"{tx_base}:p.{ref3}{int(aa_pos)}{alt3}"


def _load_cache() -> dict[str, tuple[str, int, str, str]]:
    """Load cached protein→genomic mappings.

    Returns { "GENE:A123V": (chrom, pos, ref, alt) }
    """
    if _CACHE_PATH.exists():
        df = pd.read_parquet(_CACHE_PATH)
        return {
            row["lookup_key"]: (row["chrom"], int(row["hg38_pos"]), row["ref_allele"], row["alt_allele"])
            for _, row in df.iterrows()
        }
    return {}


def _save_cache(mapping: dict[str, tuple[str, int, str, str]]) -> None:
    """Persist protein→genomic mappings to parquet."""
    if not mapping:
        return
    rows = []
    for key, (chrom, pos, ref, alt) in mapping.items():
        rows.append({"lookup_key": key, "chrom": chrom, "hg38_pos": pos, "ref_allele": ref, "alt_allele": alt})
    df = pd.DataFrame(rows)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE_PATH, index=False)


def _query_variant_recoder_batch(hgvs_list: list[str]) -> dict[str, dict]:
    """POST batch variant recoder to Ensembl.

    Returns {hgvs_input: {"chrom": ..., "pos": ..., "ref": ..., "alt": ...}}
    """
    import requests

    results: dict[str, dict] = {}
    url = f"{_ENSEMBL_BASE}/variant_recoder/human"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    for batch in batch_iter(hgvs_list, 200):
        try:
            time.sleep(0.5)
            resp = requests.post(
                url, headers=headers,
                json={"ids": batch},
                timeout=60,
            )
            if resp.status_code == 429:
                retry = float(resp.headers.get("Retry-After", 5))
                logger.warning("Ensembl rate limit — waiting %.1fs", retry)
                time.sleep(retry)
                resp = requests.post(url, headers=headers, json={"ids": batch}, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            for item in data:
                if isinstance(item, dict):
                    input_id = item.get("input")
                    spdi_list = item.get("spdi", [])
                    if input_id and spdi_list:
                        # Parse first SPDI: "NC_000017.11:43094845:G:A"
                        spdi = spdi_list[0] if isinstance(spdi_list, list) else str(spdi_list)
                        parts = str(spdi).split(":")
                        if len(parts) == 4:
                            # Convert RefSeq accession to chr
                            chrom_acc = parts[0]
                            pos = int(parts[1]) + 1  # SPDI is 0-based
                            ref = parts[2]
                            alt = parts[3]
                            # Map NC accession to chr number
                            chrom = _nc_to_chr(chrom_acc)
                            if chrom:
                                results[input_id] = {
                                    "chrom": chrom, "pos": pos,
                                    "ref": ref, "alt": alt,
                                }
        except Exception:
            logger.debug("Variant recoder batch failed for %d variants", len(batch))

    return results


def _nc_to_chr(nc_acc: str) -> str | None:
    """Convert NC_ accession to chr label (e.g. NC_000017.11 → chr17)."""
    # Extract chromosome number from NC_XXXXNN.version
    if not nc_acc.startswith("NC_"):
        return None
    num_part = nc_acc.split(".")[0].replace("NC_", "").lstrip("0")
    if not num_part:
        return None
    n = int(num_part)
    if 1 <= n <= 22:
        return f"chr{n}"
    if n == 23:
        return "chrX"
    if n == 24:
        return "chrY"
    return None


def standardize_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Lift protein-level variants to genomic coordinates where missing.

    Only operates on rows where ``genomic_coord`` is null but ``gene``,
    ``aa_pos``, ``aa_ref``, and ``aa_alt`` are all present.
    """
    needs_lift = (
        df["genomic_coord"].isna()
        & df["gene"].notna()
        & df["aa_pos"].notna()
        & df["aa_ref"].notna()
        & df["aa_alt"].notna()
    )
    n_need = needs_lift.sum()
    if n_need == 0:
        logger.info("Coordinate lifting: no rows need lifting")
        return df

    logger.info("Coordinate lifting: %d rows need protein→genomic mapping", n_need)

    # Load cache
    cached = _load_cache()
    new_mappings: dict[str, tuple[str, int, str, str]] = {}

    # Build lookup keys and HGVS strings for uncached rows
    todo_rows = df.loc[needs_lift].copy()
    hgvs_to_key: dict[str, tuple[str, int]] = {}  # hgvs_string -> (lookup_key, df_index)

    uncached_hgvs = []
    for idx, row in todo_rows.iterrows():
        gene = str(row["gene"])
        aa_ref = str(row["aa_ref"])
        aa_pos = int(row["aa_pos"])
        aa_alt = str(row["aa_alt"])
        key = f"{gene}:{aa_ref}{aa_pos}{aa_alt}"

        if key in cached:
            chrom, pos, ref, alt = cached[key]
            df.at[idx, "chrom"] = chrom
            df.at[idx, "hg38_pos"] = pos
            df.at[idx, "ref_allele"] = ref
            df.at[idx, "alt_allele"] = alt
            df.at[idx, "genomic_coord"] = f"{chrom}:{pos}:{ref}:{alt}"
            df.at[idx, "coord_mapping_method"] = "protein_lifted"
            continue

        hgvs = _build_hgvs_p(gene, aa_ref, aa_pos, aa_alt)
        if hgvs and hgvs not in hgvs_to_key:
            hgvs_to_key[hgvs] = key
            uncached_hgvs.append(hgvs)

    n_from_cache = n_need - len(uncached_hgvs)
    if n_from_cache > 0:
        logger.info("Coordinate lifting: %d resolved from cache", n_from_cache)

    if not uncached_hgvs:
        logger.info("Coordinate lifting: all resolved from cache")
        return df

    logger.info("Coordinate lifting: querying Ensembl for %d unique variants...", len(uncached_hgvs))
    api_results = _query_variant_recoder_batch(uncached_hgvs)
    logger.info("Coordinate lifting: %d / %d resolved by Ensembl", len(api_results), len(uncached_hgvs))

    # Map results back
    for hgvs, result in api_results.items():
        key = hgvs_to_key.get(hgvs)
        if key:
            mapping = (result["chrom"], result["pos"], result["ref"], result["alt"])
            new_mappings[key] = mapping
            cached[key] = mapping

    # Apply new mappings to dataframe
    applied = 0
    for idx, row in todo_rows.iterrows():
        if df.at[idx, "genomic_coord"] is not pd.NA and pd.notna(df.at[idx, "genomic_coord"]):
            continue  # already resolved from cache above
        gene = str(row["gene"])
        aa_ref = str(row["aa_ref"])
        aa_pos = int(row["aa_pos"])
        aa_alt = str(row["aa_alt"])
        key = f"{gene}:{aa_ref}{aa_pos}{aa_alt}"

        if key in cached:
            chrom, pos, ref, alt = cached[key]
            df.at[idx, "chrom"] = chrom
            df.at[idx, "hg38_pos"] = pos
            df.at[idx, "ref_allele"] = ref
            df.at[idx, "alt_allele"] = alt
            df.at[idx, "genomic_coord"] = f"{chrom}:{pos}:{ref}:{alt}"
            df.at[idx, "coord_mapping_method"] = "protein_lifted"
            applied += 1

    # Save updated cache
    _save_cache(cached)

    total_with_coord = df["genomic_coord"].notna().sum()
    logger.info(
        "Coordinate lifting complete: %d newly lifted, %d / %d total with genomic_coord",
        applied, total_with_coord, len(df),
    )
    return df
