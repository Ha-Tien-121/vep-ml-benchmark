"""Loader for LabelSEQ functional assay datasets.

LabelSEQ provides amino-acid-level DMS scores with no genomic coordinates.
Gene identity must be inferred from the ``library`` column using the mapping
in ``config.LABELSEQ_LIBRARY_TO_GENE``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pipeline.config import AA1TO3, LABELSEQ_LIBRARY_TO_GENE

logger = logging.getLogger(__name__)


def load_labelseq(filepath: str | Path) -> pd.DataFrame:
    """Read a LabelSEQ CSV and map to unified schema columns.

    The primary functional score is ``standard-adjusted score``, falling back
    to ``average score`` when the adjusted score is null.
    """
    filepath = Path(filepath)
    logger.info("Loading LabelSEQ: %s", filepath.name)

    sep = "\t" if filepath.suffix == ".tsv" else ","
    df = pd.read_csv(filepath, sep=sep, low_memory=False)
    # Drop unnamed index columns that some TSV exports include
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    logger.info("  Raw shape: %d rows x %d cols", *df.shape)

    if df.empty:
        logger.warning("  LabelSEQ file is empty: %s", filepath.name)
        return df

    # Gene inference from library column 
    if "library" in df.columns:
        df["gene"] = df["library"].map(LABELSEQ_LIBRARY_TO_GENE)
        n_unmapped = df["gene"].isna().sum()
        if n_unmapped:
            unmapped = df.loc[df["gene"].isna(), "library"].unique().tolist()
            logger.warning(
                "  %d rows have unmapped library values (gene=NaN): %s. "
                "Add them to config.LABELSEQ_LIBRARY_TO_GENE.",
                n_unmapped, unmapped,
            )
    else:
        df["gene"] = pd.NA
        logger.warning("  No 'library' column found; gene cannot be inferred")

    # Amino acid fields
    if "Wild Type Residue" in df.columns:
        df["aa_ref"] = df["Wild Type Residue"].astype(str).str.strip().str.upper()
    if "Position" in df.columns:
        df["aa_pos"] = pd.to_numeric(df["Position"], errors="coerce").astype("Int64")
    if "Mutation" in df.columns:
        df["aa_alt"] = df["Mutation"].astype(str).str.strip().str.upper()

    # HGVS-p from parts
    def _build_hgvs_p(row: pd.Series) -> str | None:
        ref = row.get("aa_ref")
        alt = row.get("aa_alt")
        pos = row.get("aa_pos")
        if not ref or not alt or pd.isna(pos):
            return None
        ref3 = AA1TO3.get(ref, ref)
        alt3 = AA1TO3.get(alt, alt)
        return f"p.{ref3}{int(pos)}{alt3}"

    df["hgvs_p"] = df.apply(_build_hgvs_p, axis=1)

    # Variant type
    if "Mutation Type" in df.columns:
        from pipeline.config import VARIANT_TYPE_MAP
        df["variant_type_harmonized"] = (
            df["Mutation Type"].map(VARIANT_TYPE_MAP).fillna("other")
        )
    else:
        df["variant_type_harmonized"] = "other"

    # Source-specific named scores (preserved separately for downstream use)
    if "standard-adjusted score" in df.columns:
        df["functional_score_labelseq_standard"] = pd.to_numeric(
            df["standard-adjusted score"], errors="coerce"
        )
    else:
        df["functional_score_labelseq_standard"] = pd.NA

    if "intercept_0_standard-adjusted score" in df.columns:
        df["functional_score_labelseq_intercept"] = pd.to_numeric(
            df["intercept_0_standard-adjusted score"], errors="coerce"
        )
    else:
        df["functional_score_labelseq_intercept"] = pd.NA

    # Primary functional score — prefer standard-adjusted, then intercept-adjusted, then average
    _score_cols = [
        "standard-adjusted score",
        "intercept_0_standard-adjusted score",
        "average score",
    ]
    df["functional_score_labelseq"] = pd.NA
    for _sc in _score_cols:
        if _sc in df.columns:
            parsed = pd.to_numeric(df[_sc], errors="coerce")
            mask = df["functional_score_labelseq"].isna()
            df.loc[mask, "functional_score_labelseq"] = parsed[mask]

    # Preserve supplementary columns
    rename_supp: dict[str, str] = {}
    for col in ("Number of Barcodes", "variant_frequency",
                "score_1", "score_2", "score_3",
                "date", "library", "assay", "variant",
                "assay_treatment", "classification_2.5pct",
                "intercept_0_standard-adjusted score",
                "intercept_0_std_adj_score_1",
                "intercept_0_std_adj_score_2",
                "intercept_0_std_adj_score_3",
                "protein", "uniprot_accession"):
        if col in df.columns:
            safe = col.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
            rename_supp[col] = f"labelseq_{safe}"
    # Also rename raw columns that haven't been mapped yet
    for col in ("Wild Type Residue", "Position", "Mutation",
                "Mutation Type", "average score", "standard-adjusted score"):
        if col in df.columns and col not in rename_supp:
            rename_supp[col] = f"labelseq_raw_{col.lower().replace(' ', '_').replace('-', '_')}"
    df = df.rename(columns=rename_supp)

    # Convenience alias
    if "labelseq_number_of_barcodes" in df.columns:
        df["labelseq_n_barcodes"] = pd.to_numeric(
            df["labelseq_number_of_barcodes"], errors="coerce"
        ).astype("Int64")

    df["coord_mapping_method"] = "protein_lifted"
    df["_source"] = "labelseq"
    df["source_file"] = filepath.name

    logger.info(
        "  Loaded %d rows; %d with gene; %d with functional score",
        len(df),
        df["gene"].notna().sum(),
        df["functional_score_labelseq"].notna().sum(),
    )
    return df
