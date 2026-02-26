"""Loader for VampSEQ data.

VampSEQ is a variant-to-phenotype mapping assay (not yet acquired).
Expected schema (to be confirmed): variant identifiers, functional scores,
and potentially cell-type or assay condition metadata — likely similar to
LabelSEQ or FisSEQ in structure.

TODO: Implement once VampSEQ data is available.
- Inspect variant format (likely protein-level compact notation or HGVS)
- Identify primary functional score column
- Assign gene from metadata or filename
- Map variants to genomic coordinates using same approach as LabelSEQ/FisSEQ
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_vampseq(filepath: str | Path) -> pd.DataFrame:
    """Loader for VampSEQ data.

    Raises :class:`NotImplementedError` until the dataset is available and
    the schema has been inspected.

    Parameters
    ----------
    filepath:
        Path to a VampSEQ CSV/TSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with unified schema columns including at minimum:
        ``gene``, ``aa_ref``, ``aa_pos``, ``aa_alt``,
        ``functional_score_vampseq``, ``variant_type_harmonized``,
        ``_source``, ``source_file``.
    """
    raise NotImplementedError(
        "VampSEQ data has not been acquired yet. "
        "Implement this loader once the dataset is available and its schema "
        "has been inspected. Expected outputs: gene, aa_ref, aa_pos, aa_alt, "
        "functional_score_vampseq, variant_type_harmonized."
    )
