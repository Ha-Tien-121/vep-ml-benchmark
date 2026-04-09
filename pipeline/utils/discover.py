"""Auto-discover datasets in the Data/ directory.

Supports two modes:
1. **Manifest-driven**: reads ``Data/manifest.yaml`` for explicit file→source mappings
2. **Convention-driven**: detects source type by column signature (fallback)

Adding a new dataset requires only placing the file in Data/ and optionally
updating manifest.yaml — no code changes needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetEntry:
    """A single discovered dataset file."""
    path: Path
    source_type: str  # "pillar", "fisseq_supp", "fisseq_features", "labelseq", "vampseq"
    gene: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DiscoveredDatasets:
    """All discovered datasets, grouped by source type."""
    pillar: list[DatasetEntry] = field(default_factory=list)
    fisseq_supp: list[DatasetEntry] = field(default_factory=list)
    fisseq_features: list[DatasetEntry] = field(default_factory=list)
    labelseq: list[DatasetEntry] = field(default_factory=list)
    vampseq: list[DatasetEntry] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        for name in ("pillar", "fisseq_supp", "fisseq_features", "labelseq", "vampseq"):
            entries = getattr(self, name)
            if entries:
                files = ", ".join(e.path.name for e in entries)
                parts.append(f"  {name}: {len(entries)} files ({files})")
        return "\n".join(parts) if parts else "  (no datasets found)"


# ── Column signatures for auto-detection ─────────────────────────────

_PILLAR_SIGNATURE = {"ID", "Gene", "hg38_start", "ref_allele", "alt_allele"}
_FISSEQ_SUPP_SIGNATURE = {"Variant", "Morphological Impact Score"}
_FISSEQ_SUPP2_SIGNATURE = {"Variant", "Morphological Impact Score iPSC"}
_FISSEQ_FEATURES_SIGNATURE = {"Variant", "Variant_Class"}
_LABELSEQ_SIGNATURE = {"Wild Type Residue", "Position", "Mutation"}
_VAMPSEQ_SIGNATURE = {"variant", "vampseq_score"}  # placeholder


def _detect_source_type(filepath: Path) -> str | None:
    """Detect source type by reading the first row and checking column signatures."""
    try:
        if filepath.suffix == ".gz":
            df = pd.read_csv(filepath, nrows=1, compression="gzip", low_memory=False)
        else:
            df = pd.read_csv(filepath, nrows=1, low_memory=False)
        cols = set(df.columns)
    except Exception:
        logger.debug("Cannot read %s for signature detection", filepath.name)
        return None

    if _PILLAR_SIGNATURE.issubset(cols):
        return "pillar"
    # Check features BEFORE supp (features have Variant_Class which supp lacks)
    if _FISSEQ_FEATURES_SIGNATURE.issubset(cols):
        return "fisseq_features"
    if _FISSEQ_SUPP2_SIGNATURE.issubset(cols):
        return "fisseq_supp"
    if _FISSEQ_SUPP_SIGNATURE.issubset(cols):
        return "fisseq_supp"
    if _LABELSEQ_SIGNATURE.issubset(cols):
        return "labelseq"
    if _VAMPSEQ_SIGNATURE.issubset(cols):
        return "vampseq"

    return None


def _infer_gene_from_filename(name: str) -> str | None:
    """Try to extract gene name from filename."""
    name_lower = name.lower()
    # Common gene names that appear in filenames
    known_genes = [
        "BRCA1", "BRCA2", "LMNA", "PTEN", "TP53", "MSH2", "VHL",
        "KCNH2", "CBS", "HMBS", "GCK", "BAP1", "F9", "ASPA", "PAX6",
        "G6PD", "TSC2", "DDX3X", "BARD1", "CHEK2", "RAD51C", "PALB2",
        "CARD11", "KCNE1", "KCNQ4", "LARGE1", "SGCB", "NDUFAF6", "CRX",
        "CTCF", "TPK1", "FKRP", "RAD51D", "XRCC2", "JAG1", "SFPQ",
        "OTC", "TARDBP", "SCN5A", "RHO",
    ]
    for gene in known_genes:
        if gene.lower() in name_lower:
            return gene
    return None


def discover_datasets(
    data_dir: Path,
    manifest_path: Path | None = None,
) -> DiscoveredDatasets:
    """Discover datasets in data_dir via manifest or column-signature detection.

    Parameters
    ----------
    data_dir : Path
        Directory containing input data files.
    manifest_path : Path, optional
        Path to manifest.yaml.  If None, looks for ``data_dir/manifest.yaml``.
    """
    result = DiscoveredDatasets()

    # Try manifest first
    manifest_file = manifest_path or (data_dir / "manifest.yaml")
    if manifest_file.exists():
        result = _load_manifest(manifest_file, data_dir)
        logger.info("Discovered datasets from manifest:\n%s", result.summary())
        return result

    # Fall back to convention-based discovery
    logger.info("No manifest.yaml found — auto-discovering datasets in %s", data_dir)

    csv_files = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.csv.gz"))
    tsv_files = sorted(data_dir.glob("*.tsv")) + sorted(data_dir.glob("*.tsv.gz"))
    all_files = csv_files + tsv_files

    for filepath in all_files:
        source_type = _detect_source_type(filepath)
        if source_type is None:
            logger.debug("  Skipping %s (unknown source type)", filepath.name)
            continue

        gene = _infer_gene_from_filename(filepath.name)
        entry = DatasetEntry(path=filepath, source_type=source_type, gene=gene)
        getattr(result, source_type).append(entry)
        logger.info("  Found %s: %s (gene=%s)", source_type, filepath.name, gene)

    logger.info("Discovered datasets:\n%s", result.summary())
    return result


def _load_manifest(manifest_path: Path, data_dir: Path) -> DiscoveredDatasets:
    """Parse a manifest.yaml file into DiscoveredDatasets."""
    try:
        import yaml
    except ImportError:
        # Fall back to simple parsing if PyYAML not installed
        logger.warning("PyYAML not installed — falling back to auto-discovery")
        return DiscoveredDatasets()

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    result = DiscoveredDatasets()

    for source_type in ("pillar", "fisseq_supp", "fisseq_features", "labelseq", "vampseq"):
        section = raw.get(source_type, {})
        if not section:
            continue

        # Support both "glob" and "files" modes
        if "glob" in section:
            for path in sorted(data_dir.glob(section["glob"])):
                gene = _infer_gene_from_filename(path.name)
                entry = DatasetEntry(path=path, source_type=source_type, gene=gene)
                getattr(result, source_type).append(entry)

        if "files" in section:
            for file_spec in section["files"]:
                if isinstance(file_spec, str):
                    path = data_dir / file_spec
                    gene = _infer_gene_from_filename(file_spec)
                else:
                    path = data_dir / file_spec["path"]
                    gene = file_spec.get("gene") or _infer_gene_from_filename(file_spec["path"])
                if path.exists():
                    entry = DatasetEntry(path=path, source_type=source_type, gene=gene)
                    getattr(result, source_type).append(entry)
                else:
                    logger.warning("Manifest file not found: %s", path)

    return result
