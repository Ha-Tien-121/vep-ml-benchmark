"""Central configuration for the VEP benchmark data pipeline.

Every threshold, path, gene mapping, score orientation flag, and controlled
vocabulary lives here so that no magic constants leak into transformation code.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "Data"
OUTPUT_DIR = REPO_ROOT / "outputs"
PROTEIN_FASTA_DIR = REPO_ROOT / "data" / "sequences" / "protein"
DNA_FASTA_DIR = REPO_ROOT / "data" / "sequences" / "dna"
MSA_DIR = REPO_ROOT / "data" / "msa"
CACHE_DIR = REPO_ROOT / "data" / "cache"

PILLAR_FILES: list[str] = [
    "BRCA1_pillar_data.csv",
]

FISSEQ_SUPP_FILES: dict[str, str] = {
    "SupplementaryTable1.csv": "LMNA",
    "SupplementaryTable2.csv": "PTEN",
}

FISSEQ_FEATURE_FILES: dict[str, str] = {
    "LMNA_averaged_medianplusEMD_clusterlabeled_021325.csv": "LMNA",
    "PTENT3.merged.consensusprofiles.clustered.011725.csv": "PTEN",
}

LABELSEQ_FILES: list[str] = [
    "LabelSEQ-example - Sheet1.csv",
]

GENOME_BUILD = "GRCh38"

CANONICAL_TRANSCRIPTS: dict[str, str] = {
    "BRCA1": "ENST00000357654.9",
    "LMNA": "ENST00000368300.9",
    "PTEN": "ENST00000371953.8",
}

CANONICAL_REFSEQ: dict[str, str] = {
    "BRCA1": "NM_007294.3",
    "LMNA": "NM_170707.4",
    "PTEN": "NM_000314.8",
}

LABELSEQ_LIBRARY_TO_GENE: dict[str, str] = {
    # "library_name": "GENE_SYMBOL",
}

FISSEQ_FILE_TO_GENE: dict[str, str] = {
    "SupplementaryTable1.csv": "LMNA",
    "SupplementaryTable2.csv": "PTEN",
    "LMNA_averaged_medianplusEMD_clusterlabeled_021325.csv": "LMNA",
    "PTENT3.merged.consensusprofiles.clustered.011725.csv": "PTEN",
}

# Functional score orientation
#   True  → higher value = more functional (benign)
#   False → higher value = more dysfunctional (pathogenic) — will be inverted
SCORE_ORIENTATION: dict[str, bool | None] = {
    "pillar": True,
    "labelseq": True,
    "fisseq": False,
    "vampseq": None,
}

CONSENSUS_LOF_THRESHOLD = 0.3
CONSENSUS_FUNC_THRESHOLD = 0.7

CONFLICT_SCORE_THRESHOLD = 0.4

# Minimum coverage for QC flags
MIN_LABELSEQ_BARCODES = 10
MIN_FISSEQ_CELLS = 100

VARIANT_TYPE_MAP: dict[str, str] = {
    "Missense": "missense",
    "missense": "missense",
    "missense_variant": "missense",
    "Synonymous": "synonymous",
    "synonymous": "synonymous",
    "synonymous_variant": "synonymous",
    "Frameshift": "frameshift",
    "frameshift": "frameshift",
    "frameshift_variant": "frameshift",
    "Nonsense": "nonsense",
    "nonsense": "nonsense",
    "stop_gained": "nonsense",
    "3nt Deletion": "in_frame_indel",
    "in_frame_indel": "in_frame_indel",
    "in_frame_deletion": "in_frame_indel",
    "in_frame_insertion": "in_frame_indel",
    "Splice region": "splicing_variant",
    "Canonical splice": "splicing_variant",
    "splice_site_variant": "splicing_variant",
    "splicing_variant": "splicing_variant",
    "Other": "other",
    "other": "other",
}

AA3TO1: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U",
}

AA1TO3: dict[str, str] = {v: k for k, v in AA3TO1.items()}

REQUIRED_COLUMNS: list[str] = [
    "variant_id",
    "gene",
    "genomic_coord",
    "chrom",
    "hg38_pos",
    "ref_allele",
    "alt_allele",
    "aa_pos",
    "aa_ref",
    "aa_alt",
    "hgvs_p",
    "simplified_consequence",
    "variant_type_harmonized",
    "sequence_id_protein",
    "sequence_protein",
    "sequence_id_dna",
    "sequence_dna",
    "sequence_type",
    "msa_filepath",
    "functional_score_pillar",
    "functional_score_labelseq",
    "functional_score_fisseq",
    "functional_score_vampseq",
    "consensus_functional_score",
    "consensus_functional_label",
    "alphamissense_score",
    "alphamissense_label",
    "esm3_score",
    "esm3_rank",
    "evo2_score",
    "evo2_rank",
    "source_datasets",
    "coord_mapping_method",
    "is_duplicate_flag",
    "conflict_flag",
    "low_coverage_flag",
]
