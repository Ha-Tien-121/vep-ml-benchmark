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
    "final_pillar_data_with_clinvar_18_25_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_gold_standards_condensed_111225.csv.gz",
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
    "250601_LABELseq_scores(2).tsv",
    "Replicate_scores_IGVF_submission_260225.tsv",
]

VAMPSEQ_FILES: list[str] = [
    "urn_mavedb_00001266-b-1_custom.csv",
    "urn_mavedb_00001267-a-1_custom.csv",
    "urn_mavedb_00001267-b-1_custom.csv",
    "2026025_TSC2_lib1_scores_with_consequences.csv",
    "2026025_TSC2_lib2_scores_with_consequences.csv",
    "20260217_TSC2_lib1_scores.csv",
    "20260217_TSC2_lib2_scores.csv",
]

VAMPSEQ_REFSEQ_TO_GENE: dict[str, str] = {
    "NP_001346945.1": "G6PD",
    "NP_000539.2": "TSC2",
}

GENOME_BUILD = "GRCh38"

CANONICAL_TRANSCRIPTS: dict[str, str] = {
    "BRCA1": "ENST00000357654.9",
    "LMNA": "ENST00000368300.9",
    "PTEN": "ENST00000371953.8",
    "G6PD": "ENST00000393562.5",
    "TSC2": "ENST00000219476.9",
    "KRAS": "ENST00000256078.9",
    "ARAF": "ENST00000264025.7",
    "BRAF": "ENST00000288602.11",
    "RAF1": "ENST00000251849.9",
    "MAP2K1": "ENST00000307102.8",
    "MAP2K2": "ENST00000262948.5",
    "EGFR": "ENST00000275493.7",
    "ERBB2": "ENST00000269571.10",
    "SOS1": "ENST00000402219.3",
    "SOS2": "ENST00000263979.7",
    "PTPN11": "ENST00000351677.8",
    "GRB2": "ENST00000309845.9",
    "KSR1": "ENST00000262174.6",
    "KSR2": "ENST00000358624.8",
    "MET": "ENST00000397752.8",
    "MRAS": "ENST00000394102.5",
    "RET": "ENST00000355710.8",
}

CANONICAL_REFSEQ: dict[str, str] = {
    "BRCA1": "NM_007294.3",
    "LMNA": "NM_170707.4",
    "PTEN": "NM_000314.8",
    "G6PD": "NM_001360016.2",
    "TSC2": "NM_000548.5",
    "KRAS": "NM_004985.5",
    "ARAF": "NM_001654.6",
    "BRAF": "NM_004333.6",
    "RAF1": "NM_002880.4",
    "MAP2K1": "NM_002755.4",
    "MAP2K2": "NM_030662.4",
    "EGFR": "NM_005228.5",
    "ERBB2": "NM_004448.4",
    "SOS1": "NM_005633.4",
    "SOS2": "NM_006937.3",
    "PTPN11": "NM_002834.5",
    "GRB2": "NM_002086.5",
    "KSR1": "NM_014238.3",
    "KSR2": "NM_173598.3",
    "MET": "NM_000245.4",
    "MRAS": "NM_012219.4",
    "RET": "NM_020630.6",
}

LABELSEQ_LIBRARY_TO_GENE: dict[str, str] = {
    # KRAS
    "kras": "KRAS",
    "MCP-kras": "KRAS",
    # ARAF
    "araf": "ARAF",
    "araf_cterm": "ARAF",
    "araf_nterm": "ARAF",
    "MCP-araf_nterm": "ARAF",
    # BRAF
    "braf_cterm": "BRAF",
    "braf_nterm": "BRAF",
    "MCP-braf_cterm": "BRAF",
    "MCP-braf_nterm": "BRAF",
    # RAF1 (CRAF)
    "craf_cterm": "RAF1",
    "craf_nterm": "RAF1",
    "MCP-craf_cterm": "RAF1",
    "MCP-craf_nterm": "RAF1",
    # MAP2K1 (MEK1)
    "MEK1": "MAP2K1",
    "mek1": "MAP2K1",
    # MAP2K2 (MEK2)
    "mek2": "MAP2K2",
    # EGFR
    "EGFR": "EGFR",
    "EGFR-MCP": "EGFR",
    "egfr": "EGFR",
    # ERBB2
    "ERBB2-MCP": "ERBB2",
    "erbb2": "ERBB2",
    # SOS1
    "SOS1-MCP": "SOS1",
    "sos1": "SOS1",
    # SOS2
    "sos2": "SOS2",
    # PTPN11 (SHP2)
    "MCP-SHP2": "PTPN11",
    "shp2": "PTPN11",
    # GRB2
    "grb2": "GRB2",
    # KSR1
    "ksr1_cterm": "KSR1",
    "ksr1_nterm": "KSR1",
    # KSR2
    "ksr2_cterm": "KSR2",
    # MET
    "met": "MET",
    # MRAS
    "mras": "MRAS",
    # RET
    "ret": "RET",
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
    "vampseq": True,  # higher VAMP-seq score = more stable/functional (benign)
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

# ── Scorer configuration ─────────────────────────────────────────────
ALPHAMISSENSE_ZENODO_URL = (
    "https://zenodo.org/records/8208688/files/AlphaMissense_hg38.tsv.gz"
)
ALPHAMISSENSE_DB_PATH = CACHE_DIR / "alphamissense_hg38.db"

# SpliceAI: precomputed scores from Illumina BaseSpace (manual download required).
# Supports VCF (.vcf, .vcf.gz) or TSV (.tsv, .tsv.gz, .txt, .csv) inputs.
# Set to None to disable SpliceAI scoring entirely.
SPLICEAI_FILE = CACHE_DIR / "spliceai_scores.raw.snv.ensembl_mane_v1.4.grch38.vcf.gz"
SPLICEAI_DB_PATH = CACHE_DIR / "spliceai_hg38.db"

ESM_DEFAULT_MODEL = "esmc_300m"
EVO2_DEFAULT_MODEL = "evo2_7b"

SCORER_CACHE_DIR = CACHE_DIR / "scorer_results"
GENE_METADATA_CACHE_PATH = CACHE_DIR / "gene_metadata.json"

# Auto-discovery: when True, the pipeline scans Data/ for datasets by column
# signature instead of relying solely on the hardcoded file lists above.
# Set to False to use only the explicit PILLAR_FILES / FISSEQ_SUPP_FILES lists.
AUTO_DISCOVER_DATASETS = True

# ── Finetuning configuration ─────────────────────────────────────────
FINETUNING_OUTPUT_DIR = OUTPUT_DIR / "finetuning"
FINETUNING_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
FINETUNING_SEED = 42

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
    "functional_score_labelseq_standard",
    "functional_score_labelseq_intercept",
    "functional_score_fisseq",
    "functional_score_vampseq",
    "normalized_score_pillar",
    "normalized_score_labelseq",
    "normalized_score_fisseq",
    "normalized_score_vampseq",
    "consensus_functional_score",
    "consensus_functional_label",
    "alphamissense_score",
    "alphamissense_label",
    "esm3_score",
    "esm3_rank",
    "evo2_score",
    "evo2_rank",
    "spliceai_DS_AG",
    "spliceai_DS_AL",
    "spliceai_DS_DG",
    "spliceai_DS_DL",
    "spliceai_max_delta_score",
    "source_datasets",
    "coord_mapping_method",
    "is_duplicate_flag",
    "conflict_flag",
    "low_coverage_flag",
]
