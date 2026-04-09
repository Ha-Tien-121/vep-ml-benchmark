# VEP ML Benchmark Pipeline

Automated, reproducible pipeline for **Variant Effect Prediction** benchmarking. Merges functional assay data from multiple sources (Pillar, FisSEQ, LabelSEQ), attaches protein/DNA sequences, normalizes scores, and prepares data for scoring and finetuning with **ESM3**, **Evo2**, and **AlphaMissense**.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd vep-ml-benchmark
pip install -e .                    # core dependencies only
pip install -e ".[cpu]"             # + PyTorch + ESM (CPU inference)
pip install -e ".[gpu]"             # + PyTorch + ESM + Evo2 (GPU inference)
pip install -e ".[dev]"             # + pytest + ruff

# 2. Run the pipeline
python -m pipeline -v               # full pipeline (load, merge, transform, score, export)
python -m pipeline --skip-scoring   # skip model inference (useful for data prep)
```

## Pipeline Stages

```
STEP 0  Reproducibility manifest     checksums, git commit, package versions
STEP 1  Load datasets                auto-discovers files from Data/manifest.yaml
STEP 2  Validate                     checks column schemas per source
STEP 3  Merge                        outer-join on genomic_coord + protein-level fallback
STEP 4  Transform                    resolve transcripts, harmonize scores, add sequences
STEP 5  Score                        AlphaMissense, ESM, Evo2
STEP 6  Export                       Parquet, CSV, summary statistics
STEP 7  Finetuning splits            train/val/test per model format
```

## Project Structure

```
vep-ml-benchmark/
├── Data/                           # Input datasets
│   ├── manifest.yaml               # Declares which files to load (edit this)
│   └── *.csv / *.csv.gz            # Raw data files
├── data/
│   ├── cache/                      # Ensembl API cache, gene metadata, scorer results
│   ├── sequences/protein/          # Local FASTA files (optional)
│   ├── sequences/dna/              # Local FASTA files (optional)
│   └── msa/                        # Multiple sequence alignments (optional)
├── pipeline/
│   ├── main.py                     # Pipeline orchestrator
│   ├── config.py                   # All thresholds, paths, mappings
│   ├── loaders/                    # Per-source data loaders
│   │   ├── load_pillar.py          # Pillar backbone (genomic coords, scores)
│   │   ├── load_fisseq.py          # FisSEQ (protein-level morphology)
│   │   ├── load_labelseq.py        # LabelSEQ (barcode-based assay)
│   │   └── load_vampseq.py         # VampSEQ (stub - not yet implemented)
│   ├── mergers/
│   │   └── merge_all.py            # Multi-source outer join
│   ├── transformers/
│   │   ├── resolve_transcripts.py  # Auto-fill Ensembl/RefSeq transcript IDs
│   │   ├── harmonize_scores.py     # Normalize + consensus across assays
│   │   ├── add_sequences.py        # Attach protein/DNA sequences per gene
│   │   └── standardize_variants.py # Protein-to-genomic coordinate lifting
│   ├── scorers/
│   │   ├── score_alphamissense.py  # Zenodo precomputed lookup (no GPU)
│   │   ├── score_esm.py            # ESM C 300M log-likelihood ratios
│   │   └── score_evo2.py           # Evo2 DNA log-likelihood ratios
│   ├── finetuning/
│   │   ├── splits.py               # Stratified train/val/test
│   │   ├── format_esm.py           # Protein sequence + mutant + labels
│   │   ├── format_evo2.py          # DNA coordinates + labels
│   │   └── format_alphamissense.py # AM features + labels
│   ├── validators/
│   │   └── validate_schema.py      # Input/output schema checks
│   ├── utils/
│   │   ├── gene_metadata.py        # Auto-fetch transcripts from Ensembl
│   │   ├── discover.py             # Dataset auto-discovery
│   │   └── manifest.py             # Reproducibility manifest
│   ├── outputs/
│   │   └── export.py               # Parquet/CSV/summary export
│   └── tests/
│       └── test_pipeline.py
├── outputs/                        # Generated outputs
│   ├── benchmark_dataframe.parquet
│   ├── benchmark_dataframe.csv
│   ├── summary_statistics.txt
│   ├── pipeline_manifest.json
│   └── finetuning/                 # When --skip-finetuning is not set
│       ├── esm/{train,val,test}.parquet
│       ├── evo2/{train,val,test}.parquet
│       └── alphamissense/{train,val,test}.parquet
├── pyproject.toml
├── requirements.txt
└── README.md
```

## CLI Reference

```
python -m pipeline [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | `./Data` | Input data directory |
| `--output-dir PATH` | `./outputs` | Output directory |
| `-v, --verbose` | off | Debug-level logging |
| `--skip-scoring` | off | Skip model scoring (Step 5) |
| `--skip-finetuning` | off | Skip finetuning splits (Step 7) |
| `--skip-transforms` | off | Skip transcript/score/sequence transforms (Step 4) |
| `--scorers LIST` | `alphamissense,esm,evo2` | Comma-separated scorer subset |
| `--esm-model NAME` | `esmc_300m` | ESM model variant (`esmc_300m`, `esmc_600m`, `esm3_small`) |
| `--evo2-model NAME` | `evo2_7b` | Evo2 model variant |
| `--device DEVICE` | auto-detect | Force `cpu`, `cuda`, or `mps` |

### Common Workflows

```bash
# Data prep only (no models needed)
python -m pipeline --skip-scoring --skip-finetuning -v

# AlphaMissense only (no GPU needed, downloads 4GB from Zenodo on first run)
python -m pipeline --scorers alphamissense --skip-finetuning

# Full pipeline with ESM on CPU
python -m pipeline --device cpu --scorers alphamissense,esm

# Export finetuning data (requires transforms to have run)
python -m pipeline --skip-scoring
```

## Adding New Datasets

### Option 1: Manifest (recommended)

1. Place your CSV/TSV file in `Data/`
2. Add an entry to `Data/manifest.yaml`:

```yaml
pillar:
  files:
    - path: "your_new_dataset.csv.gz"

fisseq_supp:
  files:
    - path: "new_fisseq_data.csv"
      gene: YOUR_GENE
```

3. Run `python -m pipeline -v`

### Option 2: Auto-discovery (no manifest)

If `Data/manifest.yaml` does not exist, the pipeline detects source type by column signatures:

| Source Type | Required Columns |
|-------------|-----------------|
| Pillar | `ID`, `Gene`, `hg38_start`, `ref_allele`, `alt_allele` |
| FisSEQ (supp) | `Variant`, `Morphological Impact Score` |
| FisSEQ (features) | `Variant`, `Variant_Class` |
| LabelSEQ | `Wild Type Residue`, `Position`, `Mutation` |

Just drop a CSV in `Data/` with the right columns and re-run.

### Adding New Genes

No code changes needed. The pipeline auto-fetches canonical transcripts, RefSeq IDs, and protein/DNA sequences from Ensembl for **any gene** present in the data. Results are cached to `data/cache/gene_metadata.json`.

## Model Scoring

### AlphaMissense (no GPU needed)

Uses precomputed pathogenicity scores from [Zenodo](https://zenodo.org/records/8208688). On first run, downloads the 4 GB TSV and converts it to an indexed SQLite database for fast lookups. Subsequent runs are instant.

### ESM (CPU or GPU)

Uses [ESM C 300M](https://github.com/evolutionaryscale/esm) by default (~600 MB, runs on CPU). Computes log-likelihood ratios (LLR) for each variant: a more negative LLR indicates a more damaging substitution.

| Model | Parameters | VRAM | CPU OK? |
|-------|-----------|------|---------|
| `esmc_300m` | 300M | ~1 GB | Yes (slower) |
| `esmc_600m` | 600M | ~2 GB | Yes (slower) |
| `esm3_small` | 1.4B | ~16 GB | No |

```bash
pip install esm torch
python -m pipeline --scorers esm --esm-model esmc_300m --device cpu
```

### Evo2 (GPU required)

Uses [Evo2](https://github.com/arcinstitute/evo2) for DNA-level variant scoring. Requires a GPU with sufficient VRAM. Gracefully skips if no GPU is available.

```bash
pip install evo2 torch
python -m pipeline --scorers evo2 --device cuda
```

## Finetuning Data

When `--skip-finetuning` is not set, the pipeline produces stratified train/val/test splits (70/15/15) for each model:

| Directory | Format | Key Columns |
|-----------|--------|-------------|
| `outputs/finetuning/esm/` | Parquet | `sequence_protein`, `mutant_sequence`, `aa_pos`, `aa_ref`, `aa_alt`, `label_continuous` |
| `outputs/finetuning/evo2/` | Parquet | `chrom`, `hg38_pos`, `ref_allele`, `alt_allele`, `sequence_dna`, `label_continuous` |
| `outputs/finetuning/alphamissense/` | Parquet | `alphamissense_score`, `aa_pos`, `aa_ref`, `aa_alt`, `label_continuous` |

Splits are stratified by gene and consensus functional label to ensure balanced representation.

## Output Schema

The benchmark dataframe contains 170+ columns. Key fields:

| Category | Columns |
|----------|---------|
| **Variant ID** | `variant_id`, `gene`, `genomic_coord`, `chrom`, `hg38_pos`, `ref_allele`, `alt_allele` |
| **Protein** | `aa_pos`, `aa_ref`, `aa_alt`, `hgvs_p`, `variant_type_harmonized` |
| **Sequences** | `sequence_protein`, `sequence_dna`, `sequence_id_protein`, `sequence_id_dna` |
| **Functional scores** | `functional_score_pillar`, `functional_score_fisseq`, `functional_score_labelseq` |
| **Consensus** | `consensus_functional_score` (0-1), `consensus_functional_label` (functional / intermediate / loss_of_function) |
| **Predictors** | `alphamissense_score`, `alphamissense_label`, `esm3_score`, `esm3_rank`, `evo2_score`, `evo2_rank` |
| **QC** | `conflict_flag`, `is_duplicate_flag`, `low_coverage_flag`, `source_datasets` |

## Reproducibility

Every pipeline run generates `outputs/pipeline_manifest.json` containing:
- SHA256 checksums of all input files
- Git commit hash and branch
- Python version and package versions
- Timestamp

This allows any output to be traced back to the exact inputs and code that produced it.

## Configuration

All thresholds, paths, and mappings are centralized in `pipeline/config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CONSENSUS_LOF_THRESHOLD` | 0.3 | Score <= this = loss-of-function |
| `CONSENSUS_FUNC_THRESHOLD` | 0.7 | Score >= this = functional |
| `CONFLICT_SCORE_THRESHOLD` | 0.4 | Flag if cross-source range exceeds this |
| `MIN_FISSEQ_CELLS` | 100 | QC flag if total cells below this |
| `MIN_LABELSEQ_BARCODES` | 10 | QC flag if barcode count below this |
| `AUTO_DISCOVER_DATASETS` | True | Use column-signature detection vs explicit file lists |

## Development

```bash
pip install -e ".[dev]"
pytest                              # run tests
ruff check pipeline/                # lint
```

## License

MIT
