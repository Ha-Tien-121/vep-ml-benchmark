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

## Project Structure


## CLI Reference

```
python -m pipeline [OPTIONS]


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


### Adding New Genes

No code changes needed. The pipeline auto-fetches canonical transcripts, RefSeq IDs, and protein/DNA sequences from Ensembl for **any gene** present in the data. Results are cached to `data/cache/gene_metadata.json`.


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


## Configuration

All thresholds, paths, and mappings are centralized in `pipeline/config.py`:


## License

MIT
