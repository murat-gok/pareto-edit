# Reproducibility

This document lists the exact commands to reproduce every result in the manuscript
"PARETO-EDIT: multi-objective optimization of base-editing guide design". All commands are
run from the repository root. Random seeds are fixed; fetched genomic windows are cached.

## 0. Environment

```bash
git clone https://github.com/murat-gok/pareto-edit.git
cd pareto-edit
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
```

Core dependencies (in `requirements.txt`): numpy, scipy, pandas, matplotlib, scikit-learn, pymoo, requests.

Optional, for the real-model biological analysis only:
- PyTorch and torchvision (BE-Hive bystander model)
- the BE-Hive `be_predict_bystander` repository (cloned separately)
- `variant_summary.txt.gz` from ClinVar (GRCh38)
- internet access for the Ensembl REST API (first run only; results are cached)

## 1. Minimal example (no downloads, surrogate oracle)

```bash
python -m experiments.run_example --input examples/example_snvs.csv --out out.csv
```
Prints per-locus Pareto-front sizes and writes designs to `out.csv`
(columns match `examples/expected_output.csv`). Expected console output:
```
chr11:5226778A>G: Pareto front = 3 non-dominated designs
chr2:96254050G>A: Pareto front = 4 non-dominated designs
chr14:77444311G>A: Pareto front = 1 non-dominated designs
Wrote 8 designs across all loci to out.csv
```
(Design counts use the surrogate oracle and are illustrative; the real-model fronts are
produced by `experiments/layer2_application.py`.)

## 2. Layer-1: optimizer benchmark (surrogate test function)

```bash
python experiments/benchmark.py
```
Produces the optimizer comparison (NSGA-II, NSGA-III, CPO-GWO, weighted-sum,
epsilon-constraint, Pareto filter, random) over 8 problems x 30 seeds; writes
`benchmark_raw.csv`, `benchmark_summary.csv`, `benchmark_wilcoxon.csv` and the HV/efficiency
figures. Reproduces Supplementary Table S2 and the critical-difference diagram (Figure S1).

## 3. Layer-2: biological application (real BE-Hive, 150 ClinVar loci)

Requires PyTorch, the BE-Hive repo, and `variant_summary.txt.gz`.

```bash
# primary analysis (direct-model editors, minimum-efficiency threshold)
python experiments/layer2_application.py \
    --bystander ./be_predict_bystander --celltype mES \
    --clinvar variant_summary.txt.gz --n 150 \
    --direct-only --min-efficiency 0.05 \
    --out-json layer2_full_direct.json

# off-target robustness (aggregate-CFD)
python experiments/layer2_application.py \
    --bystander ./be_predict_bystander --celltype mES \
    --clinvar variant_summary.txt.gz --n 150 \
    --direct-only --min-efficiency 0.05 --aggregate-offtarget \
    --out-json layer2_full_aggot.json
```
Reproduces Section 3.2, Table 1, and the 150-locus statistics (mean front size 3.9;
19% of loci with no high-purity option; mean best purity/efficiency 0.72/0.79).

On a machine with a CUDA device mismatch, force CPU with `CUDA_VISIBLE_DEVICES=""`.

## 4. Tool comparison (front vs single-composite recommendation)

```bash
PYTHONPATH=. python experiments/tool_comparison_run.py \
    --bystander ./be_predict_bystander --celltype mES \
    --clinvar variant_summary.txt.gz --n 150 \
    --out-json tool_comparison.json
```
Reproduces Section 3.4 (single pick on front 150/150; mean 4.5 extra non-dominated designs;
activity-first higher-purity hidden at 144/150).

## 5. Measured-purity validation (Arbab 2020)

```bash
PYTHONPATH=. python experiments/purity_recovery_run.py \
    --pkl-dir <ARBAB_FIGSHARE_DIR> --cell HEK293T \
    --out-json purity_recovery.json
```
Requires the Arbab 2020 processed data (figshare doi:10.6084/m9.figshare.10678097).
Reproduces Section 3.5 (measured purity 0.74 vs 0.29; Wilcoxon P ~ 8e-18).

## Data sources and versions
- ClinVar `variant_summary.txt.gz`, GRCh38 (NCBI)
- GRCh38 reference sequence, Ensembl REST API release 116
- BE-dataHIVE (Schneider and Minary, 2024; doi:10.1186/s12859-024-05898-0)
- Arbab et al. (2020) genotype-resolved screens (doi:10.1016/j.cell.2020.05.037;
  processed data doi:10.6084/m9.figshare.10678097)
- Doench et al. (2016) CFD matrices, via CRISPOR
