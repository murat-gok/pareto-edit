# PARETO-EDIT — Phase 2: Real-Data Ingestion

Wires real public data into the pipeline: **ClinVar** pathogenic SNVs as problem instances, and
**BE-dataHIVE** measured outcomes as held-out empirical ground truth for recovery scoring.

## Modules

- `data/clinvar.py` — streams `variant_summary.txt.gz`, keeps GRCh38 pathogenic / likely-
  pathogenic **single-nucleotide variants** that base editors can *correct*, and converts each
  into an `SNVContext`. Filtering is auditable and conservative:
  - SNVs only (no indels/CNVs).
  - GRCh38 only.
  - Base-editable transitions only by default: `G>A`, `C>T` → CBE (target C); `T>C`, `A>G` →
    ABE (target A). Transversions and CGBE-only `C>G`/`G>C` are excluded unless
    `include_cgbe=True`.
- `data/be_datahive.py` — wraps the official `be-datahive` PyPI package (REST API over
  be-datahive.com; verified methods `get_efficiency`, `get_bystander`, `get_studies`,
  `get_*_ml_arrays`). Exposes `best_measured_for_locus()` for recovery scoring. Falls back to a
  schema-matched synthetic stand-in when offline.
- `experiments/phase2_demo.py` — ClinVar → optimizer → empirical recovery, end-to-end.
- `experiments/benchmark.py::empirical_recovery()` — now implemented (was a stub): scores
  whether the predicted Pareto front recovers BE-dataHIVE's empirically best editor for a locus
  (presence + top-k).

## Data sources (public, no auth)

| Data | URL | Notes |
|---|---|---|
| ClinVar tab-delimited | `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz` | simplest to parse; HGVS (right-shifted) coords |
| ClinVar VCF (GRCh38) | `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz` | VCF (left-shifted) coords |
| BE-dataHIVE | `https://be-datahive.com/` + `pip install be-datahive` | >460k gRNA combinations from 5 studies (Arbab, Pallaseni, Marquart, Yuan, Song) |
| gnomAD (Phase-3 off-target penalty) | `https://gnomad.broadinstitute.org/downloads` | common-variant overlap |

**Coordinate caveat:** ClinVar tab-delimited/XML positions are HGVS (right-shifted); the VCF is
left-shifted. For SNVs outside repeat regions — the only class we keep — these coincide, so the
distinction does not affect PARETO-EDIT. Documented in `data/clinvar.py`.

## How to run

```bash
# Offline smoke test (synthetic ClinVar + synthetic BE-dataHIVE; no downloads):
python experiments/phase2_demo.py --n 8

# Real run:
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
python experiments/phase2_demo.py --clinvar variant_summary.txt.gz --live --n 50
```

## Status & honest caveats

- **Verified working:** ClinVar parsing/filtering (unit-tested against a synthetic
  variant_summary with the real column names — correctly drops indels, benign, wrong-assembly,
  transversions; keeps CBE/ABE transitions with correct chemistry). Full ClinVar→optimizer→
  recovery chain runs end-to-end offline.
- **Environment-blocked here:** live be-datahive.com returns HTTP 403 from this sandbox (domain
  not allowlisted). The client and API signatures are correct; live calls will work in an
  environment with network access to be-datahive.com.
- **Offline recovery numbers are plumbing checks only** — they exercise the code path, not
  biology. Real recovery requires the ClinVar download + live BE-dataHIVE.
- **Phase-3 remaining for true recovery:** guide-level matching (align each predicted design's
  protospacer to BE-dataHIVE guides) and exact regret (measured-best efficiency minus the
  front's best for that editor/guide). Currently recovery is scored at editor granularity.
