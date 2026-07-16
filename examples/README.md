# Minimal reproducible example

This folder contains a small, self-contained test input so that reviewers (and users)
can run PARETO-EDIT end to end without downloading the large ClinVar / BE-dataHIVE files.

## Files
- `example_snvs.csv` — three pathogenic SNVs (input). Columns:
  - `chrom, pos, ref, alt` — variant coordinates (GRCh38)
  - `target_base` — the sense-strand base the editor deaminates
  - `required_edit` — the full correction conversion (e.g. `A>G` for a G>A lesion)
- `expected_output.csv` — an example of the Pareto-front designs the tool returns for the
  locus `chr2:96254050 G>A`, with columns:
  `locus, editor, pam_variant, target_pos, efficiency, bystander_purity, offtarget_risk`.

## Run

From the repository root:

```bash
python -m experiments.run_example --input examples/example_snvs.csv --out out.csv
```

This runs the surrogate (offline) oracle so it needs no downloads or GPU. It prints each
locus's Pareto-front size and writes the non-dominated designs to `out.csv`, whose columns
match `expected_output.csv`. Numerical values from the surrogate oracle are illustrative;
the real-model (BE-Hive) fronts reported in the paper are produced by
`experiments/layer2_application.py` (see the main README).
