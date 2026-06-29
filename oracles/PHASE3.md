# PARETO-EDIT — Phase 3: Real Oracle Integration (BE-Hive)

Replaces the surrogate with a **real, published editing-outcome model** so the optimizer
produces biologically grounded Pareto fronts. This is the phase that moves the project from
"the optimization machinery works" to "the framework produces meaningful editing designs."

## Why BE-Hive
Chosen over DeepBE because its interface is clean, pure-Python, and gives exactly what the
novel **bystander-purity** objective needs:
- `init_model(base_editor, celltype)` + `predict(seq)` returning a genotype probability
  distribution, plus `add_genotype_column()` to materialize full genotypes.
- Separate `be_predict_efficiency` and `be_predict_bystander` repos (efficiency + bystander).
- Runs on CPU (no GPU needed) — fits the office Xeon workflow.
- Ships editors overlapping our catalogue: BE4, evoAPOBEC, ABE, CGBEs, etc.

## Modules
- `oracles/behive.py` — `BEHiveOracle` implementing the `Oracle` interface.
  - Builds BE-Hive's 50-nt input (protospacer at `seq[20:40]`, target base at index `19+pos`).
  - **efficiency**: from `be_predict_efficiency` if present, else proxied by total edited
    probability mass `(1 − WT genotype frequency)`.
  - **bystander_purity** (the novel objective): from the bystander genotype distribution,
    the summed frequency of genotypes where **only the target position changed**, divided by
    total edited mass. *(Purity computation unit-tested independently of PyTorch — verified
    exact on a hand-built distribution.)*
  - **offtarget_risk**: not provided (BE-Hive doesn't model it) — sourced separately.
- `oracles/base.py::CompositeOracle` — composes per-objective oracles (efficiency+purity from
  BE-Hive, off-target from crispAI/CFD). This is the architecturally correct setup: each
  objective comes from the model validated for it. *(End-to-end tested through the pipeline.)*
- `experiments/phase3_behive.py` — driver that optimizes against the real BE-Hive oracle.

## Editor name mapping (`BEHIVE_EDITOR_MAP`)
Our catalogue → BE-Hive public names. Exact where BE-Hive ships the editor (BE4max→BE4,
evoAPOBEC, ABE), nearest-proxy where it does not (YE1→BE4 as a narrow-window CBE proxy;
CGBE1→CG-eA3A). Proxies are documented in code and must be stated as a limitation in the paper.

## How to run (user's machine — needs torch + repos)
```bash
git clone https://github.com/maxwshen/be_predict_bystander.git
git clone https://github.com/maxwshen/be_predict_efficiency.git
pip install torch pandas numpy            # CPU torch is sufficient

python experiments/phase3_behive.py \
    --bystander /path/to/be_predict_bystander \
    --efficiency /path/to/be_predict_efficiency \
    --celltype mES --optimizer nsga2
```

## Status & honest caveats
- **Verified here (without torch):** the purity computation logic (exact unit test); the
  `Oracle` interface conformance; `CompositeOracle` running end-to-end through NSGA-II.
- **Environment-blocked here:** PyTorch cannot be installed in this sandbox
  (download.pytorch.org not allowlisted; no PyPI wheel available), so the *live* BE-Hive model
  cannot be executed here. The repos clone fine; the adapter is written against their verified
  interface and will run where torch is present. The driver detects missing torch and exits
  with instructions rather than fabricating real-model numbers.
- **Phase-3b (next):** swap the off-target source from the surrogate to a real model
  (crispAI/CFD); add guide-level matching to `empirical_recovery()`; report
  surrogate-vs-real-oracle agreement (rank correlation of fronts) as a validation figure.
- **Validation plan for the paper:** (1) run BE-Hive front on a panel of ClinVar loci;
  (2) compare to surrogate front (Spearman of objective rankings) to quantify how much the
  surrogate flatters/distorts; (3) score `empirical_recovery()` against BE-dataHIVE measured
  outcomes — the headline reviewer-proofing result.
```
