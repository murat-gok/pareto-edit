# PARETO-EDIT

**Multi-objective optimization of base-editing guide design across efficiency, bystander purity, and off-target risk.**

PARETO-EDIT formulates base-editing guide design as a three-objective optimization problem and
returns the full **Pareto front** of candidate designs — combinations of base editor, PAM variant,
and protospacer placement — rather than collapsing the trade-off into a single weighted score. For
a target single-nucleotide variant it surfaces the set of non-dominated designs that trade
**on-target efficiency**, **bystander purity**, and **off-target risk**, leaving the final choice
to the user's priorities.

The novel objective is **bystander purity** — the fraction of edited outcomes in which only the
target base is changed — treated as an *explicit optimization objective* rather than a predicted
by-product. An ablation shows this matters: dropping the purity objective and optimizing only
efficiency and off-target systematically yields lower-purity designs (mean best purity 0.70 → 0.41
across 30 loci; Wilcoxon p ≈ 6×10⁻⁶).

> Research software accompanying the manuscript *"PARETO-EDIT: multi-objective optimization of
> base-editing guide design across efficiency, bystander purity, and off-target risk."* (under
> review).

---

## Key ideas

- **Pareto front, not a single score.** Two metaheuristics (a from-scratch NSGA-II and a CPO–GWO
  hybrid) recover the trade-off surface. On a benchmark with a known exact front, they recover
  ~99% of it versus ~57% for weighted-sum scalarization, and reach comparable quality with 5–7×
  fewer objective evaluations once the design space is too large to enumerate — the regime that
  holds when a real, slow predictive model scores each design.
- **Modular oracle architecture.** The optimizer is fully decoupled from the biology. Each
  objective is supplied by a pluggable oracle, so published models can be swapped in without
  touching the search:
  - **purity** ← the **BE-Hive** bystander model (Arbab et al., 2020)
  - **off-target** ← **CFD** (Doench et al., 2016), single-mismatch propensity or an optional
    genome-aware-style aggregate-CFD specificity score (Hsu 2013 family)
  - **efficiency** ← a biophysical predictor calibrated to **BE-dataHIVE** measured efficiencies
- **Honest evaluation.** The repository includes the full benchmark harness, ablations, expanded
  baselines (weighted-sum, ε-constraint, Pareto-filter, random), and a guide-level efficiency
  recovery against measured outcomes — including negative findings, reported as such.

The optimization layer is a benchmarking test function; the biological layer uses published
models. These are kept deliberately separate (see `APPROACH_A.md` in the development tree).

---

## Installation

```bash
git clone https://github.com/murat-gok/pareto-edit.git
cd pareto-edit
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ recommended.

### External data and models (not bundled)

These are downloaded/obtained separately and are intentionally git-ignored:

| Resource | Used for | Source |
|---|---|---|
| BE-Hive `be_predict_bystander` | bystander purity | github.com/maxwshen/be_predict_bystander |
| CFD matrices (`mismatch_score.pkl`, `pam_scores.pkl`) | off-target | CRISPOR distribution |
| ClinVar `variant_summary.txt.gz` (GRCh38) | target loci | ncbi.nlm.nih.gov/clinvar |
| BE-dataHIVE efficiency table | efficiency calibration / recovery | be-datahive.com |
| Ensembl REST | genomic context (GRCh38 flanks) | rest.ensembl.org |

---

## Quick start

```python
from design_space import SNVContext
from oracles.surrogate import SurrogateOracle
from optimizers.nsga2 import NSGA2
from pipeline import ParetoEditPipeline

snv = SNVContext(chrom="chr11", pos=5226778, ref="A", alt="G", target_base="C")
pipe = ParetoEditPipeline(oracle=SurrogateOracle(), optimizer=NSGA2())
res = pipe.run(snv, seed=0)
for r in pipe.report(res, snv):
    print(r["editor"], r["pam_variant"], "pos", r["target_pos"],
          "| eff", round(r["efficiency"], 2),
          "purity", round(r["bystander_purity"], 2),
          "offtgt", round(r["offtarget_risk"], 2))
```

Run scripts as modules from the project root, e.g.:

```bash
python -m experiments.benchmark --seeds 30 --per-chem 4   # Layer-1 benchmark
python -m experiments.ablation --n 30                     # objective-dropping + sensitivity
```

---

## Repository layout

```
design_space.py            editor catalogue, design enumeration, search bounds
pipeline.py                SNV -> enumerate -> optimize -> Pareto front
objectives.py              the 3 objectives (minimization vector)
metrics.py                 hypervolume, IGD, spacing; baselines
optimizers/
  nsga2.py                 from-scratch NSGA-II (dynamic objective dimension)
  cpo_gwo.py               CPO-GWO hybrid comparator
oracles/
  base.py                  Oracle ABC + CompositeOracle (pluggable objectives)
  surrogate.py             biophysical surrogate (optimization test function)
  behive.py                real BE-Hive bystander purity (+ proxy-editor flagging)
  cfd.py                   CFD off-target (single-mismatch + aggregate-CFD specificity)
  efficiency_calibration.py  quantile-map surrogate efficiency to BE-dataHIVE measurements
data/
  clinvar.py, genome.py, be_datahive.py   data loaders
experiments/
  benchmark.py             Layer-1 benchmark harness (6 methods)
  layer2_application.py    Layer-2 real-model application over ClinVar loci
  ablation.py              objective-dropping + hyperparameter sensitivity
  efficiency_recovery_run.py  guide-level efficiency recovery on BE-dataHIVE targets
  fetch_be_datahive.py     resilient BE-dataHIVE downloader
```

---

## Reproducing the paper

The exact command sequence that produces every number in the manuscript (benchmark, 150-locus
application, ablations, recovery) is documented in the development tree's run guide. In brief:
fit the efficiency calibration, run the Layer-1 benchmark, run the Layer-2 application over 150
ClinVar loci with the real BE-Hive/CFD oracles, then run the ablations and recovery.

---

## Citation

If you use PARETO-EDIT, please cite the manuscript (citation to be updated on publication):

```bibtex
@article{paretoedit,
  title   = {PARETO-EDIT: multi-objective optimization of base-editing guide design
             across efficiency, bystander purity, and off-target risk},
  author  = {G{\"o}k, Murat and others},
  journal = {under review},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on the BE-Hive bystander model (Arbab et al., 2020), the Doench et al. (2016) CFD scoring
matrices, ClinVar, Ensembl, and BE-dataHIVE. PARETO-EDIT is complementary to outcome predictors
such as BE-Hive, BE-DICT, and DeepBE: it consumes their predictions as objectives rather than
competing with them.
