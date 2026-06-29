# PARETO-EDIT

**Multi-objective evolutionary framework for Pareto-optimal base-editor / sgRNA selection,
balancing on-target efficiency, bystander purity, and off-target risk.**

Target venue: *Briefings in Bioinformatics* / *NAR Genomics and Bioinformatics*.

---

## 1. The problem (one paragraph)

Given a pathogenic SNV, a designer must jointly choose a **base editor** (CBE/ABE/CGBE variant),
a **Cas9 PAM variant**, and a **protospacer placement** so that the target base lands in the
editing window with high efficiency, while neighboring **bystander** bases are *not* edited and
genome-wide **off-target** risk is minimized. These objectives genuinely conflict. Every existing
tool collapses them into a single additive score and returns one ranked list. PARETO-EDIT instead
returns the full **Pareto front** of non-dominated designs, computed by true multi-objective
metaheuristics over the combinatorial design space.

Novelty claims encoded in this repo:
1. **First metaheuristic multi-objective optimizer for base-editing guide design** (no published
   GA/PSO/ACO/NSGA-II tool exists for this task).
2. A new **bystander-purity** objective: predicted fraction of edited reads carrying *only* the
   intended conversion.
3. A pluggable **Oracle** abstraction so validated predictors (BE-Hive, DeepBE, CRISPRon, crispAI)
   drop in as black-box objective functions without changing the optimizer.

---

## 2. Architecture (modules)

```
pareto_edit/
├── design_space.py     # Decision variables: editor × PAM × protospacer offset × guide
├── oracles/
│   ├── base.py         # Abstract Oracle interface (efficiency / bystander / offtarget)
│   ├── surrogate.py    # Biophysically-motivated surrogate oracles (phase-1 default)
│   └── real.py         # Adapters for BE-Hive / DeepBE / CRISPRon / crispAI (phase-3 stubs)
├── objectives.py       # Maps a Design -> objective vector (to MINIMIZE)
├── optimizers/
│   ├── base.py         # Abstract Optimizer interface + Result container
│   ├── nsga2.py        # From-scratch NSGA-II (fast non-dom sort + crowding)
│   └── cpo_gwo.py      # CPO-GWO multi-objective hybrid (archive + leaders)
├── metrics.py          # Hypervolume, IGD, spacing, front coverage
├── pipeline.py         # End-to-end: SNV -> enumerate -> optimize -> Pareto front
└── experiments/
    └── run_demo.py     # Reproducible demo + figures
```

Objective convention: **everything is minimization**. We minimize
`[ -efficiency, -bystander_purity, offtarget_risk ]`.

---

## 3. Step-by-step roadmap

### Phase 0 — Scaffold & contracts (THIS commit)
- [x] Decision-variable encoding for `(editor, pam_variant, protospacer_offset, guide_seq)`.
- [x] Abstract `Oracle` and `Optimizer` interfaces (stable contracts).
- [x] Surrogate oracles with biologically sensible structure (window curve, bystander
      composition, mismatch-weighted off-target) so the whole pipeline runs end-to-end today.
- [x] From-scratch NSGA-II + CPO-GWO behind one interface.
- [x] Quality metrics (HV, IGD) and a runnable demo producing a Pareto front + figures.

### Phase 1 — Realistic surrogate calibration
- [ ] Replace hand-set surrogate constants with values fit to public summary stats
      (editing-window profiles from Arbab 2020; PAM efficiency priors).
- [ ] Add PAM enumeration against a real genomic context (Ensembl/UCSC GRCh38 slice).
- [ ] Unit tests for monotonicity / sanity (e.g. centered target ⇒ higher efficiency).

### Phase 2 — Real data ingestion
- [ ] ClinVar pathogenic C>T / A>G SNV loader (problem instances).
- [ ] gnomAD common-variant overlap penalty for off-target sites.
- [ ] BE-dataHIVE API client (>460k gRNA combinations) for empirical lookup / validation.

### Phase 3 — Real oracles (drop-in)
- [ ] `real.py` adapters: BE-Hive & DeepBE (efficiency + bystander spectra),
      CRISPRon ΔG_B (binding energy), crispAI / CFD (off-target).
- [ ] Caching layer (oracle calls are the cost bottleneck).

### Phase 4 — Benchmarking (the paper's results)
- [ ] Baselines to beat: weighted-sum scalarization, additive composite (PnB-style),
      random search, Bayesian optimization (Pareto screening).
- [ ] Metrics: hypervolume, IGD, spacing, runtime, #oracle-calls.
- [ ] **Held-out empirical-library recovery**: does the Pareto front contain the
      empirically best editor/guide measured in Arbab / DeepBE? (top-k recall, regret).
- [ ] Statistical tests across 30 seeds (Wilcoxon, as in your prior swarm benchmarks).

### Phase 5 — Manuscript artifacts
- [ ] Paper-ready LaTeX tables + figures (Pareto fronts, HV convergence, ablations).
- [ ] Reproducibility: fixed seeds, config files, environment lock.

---

## 4. Validation philosophy
Oracles carry real error, so all headline results are reported on **held-out empirically
measured libraries**, never on predicted scores alone. This is the reviewer-proofing strategy.
