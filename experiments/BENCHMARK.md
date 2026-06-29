# PARETO-EDIT — Phase 4 Benchmark Harness

## What it does
Reproducible multi-objective benchmark producing the paper's core results table.

- **Methods:** NSGA-II, CPO-GWO, Weighted-sum (scalarization sweep), Random search
- **Problems:** reproducible SNV suite (CBE target=C + ABE target=A chemistries)
- **Seeds:** 30 (matches the author's prior swarm-benchmark protocol)
- **Two regimes:**
  - **small** — design space fully enumerable, so the *exact* Pareto front is the ground
    truth. Measures **front recovery** (fraction of the true front found) and exact HV.
  - **scaled** — enlarged design space (~1296 designs) under a **capped evaluation budget**.
    Exhaustive search is expensive here, so this regime measures **search efficiency** —
    the regime that matters once real, costly oracles (BE-Hive/DeepBE) replace the surrogate.

## Metrics (per method × problem × seed)
- **HV** hypervolume (↑ better) — Monte-Carlo, fixed reference point.
- **IGD** inverted generational distance vs reference front (↓ better) — captures coverage/spread.
- **Spacing** uniformity of the front (↓ better).
- **Oracle calls** unique evaluations (↓ cheaper) — the cost proxy for real models.
- **Recovery** |found ∩ true| / |true| (small regime only).

## How to run
```bash
python experiments/benchmark.py --quick                 # smoke test (5 seeds, 4 problems)
python experiments/benchmark.py --seeds 30 --per-chem 3 # full run (~3 min)
python experiments/make_tables.py                       # LaTeX tables from the CSVs
```

Outputs: `benchmark_raw.csv`, `benchmark_summary.csv`, `benchmark_wilcoxon.csv`,
`fig_benchmark_hv.png`, `fig_benchmark_efficiency.png`, `table_main.tex`, `table_wilcoxon.tex`.

## How to read the headline result (30-seed run)

| regime | method | HV ↑ | IGD ↓ | oracle calls ↓ | recovery ↑ |
|---|---|---|---|---|---|
| small | **NSGA-II** | 0.733 | **0.002** | 102 | **0.988** |
| small | CPO-GWO | 0.731 | 0.007 | **75** | 0.935 |
| small | Weighted-sum | 0.728 | 0.115 | 150 | 0.649 |
| small | Random | 0.707 | 0.033 | 200 | 0.745 |
| scaled | **NSGA-II** | 0.745 | **0.019** | 217 | — |
| scaled | CPO-GWO | 0.732 | 0.051 | **157** | — |
| scaled | Weighted-sum | 0.739 | 0.243 | 1080 | — |
| scaled | Random | 0.667 | 0.098 | 280 | — |

**Interpretation (the paper's argument).**
1. **Scalarization misses the front.** Weighted-sum reaches competitive HV but its IGD is
   5–60× worse than the metaheuristics: it finds a few good points, not a well-spread front.
   In the small regime it recovers only ~65% of the true Pareto set vs ~99% for NSGA-II. This
   is the core case against existing additive guide-scoring tools.
2. **Cost matters at scale.** In the scaled regime weighted-sum needs the *entire* design space
   (1080 calls) to do this; the metaheuristics achieve comparable HV with 157–217 calls. Once
   the oracle is a real, slow base-editing model, that 5–7× cost gap is decisive.
3. **CPO-GWO is cost-competitive.** It matches NSGA-II's HV with ~30% fewer oracle calls;
   NSGA-II has the edge on IGD/recovery. Both dominate the non-metaheuristic baselines.

**Wilcoxon (NSGA-II vs others, 12 cases):** NSGA-II significantly beats Random in 12/12 on both
HV and IGD; beats Weighted-sum on IGD in 11/12; differences vs CPO-GWO are small (6–8/12),
consistent with two strong metaheuristics bracketing the same front.

## Caveats (state honestly in the paper)
- All numbers use the **surrogate oracle**; they demonstrate the *optimization machinery*, not
  biological accuracy. Headline biological claims must be re-run with real oracles (Phase 3) and
  validated on **held-out empirical libraries** (`empirical_recovery()` hook, Phase 2/3).
- HV alone can flatter scalarization (a few extreme points); always report HV **with** IGD and
  spacing, as done here.
- The scaled-regime reference front is a proxy (union of optimizer outputs), so its absolute IGD
  values are only meaningful *between methods*, not in isolation.
