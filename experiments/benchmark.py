"""
experiments/benchmark.py
========================
PARETO-EDIT Phase-4 benchmark harness.

Produces the paper's core results table:
  methods  : NSGA-II, CPO-GWO, weighted-sum, random search
  problems : a reproducible SNV suite (CBE + ABE chemistries)
  seeds    : 30 (matching the author's prior swarm-benchmark protocol)
  regimes  : "small"  -> exact recovery vs exhaustive ground truth
             "scaled" -> search efficiency under a capped evaluation budget

Per (method, problem, seed) we record:
  - hypervolume (HV)            [higher better]
  - inverted generational distance (IGD) vs reference front [lower better]
  - spacing                    [lower better, uniformity]
  - unique oracle calls        [lower better, cost proxy]
  - front_recovery             [small regime only: |found ∩ true| / |true|]

Statistics: per problem, Wilcoxon signed-rank tests (NSGA-II vs each other method) across
seeds on HV and IGD, plus mean ± std aggregation. Outputs CSV + a summary table + figures.

Held-out empirical recovery (Phase-4 hook): `empirical_recovery()` is stubbed to plug in
BE-dataHIVE / Arbab measured libraries once available; the interface is defined here.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from design_space import SNVContext
import design_space as ds_mod
from oracles.surrogate import SurrogateOracle
from optimizers.nsga2 import NSGA2
from optimizers.cpo_gwo import CPO_GWO
from optimizers.nsga3 import NSGA3
from pipeline import ParetoEditPipeline
from metrics import (
    exhaustive_front, hypervolume, igd, spacing,
    baseline_weighted_sum, baseline_random_search,
    baseline_epsilon_constraint, baseline_pareto_filter,
)
from optimizers.base import non_dominated_mask
from experiments.suite import make_problems, scaled_space, space_size

OUT = os.path.dirname(os.path.abspath(__file__))
REF = np.array([0.05, 0.05, 1.30])  # HV reference point (dominated by all feasible objectives)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class Record:
    regime: str
    problem: str
    method: str
    seed: int
    hv: float
    igd: float
    spacing: float
    n_evals: int
    front_size: int
    front_recovery: float
    runtime_s: float


# --------------------------------------------------------------------------- #
# Method runners (all return (designs, F, n_evals))
# --------------------------------------------------------------------------- #

def _run_optimizer(OptCls, kwargs, snv, oracle, seed):
    pipe = ParetoEditPipeline(oracle=oracle, optimizer=OptCls(**kwargs))
    res = pipe.run(snv, seed=seed)
    return res.designs, res.objectives, res.n_evals


_WS_CACHE: dict = {}


def _run_weighted_sum(snv, oracle, seed, n_weights=15):
    """Weighted-sum scalarization SWEEP: unlike a single fixed weight, we sweep weight vectors
    and collect the non-dominated union. This is the strongest fair version of the additive-
    scoring baseline (gives scalarization its best shot at approximating the front).
    The expensive full enumeration of objectives is deterministic per SNV, so it is cached;
    only the weight sweep uses the seed."""
    from design_space import enumerate_designs
    from objectives import evaluate
    key = (snv.chrom, snv.pos, snv.target_base, len(ds_mod.EDITORS), len(ds_mod.PAM_VARIANTS),
           ds_mod.OFFSET_RANGE)
    if key not in _WS_CACHE:
        designs = enumerate_designs(snv)
        F = np.array([evaluate(d, snv, oracle) for d in designs])
        _WS_CACHE[key] = (designs, F)
    designs, F = _WS_CACHE[key]
    rng = np.random.default_rng(seed)
    span = (F.max(axis=0) - F.min(axis=0)); span[span == 0] = 1
    Fn = (F - F.min(axis=0)) / span
    picks = set()
    for _ in range(n_weights):
        w = rng.dirichlet(np.ones(3))
        picks.add(int(np.argmin(Fn @ w)))
    idx = sorted(picks)
    subF = F[idx]
    nd = non_dominated_mask(subF)
    return [designs[idx[i]] for i in np.where(nd)[0]], subF[nd], len(designs)


def _run_random(snv, oracle, seed, budget):
    d, F = baseline_random_search(snv, oracle, n_evals=budget, seed=seed)
    return d, F, budget


def _run_epsilon(snv, oracle, seed):
    d, F = baseline_epsilon_constraint(snv, oracle, seed=seed)
    # cost = one full enumeration pass (like weighted-sum)
    from design_space import enumerate_designs
    return d, F, len(enumerate_designs(snv))


def _run_pareto_filter(snv, oracle, seed, budget):
    d, F = baseline_pareto_filter(snv, oracle, n_evals=budget, seed=seed)
    return d, F, budget


# --------------------------------------------------------------------------- #
# Single evaluation
# --------------------------------------------------------------------------- #

def _design_keys(designs):
    return set(d.as_tuple() for d in designs)


def evaluate_run(regime, problem_name, method, snv, oracle, seed, gt_designs, gt_F, budget):
    t0 = time.perf_counter()

    if method == "NSGA-II":
        d, F, ne = _run_optimizer(NSGA2, dict(pop_size=40, n_gen=max(8, budget // 40)), snv, oracle, seed)
    elif method == "NSGA-III":
        d, F, ne = _run_optimizer(NSGA3, dict(pop_size=40, n_gen=max(8, budget // 40)), snv, oracle, seed)
    elif method == "CPO-GWO":
        d, F, ne = _run_optimizer(CPO_GWO, dict(pop_size=40, n_gen=max(8, budget // 40)), snv, oracle, seed)
    elif method == "WeightedSum":
        d, F, ne = _run_weighted_sum(snv, oracle, seed)
    elif method == "EpsilonConstraint":
        d, F, ne = _run_epsilon(snv, oracle, seed)
    elif method == "ParetoFilter":
        d, F, ne = _run_pareto_filter(snv, oracle, seed, budget)
    elif method == "Random":
        d, F, ne = _run_random(snv, oracle, seed, budget)
    else:
        raise ValueError(method)

    rt = time.perf_counter() - t0
    hv = hypervolume(F, REF, seed=12345)
    ig = igd(F, gt_F) if gt_F is not None and len(gt_F) else float("nan")
    sp = spacing(F)

    # exact front recovery (small regime only, where gt is the true front)
    rec = float("nan")
    if gt_designs is not None:
        true_keys = _design_keys(gt_designs)
        found = _design_keys(d) & true_keys
        rec = len(found) / max(1, len(true_keys))

    return Record(regime, problem_name, method, seed, hv, ig, sp, ne, len(F), rec, rt)


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #

#METHODS = ["NSGA-II", "CPO-GWO", "WeightedSum", "EpsilonConstraint", "ParetoFilter", "Random"]
METHODS = ["NSGA-II", "NSGA-III", "CPO-GWO", "WeightedSum", "EpsilonConstraint", "ParetoFilter", "Random"]

def run_regime(regime: str, problems, seeds: List[int], oracle, budget: int,
               enumerate_gt: bool) -> List[Record]:
    records: List[Record] = []
    for p in problems:
        gt_designs, gt_F = (exhaustive_front(p.snv, oracle) if enumerate_gt else (None, None))
        if not enumerate_gt:
            gt_F = _proxy_reference(p.snv, oracle, seeds, budget)  # computed ONCE per problem
        for method in METHODS:
            for s in seeds:
                rec = evaluate_run(regime, p.name, method, p.snv, oracle, s,
                                   gt_designs, gt_F, budget)
                records.append(rec)
        print(f"   done: {regime}/{p.name}")
    return records


def _proxy_reference(snv, oracle, seeds, budget):
    """Build a proxy reference front for the scaled regime as the non-dominated union of all
    optimizer outputs across a few seeds (standard practice when the true front is unknown)."""
    from objectives import evaluate
    allF = []
    for s in seeds[: min(5, len(seeds))]:
        for OptCls in (NSGA2, NSGA3, CPO_GWO):
            _, F, _ = _run_optimizer(OptCls, dict(pop_size=40, n_gen=max(8, budget // 40)),
                                     snv, oracle, s)
            allF.append(F)
    U = np.vstack(allF)
    nd = non_dominated_mask(U)
    return U[nd]


# --------------------------------------------------------------------------- #
# Aggregation + statistics
# --------------------------------------------------------------------------- #

def aggregate(records: List[Record]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in records])
    g = (df.groupby(["regime", "problem", "method"])
           .agg(hv_mean=("hv", "mean"), hv_std=("hv", "std"),
                igd_mean=("igd", "mean"), igd_std=("igd", "std"),
                spacing_mean=("spacing", "mean"),
                evals_mean=("n_evals", "mean"),
                recovery_mean=("front_recovery", "mean"),
                runtime_mean=("runtime_s", "mean"))
           .reset_index())
    return df, g


def wilcoxon_table(df: pd.DataFrame, baseline="NSGA-II", metric="hv") -> pd.DataFrame:
    """Per (regime, problem): Wilcoxon signed-rank of baseline vs each other method across seeds.
    Reports p-value and median difference (baseline - other)."""
    rows = []
    for (regime, problem), sub in df.groupby(["regime", "problem"]):
        base = sub[sub.method == baseline].sort_values("seed")[metric].values
        for method in sub.method.unique():
            if method == baseline:
                continue
            other = sub[sub.method == method].sort_values("seed")[metric].values
            n = min(len(base), len(other))
            b, o = base[:n], other[:n]
            diff = b - o
            if np.allclose(diff, 0):
                pval, stat = 1.0, 0.0
            else:
                try:
                    stat, pval = wilcoxon(b, o, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    stat, pval = float("nan"), 1.0
            rows.append({"regime": regime, "problem": problem, "metric": metric,
                         "baseline": baseline, "method": method,
                         "median_diff": float(np.median(diff)), "p_value": float(pval)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Held-out empirical recovery (Phase-4 hook, stubbed)
# --------------------------------------------------------------------------- #

def empirical_recovery(front_designs, snv, oracle=None, client=None, top_k=3):
    """Guide-level held-out empirical recovery against BE-dataHIVE measured outcomes.

    For each design in the predicted Pareto front, look up its protospacer in BE-dataHIVE's
    measured gRNA library (exact match on the 'grna' column). Among designs whose protospacer was
    actually measured, identify the one with the highest *measured* efficiency (the empirical
    optimum). Then ask whether PARETO-EDIT ranks that same guide highly by its *predicted*
    efficiency — i.e. does the predicted front recover the empirically best measured design?

    This scores predictions against real measurements at guide resolution, not editor resolution,
    and is the reviewer-proofing validation.

    Returns a dict:
      n_measured        : number of front designs with a BE-dataHIVE measurement
      empirical_best     : (editor, protospacer, measured_efficiency) of the measured optimum
      recovered_top_k    : whether the empirical-best guide is within the front's top-k by
                           PREDICTED efficiency
      rank_corr          : Spearman correlation between predicted and measured efficiency over the
                           measured subset (NaN if <3 points)
      regret             : measured efficiency of the empirical best minus the measured efficiency
                           of the design the predictor ranked #1 (0 = perfect recovery; NaN if not
                           computable)

    Requires a reachable BE-dataHIVE (or PARETO_EDIT_OFFLINE=1 for a synthetic smoke test) and the
    `oracle` used to score predicted efficiency for the front designs.
    """
    from data.be_datahive import BEDataHiveClient
    from objectives import evaluate
    import numpy as np

    client = client or BEDataHiveClient()

    # collect designs whose protospacer was actually measured
    measured = []  # (design, protospacer, measured_eff, predicted_eff)
    for d in front_designs:
        proto = snv.window_context(d)
        m = client.measured_for_guide(proto, editor=d.editor)
        if m is None or m.efficiency != m.efficiency:  # None or NaN
            continue
        pred_eff = float(-evaluate(d, snv, oracle)[0]) if oracle is not None else float("nan")
        measured.append((d, proto, float(m.efficiency), pred_eff))

    if not measured:
        return {"n_measured": 0, "empirical_best": None, "recovered_top_k": None,
                "rank_corr": float("nan"), "regret": float("nan")}

    # empirical optimum (highest measured efficiency)
    measured.sort(key=lambda x: x[2], reverse=True)
    best_d, best_proto, best_meas, _ = measured[0]

    # predicted ranking (by predicted efficiency, descending)
    by_pred = sorted(measured, key=lambda x: x[3], reverse=True)
    topk = by_pred[:top_k]
    recovered_top_k = any(x[1] == best_proto for x in topk)

    # rank correlation between predicted and measured efficiency
    rank_corr = float("nan")
    if len(measured) >= 3:
        from scipy.stats import spearmanr
        pred = [x[3] for x in measured]
        meas = [x[2] for x in measured]
        if np.std(pred) > 0 and np.std(meas) > 0:
            rank_corr = float(spearmanr(pred, meas).correlation)

    # regret: measured eff of empirical best minus measured eff of the predictor's #1 pick
    pred_top_design = by_pred[0]
    regret = float(best_meas - pred_top_design[2])

    return {
        "n_measured": len(measured),
        "empirical_best": (best_d.editor, best_proto, best_meas),
        "recovered_top_k": recovered_top_k,
        "rank_corr": rank_corr,
        "regret": regret,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(n_seeds: int = 30, n_per_chem: int = 4, quick: bool = False):
    seeds = list(range(n_seeds if not quick else 5))
    problems = make_problems(n_per_chem=n_per_chem if not quick else 2, seed=0)
    oracle = SurrogateOracle()

    all_records: List[Record] = []

    print(f"[small regime] {len(problems)} problems x {len(METHODS)} methods x {len(seeds)} seeds")
    small = run_regime("small", problems, seeds, oracle, budget=200, enumerate_gt=True)
    all_records += small

    print(f"[scaled regime] enlarging design space, capped budget ...")
    with scaled_space(offset_hi=16):
        # report space size on first problem
        print(f"   scaled design-space size ~ {space_size(problems[0].snv)} per problem")
        scaled = run_regime("scaled", problems, seeds, oracle, budget=280, enumerate_gt=False)
    all_records += scaled

    df, g = aggregate(all_records)
    df.to_csv(os.path.join(OUT, "benchmark_raw.csv"), index=False)
    g.to_csv(os.path.join(OUT, "benchmark_summary.csv"), index=False)

    w_hv = wilcoxon_table(df, baseline="NSGA-II", metric="hv")
    w_igd = wilcoxon_table(df, baseline="NSGA-II", metric="igd")
    w = pd.concat([w_hv, w_igd], ignore_index=True)
    w.to_csv(os.path.join(OUT, "benchmark_wilcoxon.csv"), index=False)

    _print_headline(g, w)
    _figures(df, g)
    print(f"\nWrote: benchmark_raw.csv, benchmark_summary.csv, benchmark_wilcoxon.csv,"
          f" fig_benchmark_hv.png, fig_benchmark_efficiency.png in {OUT}")
    return df, g, w


def _print_headline(g, w):
    print("\n=== SUMMARY (mean over seeds) ===")
    for regime in ["small", "scaled"]:
        sub = g[g.regime == regime]
        if sub.empty:
            continue
        print(f"\n[{regime}]")
        agg = (sub.groupby("method")
                  .agg(HV=("hv_mean", "mean"), IGD=("igd_mean", "mean"),
                       recovery=("recovery_mean", "mean"),
                       evals=("evals_mean", "mean"))
                  .reindex(METHODS))
        with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
            print(agg.to_string())


def _figures(df, g):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"NSGA-II": "#1f77b4", "NSGA-III": "#2ca02c", "CPO-GWO": "#d62728",
              "WeightedSum": "#2ca02c", "EpsilonConstraint": "#9467bd",
              "ParetoFilter": "#8c564b", "Random": "#7f7f7f"}
    # HV by method per regime (box-ish bar with std)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, regime in zip(axes, ["small", "scaled"]):
        sub = df[df.regime == regime]
        means = sub.groupby("method")["hv"].mean().reindex(METHODS)
        stds = sub.groupby("method")["hv"].std().reindex(METHODS)
        ax.bar(range(len(METHODS)), means.values, yerr=stds.values, capsize=4,
               color=[colors[m] for m in METHODS])
        ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=20, fontsize=8)
        ax.set_ylabel("hypervolume"); ax.set_title(f"HV by method ({regime})"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_benchmark_hv.png"), dpi=150); plt.close(fig)

    # Efficiency: HV vs unique oracle calls (scaled regime) -- the key cost argument
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sub = df[df.regime == "scaled"]
    for m in METHODS:
        s = sub[sub.method == m]
        ax.scatter(s["n_evals"], s["hv"], s=18, alpha=0.5, c=colors[m], label=m)
        ax.scatter(s["n_evals"].mean(), s["hv"].mean(), s=160, c=colors[m],
                   edgecolor="black", marker="D", zorder=5)
    ax.set_xlabel("unique oracle calls (cost)"); ax.set_ylabel("hypervolume")
    ax.set_title("Quality vs cost (scaled regime)\nupper-left = better"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_benchmark_efficiency.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run (5 seeds, 2 problems/chem)")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--per-chem", type=int, default=4)
    args = ap.parse_args()
    main(n_seeds=args.seeds, n_per_chem=args.per_chem, quick=args.quick)
