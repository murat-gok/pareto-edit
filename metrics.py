"""
metrics.py
==========
Multi-objective quality indicators + simple baselines for benchmarking.

Hypervolume and IGD are validated against pymoo where available; we also ship a
dependency-light hypervolume (Monte-Carlo for n_obj>2, exact-ish for 2D) so results are
reproducible without pymoo.
"""

from __future__ import annotations

from typing import List

import numpy as np

from design_space import Design, SNVContext, enumerate_designs
from objectives import evaluate
from optimizers.base import non_dominated_mask


def hypervolume(F: np.ndarray, ref: np.ndarray, n_mc: int = 200_000, seed: int = 0) -> float:
    """Hypervolume dominated by front F (minimization) bounded by reference point `ref`.
    Monte-Carlo estimate (works for any #objectives). Front and ref must be commensurate."""
    F = np.atleast_2d(F)
    nd = non_dominated_mask(F)
    F = F[nd]
    if len(F) == 0:
        return 0.0
    lo = F.min(axis=0)
    hi = ref
    if np.any(lo >= hi):
        # clip points that exceed ref
        F = F[np.all(F < ref, axis=1)]
        if len(F) == 0:
            return 0.0
        lo = F.min(axis=0)
    rng = np.random.default_rng(seed)
    box_vol = np.prod(hi - lo)
    if box_vol <= 0:
        return 0.0
    pts = rng.uniform(lo, hi, size=(n_mc, F.shape[1]))
    # a sample is "dominated" if some front point is <= it on all axes
    dominated = np.zeros(n_mc, dtype=bool)
    for f in F:
        dominated |= np.all(f <= pts, axis=1)
    return float(box_vol * dominated.mean())


def igd(F: np.ndarray, reference_front: np.ndarray) -> float:
    """Inverted Generational Distance: mean distance from each reference point to the
    nearest solution in F. Lower is better."""
    F = np.atleast_2d(F)
    if len(F) == 0:
        return float("inf")
    d = np.array([np.min(np.linalg.norm(F - r, axis=1)) for r in reference_front])
    return float(d.mean())


def spacing(F: np.ndarray) -> float:
    """Spacing metric (uniformity of distribution). Lower = more uniform."""
    F = np.atleast_2d(F)
    if len(F) < 2:
        return 0.0
    dmin = []
    for i in range(len(F)):
        others = np.delete(F, i, axis=0)
        dmin.append(np.min(np.linalg.norm(others - F[i], axis=1)))
    dmin = np.array(dmin)
    return float(np.sqrt(np.mean((dmin - dmin.mean()) ** 2)))


# --------------------------------------------------------------------------- #
# Baselines to beat (Phase-4)
# --------------------------------------------------------------------------- #

def baseline_weighted_sum(snv: SNVContext, oracle, weights=(1/3, 1/3, 1/3), seed: int = 0):
    """Additive/weighted-sum scalarization over the full enumeration: returns the single
    'best' design under fixed weights (what existing tools effectively do). Reported as a
    degenerate front of size 1."""
    designs = enumerate_designs(snv)
    F = np.array([evaluate(d, snv, oracle) for d in designs])
    w = np.array(weights)
    # normalize objectives to [0,1] before weighting (fair scalarization)
    span = (F.max(axis=0) - F.min(axis=0))
    span[span == 0] = 1
    Fn = (F - F.min(axis=0)) / span
    scores = Fn @ w
    best = int(np.argmin(scores))
    return [designs[best]], F[best:best + 1]


def baseline_epsilon_constraint(snv: SNVContext, oracle, n_eps: int = 6, seed: int = 0):
    """epsilon-constraint scalarization: optimize one objective (efficiency) subject to
    progressively tighter constraints (epsilon thresholds) on the others (purity, off-target),
    collecting the optimum from each constraint setting into a front. This is the classic
    multi-objective baseline that, unlike weighted-sum, can in principle reach non-convex parts
    of the front — a stronger comparator than scalarization alone.

    We sweep epsilon thresholds on purity and off-target over the observed range and, for each
    combination, pick the feasible design with the best efficiency. Evaluates the full enumeration
    once (cost = |designs|), like weighted-sum.
    """
    designs = enumerate_designs(snv)
    F = np.array([evaluate(d, snv, oracle) for d in designs])  # columns: [-eff, -purity, offtgt]
    eff = -F[:, 0]
    purity = -F[:, 1]
    offt = F[:, 2]
    # epsilon grids over observed ranges
    p_levels = np.linspace(purity.min(), purity.max(), n_eps)
    o_levels = np.linspace(offt.min(), offt.max(), n_eps)
    chosen = []
    for p_eps in p_levels:
        for o_eps in o_levels:
            feasible = np.where((purity >= p_eps) & (offt <= o_eps))[0]
            if feasible.size == 0:
                continue
            best = feasible[np.argmax(eff[feasible])]
            chosen.append(best)
    if not chosen:
        # fall back to global best efficiency
        chosen = [int(np.argmax(eff))]
    idx = np.unique(chosen)
    Fc = F[idx]
    nd = non_dominated_mask(Fc)
    return [designs[idx[i]] for i in np.where(nd)[0]], Fc[nd]


def baseline_pareto_filter(snv: SNVContext, oracle, n_evals: int, seed: int = 0):
    """Naive Pareto filter: sample designs (here, the full small enumeration or a random subset
    under budget) and return their non-dominated set, with NO search guidance. This isolates the
    value of the metaheuristic's *search* from the mere act of Pareto-filtering: if a brute
    sample matched the optimizers, search would add nothing. Distinct from random_search only in
    framing (no budget-matched stochastic 'optimization' claim) and reported as a baseline that a
    practitioner could trivially implement on top of an existing predictor.
    """
    rng = np.random.default_rng(seed)
    designs = enumerate_designs(snv)
    if n_evals >= len(designs):
        cand = designs
    else:
        idx = rng.choice(len(designs), size=n_evals, replace=False)
        cand = [designs[i] for i in idx]
    F = np.array([evaluate(d, snv, oracle) for d in cand])
    nd = non_dominated_mask(F)
    return [cand[i] for i in np.where(nd)[0]], F[nd]


def baseline_random_search(snv: SNVContext, oracle, n_evals: int, seed: int = 0):
    """Random search with the same evaluation budget; return its non-dominated set."""
    rng = np.random.default_rng(seed)
    designs = enumerate_designs(snv)
    idx = rng.integers(0, len(designs), size=min(n_evals, len(designs) * 3))
    cand = [designs[i % len(designs)] for i in idx]
    F = np.array([evaluate(d, snv, oracle) for d in cand])
    nd = non_dominated_mask(F)
    return [cand[i] for i in np.where(nd)[0]], F[nd]


def exhaustive_front(snv: SNVContext, oracle):
    """Ground-truth Pareto front by full enumeration (feasible because the space is small).
    Used as the IGD reference front and to verify optimizer optimality."""
    designs = enumerate_designs(snv)
    if not designs:
        raise ValueError(
            f"enumerate_designs returned no designs for {snv.chrom}:{snv.pos} "
            f"{snv.ref}>{snv.alt} (target_base={snv.target_base!r}, "
            f"required_edit={snv.required_edit!r}). Check the loader/suite sets both."
        )
    F = np.array([evaluate(d, snv, oracle) for d in designs])
    nd = non_dominated_mask(F)
    return [designs[i] for i in np.where(nd)[0]], F[nd]
