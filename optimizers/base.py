"""
optimizers/base.py
==================
Abstract Optimizer interface + shared multi-objective primitives.

Both NSGA-II and CPO-GWO implement Optimizer.optimize(...) and return an OptResult, so the
pipeline and benchmarking code treat them interchangeably -- exactly the "common interface,
NSGA-II first" design requested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from design_space import Design, SNVContext


@dataclass
class OptResult:
    designs: List[Design]            # non-dominated designs (the Pareto SET)
    objectives: np.ndarray            # corresponding objective vectors (Pareto FRONT), shape (k, n_obj)
    history: List[float] = field(default_factory=list)  # e.g. hypervolume per generation
    n_evals: int = 0
    algo: str = ""


# Objective function signature shared by all optimizers
EvalFn = Callable[[Design], np.ndarray]


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a Pareto-dominates b (minimization): a <= b everywhere, < somewhere."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows in objective matrix F (minimization).
    Vectorized O(n^2) via broadcasting (fast for the front sizes used here)."""
    F = np.atleast_2d(F)
    n = F.shape[0]
    if n <= 1:
        return np.ones(n, dtype=bool)
    # le[i,j] True if F[j] <= F[i] on all objectives; lt_any[i,j] if F[j] < F[i] on some
    le_all = np.all(F[None, :, :] <= F[:, None, :], axis=2)   # shape (n, n)
    lt_any = np.any(F[None, :, :] <  F[:, None, :], axis=2)
    dominated_by = le_all & lt_any                              # j dominates i
    np.fill_diagonal(dominated_by, False)
    keep = ~np.any(dominated_by, axis=1)
    return keep


def fast_non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    """Deb's fast non-dominated sort. Returns a list of fronts (each a list of indices).
    Domination relationships computed with a vectorized matrix for speed."""
    F = np.atleast_2d(F)
    n = F.shape[0]
    # dom[p, q] True if p dominates q
    le_all = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt_any = np.any(F[:, None, :] <  F[None, :, :], axis=2)
    dom = le_all & lt_any
    np.fill_diagonal(dom, False)

    S = [np.where(dom[p])[0].tolist() for p in range(n)]  # solutions dominated by p
    nd = dom.sum(axis=0).astype(int)                       # how many dominate p
    fronts: List[List[int]] = [[p for p in range(n) if nd[p] == 0]]
    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Crowding distance for a single front (minimization). Larger = more isolated."""
    k, m = F.shape
    if k == 0:
        return np.array([])
    dist = np.zeros(k)
    for obj in range(m):
        order = np.argsort(F[:, obj])
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = F[order[0], obj], F[order[-1], obj]
        span = fmax - fmin
        if span == 0:
            continue
        for idx in range(1, k - 1):
            dist[order[idx]] += (F[order[idx + 1], obj] - F[order[idx - 1], obj]) / span
    return dist


class Optimizer(ABC):
    name: str = "abstract"

    @abstractmethod
    def optimize(
        self,
        designs: List[Design],
        snv: SNVContext,
        eval_fn: EvalFn,
        seed: int = 0,
    ) -> OptResult:
        ...
