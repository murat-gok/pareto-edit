"""
optimizers/nsga2.py
===================
From-scratch NSGA-II for PARETO-EDIT.

The decision vector is a mixed (categorical, categorical, integer) genotype:
    [editor_index, pam_index, target_pos]
We use integer encoding with uniform crossover + per-gene resampling mutation, which is the
standard, defensible choice for small categorical/integer multi-objective spaces. Selection,
elitist replacement, fronts and crowding follow Deb et al. 2002.

Written independently of pymoo (pymoo is used only for HV/IGD validation in metrics.py).
"""

from __future__ import annotations

from typing import List

import numpy as np

from design_space import EDITORS, PAM_VARIANTS, OFFSET_RANGE, Design, SNVContext, search_bounds
from optimizers.base import (
    EvalFn,
    Optimizer,
    OptResult,
    crowding_distance,
    fast_non_dominated_sort,
    non_dominated_mask,
)


class NSGA2(Optimizer):
    name = "NSGA-II"

    def __init__(self, pop_size: int = 60, n_gen: int = 40, p_cross: float = 0.9,
                 p_mut: float | None = None):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_cross = p_cross
        self.p_mut = p_mut  # default set in optimize() = 1/n_genes

    # ----- genotype <-> Design -------------------------------------------- #
    def _decode(self, g: np.ndarray) -> Design:
        return Design(
            editor=self.editors[int(g[0])],
            pam_variant=self.pams[int(g[1])],
            target_pos=int(g[2]),
        )

    def _random_genome(self, rng) -> np.ndarray:
        return np.array([
            rng.integers(0, len(self.editors)),
            rng.integers(0, len(self.pams)),
            rng.integers(self.off_lo, self.off_hi + 1),
        ], dtype=int)

    def _eval_pop(self, G: np.ndarray, eval_fn: EvalFn) -> np.ndarray:
        first = np.atleast_1d(eval_fn(self._decode(G[0])))
        F = np.empty((len(G), first.shape[0]))
        F[0] = first
        for i in range(1, len(G)):
            F[i] = eval_fn(self._decode(G[i]))
        self.n_evals += len(G)
        return F

    # ----- variation operators -------------------------------------------- #
    def _crossover(self, p1, p2, rng):
        if rng.random() > self.p_cross:
            return p1.copy(), p2.copy()
        mask = rng.random(3) < 0.5
        c1, c2 = p1.copy(), p2.copy()
        c1[mask], c2[mask] = p2[mask], p1[mask]
        return c1, c2

    def _mutate(self, g, rng):
        for k in range(3):
            if rng.random() < self.p_mut:
                if k == 0:
                    g[0] = rng.integers(0, len(self.editors))
                elif k == 1:
                    g[1] = rng.integers(0, len(self.pams))
                else:
                    g[2] = rng.integers(self.off_lo, self.off_hi + 1)
        return g

    def _tournament(self, ranks, crowd, rng):
        i, j = rng.integers(0, self.pop_size, size=2)
        if ranks[i] < ranks[j]:
            return i
        if ranks[j] < ranks[i]:
            return j
        return i if crowd[i] > crowd[j] else j

    # ----- main loop ------------------------------------------------------- #
    def optimize(self, designs: List[Design], snv: SNVContext, eval_fn: EvalFn,
                 seed: int = 0) -> OptResult:
        rng = np.random.default_rng(seed)
        _all_ed, self.pams, (self.off_lo, self.off_hi) = search_bounds(snv)
        # Respect the editor set actually present in `designs` (pipeline may have excluded
        # proxy editors); fall back to the full chemistry-matched set if designs is empty.
        present = list(dict.fromkeys(d.editor for d in designs))
        self.editors = present or _all_ed
        if not self.editors:
            raise ValueError("No editor chemistry matches this SNV's target base.")
        self.p_mut = self.p_mut or (1.0 / 3.0)
        self.n_evals = 0

        # init
        G = np.array([self._random_genome(rng) for _ in range(self.pop_size)])
        F = self._eval_pop(G, eval_fn)
        history: List[float] = []

        for _ in range(self.n_gen):
            fronts = fast_non_dominated_sort(F)
            ranks = np.empty(len(G), dtype=int)
            crowd = np.zeros(len(G))
            for r, fr in enumerate(fronts):
                ranks[fr] = r
                crowd[fr] = crowding_distance(F[fr])

            # offspring
            childG = []
            while len(childG) < self.pop_size:
                a = self._tournament(ranks, crowd, rng)
                b = self._tournament(ranks, crowd, rng)
                c1, c2 = self._crossover(G[a], G[b], rng)
                childG.append(self._mutate(c1, rng))
                if len(childG) < self.pop_size:
                    childG.append(self._mutate(c2, rng))
            childG = np.array(childG)
            childF = self._eval_pop(childG, eval_fn)

            # elitist merge + environmental selection
            mergedG = np.vstack([G, childG])
            mergedF = np.vstack([F, childF])
            fronts = fast_non_dominated_sort(mergedF)
            newIdx: List[int] = []
            for fr in fronts:
                if len(newIdx) + len(fr) <= self.pop_size:
                    newIdx.extend(fr)
                else:
                    cd = crowding_distance(mergedF[fr])
                    order = np.argsort(-cd)
                    need = self.pop_size - len(newIdx)
                    newIdx.extend([fr[o] for o in order[:need]])
                    break
            G, F = mergedG[newIdx], mergedF[newIdx]

            # track first-front hypervolume proxy (size of non-dominated set)
            nd = non_dominated_mask(F)
            history.append(int(nd.sum()))

        nd = non_dominated_mask(F)
        return OptResult(
            designs=[self._decode(g) for g in G[nd]],
            objectives=F[nd],
            history=history,
            n_evals=self.n_evals,
            algo=self.name,
        )
