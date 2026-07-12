"""
optimizers/nsga3.py
===================
From-scratch NSGA-III (Deb & Jain, 2014) for PARETO-EDIT, built on the SAME genotype,
variation operators, and evaluation/counting as NSGA-II so the two are directly comparable
under an identical budget. Only the environmental selection differs: NSGA-III replaces
crowding-distance truncation of the last accepted front with reference-point association +
niche-preserving selection, which is the standard, better-diversified choice for 3+ objectives.

No external dependency (numpy only); pymoo is NOT required.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import List

import numpy as np

from design_space import Design, SNVContext, search_bounds
from optimizers.base import (
    EvalFn, Optimizer, OptResult, fast_non_dominated_sort,
)


def das_dennis(n_obj: int, n_partitions: int) -> np.ndarray:
    """Das-Dennis structured reference points on the unit simplex (sum to 1)."""
    if n_partitions == 0:
        return np.full((1, n_obj), 1.0 / n_obj)
    pts = []
    for c in combinations_with_replacement(range(n_partitions + 1), n_obj - 1):
        cuts = [0] + list(c) + [n_partitions]
        # not used directly; build via the standard recursive composition below
        break
    # standard composition: all vectors of non-negative ints summing to n_partitions
    def rec(left, depth):
        if depth == n_obj - 1:
            yield [left]
            return
        for i in range(left + 1):
            for tail in rec(left - i, depth + 1):
                yield [i] + tail
    for comp in rec(n_partitions, 0):
        pts.append([v / n_partitions for v in comp])
    return np.array(pts, dtype=float)


class NSGA3(Optimizer):
    name = "NSGA-III"

    def __init__(self, pop_size: int = 60, n_gen: int = 40, p_cross: float = 0.9,
                 p_mut: float | None = None, n_partitions: int | None = None):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.n_partitions = n_partitions  # default chosen from n_obj in optimize()

    # ---- genotype <-> Design (identical to NSGA-II) ---------------------- #
    def _decode(self, g: np.ndarray) -> Design:
        return Design(editor=self.editors[int(g[0])],
                      pam_variant=self.pams[int(g[1])],
                      target_pos=int(g[2]))

    def _random_genome(self, rng) -> np.ndarray:
        return np.array([rng.integers(0, len(self.editors)),
                         rng.integers(0, len(self.pams)),
                         rng.integers(self.off_lo, self.off_hi + 1)], dtype=int)

    def _eval_pop(self, G: np.ndarray, eval_fn: EvalFn) -> np.ndarray:
        first = np.atleast_1d(eval_fn(self._decode(G[0])))
        F = np.empty((len(G), first.shape[0]))
        F[0] = first
        for i in range(1, len(G)):
            F[i] = eval_fn(self._decode(G[i]))
        self.n_evals += len(G)
        return F

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

    def _tournament(self, ranks, rng):
        # NSGA-III uses random parent selection (no crowding); tie-break by rank only.
        i, j = rng.integers(0, len(ranks), size=2)
        return i if ranks[i] <= ranks[j] else j

    # ---- NSGA-III environmental selection -------------------------------- #
    def _associate(self, F_norm, ref):
        # perpendicular distance of each point to each reference line (through origin)
        # d = || x - (x·r/ r·r) r ||
        dists = np.empty((F_norm.shape[0], ref.shape[0]))
        rn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
        for i, x in enumerate(F_norm):
            proj = (x @ rn.T)[:, None] * rn
            dists[i] = np.linalg.norm(x[None, :] - proj, axis=1)
        niche = dists.argmin(axis=1)
        d = dists[np.arange(F_norm.shape[0]), niche]
        return niche, d

    def _select(self, F, fronts, rng):
        """Return indices (into F) of the surviving pop_size solutions via NSGA-III niching."""
        chosen: List[int] = []
        last = None
        for fr in fronts:
            if len(chosen) + len(fr) <= self.pop_size:
                chosen.extend(fr)
                if len(chosen) == self.pop_size:
                    return chosen
            else:
                last = fr
                break
        if last is None:
            return chosen[:self.pop_size]

        # normalize objectives by ideal + nadir over the candidate pool (chosen + last)
        pool = np.array(chosen + last, dtype=int)
        Fp = F[pool]
        ideal = Fp.min(axis=0)
        nadir = Fp.max(axis=0)
        span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
        F_norm = (F - ideal) / span

        niche, d = self._associate(F_norm, self.ref)
        # niche counts among already-chosen
        counts = np.zeros(self.ref.shape[0], dtype=int)
        for idx in chosen:
            counts[niche[idx]] += 1

        last_arr = np.array(last, dtype=int)
        remaining = set(last_arr.tolist())
        while len(chosen) < self.pop_size and remaining:
            # pick least-crowded niche that still has candidates in `remaining`
            cand_niches = {niche[i] for i in remaining}
            j = min(cand_niches, key=lambda nj: (counts[nj], rng.random()))
            members = [i for i in remaining if niche[i] == j]
            if counts[j] == 0:
                pick = min(members, key=lambda i: d[i])   # closest to ref line
            else:
                pick = members[rng.integers(0, len(members))]
            chosen.append(int(pick))
            remaining.discard(pick)
            counts[j] += 1
        return chosen[:self.pop_size]

    # ---- main loop (mirrors NSGA-II; only _select differs) --------------- #
    def optimize(self, designs: List[Design], snv: SNVContext, eval_fn: EvalFn,
                 seed: int = 0) -> OptResult:
        rng = np.random.default_rng(seed)
        _all_ed, self.pams, (self.off_lo, self.off_hi) = search_bounds(snv)
        present = list(dict.fromkeys(d.editor for d in designs))
        self.editors = present or _all_ed
        if not self.editors:
            raise ValueError("No editor chemistry matches this SNV's target base.")
        self.p_mut = self.p_mut or (1.0 / 3.0)
        self.n_evals = 0

        # reference directions from the objective count
        n_obj = np.atleast_1d(eval_fn(self._decode(self._random_genome(rng)))).shape[0]
        self.n_evals = 0  # discard the probe call from the count
        parts = self.n_partitions or (12 if n_obj == 3 else 8)
        self.ref = das_dennis(n_obj, parts)

        G = np.array([self._random_genome(rng) for _ in range(self.pop_size)])
        F = self._eval_pop(G, eval_fn)
        history: List[float] = []

        for _ in range(self.n_gen):
            fronts = fast_non_dominated_sort(F)
            ranks = np.empty(len(G), dtype=int)
            for r, fr in enumerate(fronts):
                ranks[fr] = r
            childG = []
            while len(childG) < self.pop_size:
                a = self._tournament(ranks, rng)
                b = self._tournament(ranks, rng)
                c1, c2 = self._crossover(G[a], G[b], rng)
                childG.append(self._mutate(c1, rng))
                if len(childG) < self.pop_size:
                    childG.append(self._mutate(c2, rng))
            childG = np.array(childG)
            childF = self._eval_pop(childG, eval_fn)

            mergedG = np.vstack([G, childG])
            mergedF = np.vstack([F, childF])
            fronts = fast_non_dominated_sort(mergedF)
            keep = self._select(mergedF, fronts, rng)
            G = mergedG[keep]
            F = mergedF[keep]

        from optimizers.base import non_dominated_mask
        nd = non_dominated_mask(F)
        return OptResult(
            designs=[self._decode(g) for g in G[nd]],
            objectives=F[nd],
            history=history,
            n_evals=self.n_evals,
            algo=self.name,
        )
