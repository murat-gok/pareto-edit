"""
optimizers/cpo_gwo.py
=====================
Multi-objective CPO-GWO hybrid for PARETO-EDIT.

MANUSCRIPT POSITIONING (important): in the PARETO-EDIT paper this optimizer is used ONLY as a
secondary comparator / applied existing metaheuristic -- NOT as an algorithmic contribution.
The novelty of PARETO-EDIT is the multi-objective formulation of base-editing design, the
bystander-purity objective, and the pluggable oracle abstraction. A single-objective CPO-GWO
hybrid is described separately for microarray gene selection (Gok et al., under review); this
module is an external-archive multi-objective adaptation used here purely to confirm results
are not optimizer-specific. All paper conclusions are anchored on NSGA-II + the exhaustive
reference front and remain valid if this comparator is removed. See METHODS_optimizers.md.

This adapts the Crested Porcupine Optimizer (CPO) defense mechanisms together with the Grey
Wolf Optimizer (GWO) leader hierarchy into a multi-objective scheme via an EXTERNAL ARCHIVE of
non-dominated solutions (the MOPSO/MOGWO pattern). It mirrors the single-objective CPO-GWO
hybrid the author has previously developed for gene selection, here generalized to a Pareto
archive and a mixed categorical/integer genome.

Continuous-relaxation strategy: we operate on a continuous position vector x in R^3 and DECODE
to a Design by rounding/clipping. This lets the GWO position-update equations and the CPO
defense operators apply directly, while honoring the discrete decision space at evaluation time.

Leaders (alpha/beta/delta) are drawn from the archive using crowding-based selection so the
search spreads along the front rather than collapsing to one knee point.

CPO defenses (four mechanisms, applied probabilistically as the population "shrinks"):
   1. sight   - move away from a threatening direction (exploration)
   2. sound   - oscillatory perturbation
   3. odor    - diffusion toward archive leader with noise
   4. physical- contraction toward best leader (exploitation)
Cyclic population reduction (CPO's hallmark) is emulated by annealing the exploration radius.
"""

from __future__ import annotations

from typing import List

import numpy as np

from design_space import Design, SNVContext, search_bounds
from optimizers.base import EvalFn, Optimizer, OptResult, dominates, non_dominated_mask, crowding_distance


class CPO_GWO(Optimizer):
    name = "CPO-GWO"

    def __init__(self, pop_size: int = 60, n_gen: int = 40, archive_max: int = 100):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.archive_max = archive_max

    # ----- continuous <-> Design ------------------------------------------ #
    def _bounds(self, snv: SNVContext, present_editors=None):
        all_ed, self.pams, (self.off_lo, self.off_hi) = search_bounds(snv)
        self.editors = (list(dict.fromkeys(present_editors)) if present_editors else all_ed) or all_ed
        self.lb = np.array([0.0, 0.0, self.off_lo - 0.4999])
        self.ub = np.array([len(self.editors) - 1e-6, len(self.pams) - 1e-6, self.off_hi + 0.4999])

    def _decode(self, x: np.ndarray) -> Design:
        xi = np.clip(x, self.lb, self.ub)
        return Design(
            editor=self.editors[int(np.floor(xi[0]))],
            pam_variant=self.pams[int(np.floor(xi[1]))],
            target_pos=int(round(xi[2])),
        )

    def _eval(self, x: np.ndarray, eval_fn: EvalFn) -> np.ndarray:
        self.n_evals += 1
        return eval_fn(self._decode(x))

    # ----- archive management --------------------------------------------- #
    def _archive_add(self, X, F, x, f):
        # remove archive members dominated by f; reject x if dominated by any member
        keep_X, keep_F = [], []
        dominated_by_new = False
        for xa, fa in zip(X, F):
            if dominates(fa, f):
                dominated_by_new = True
            if not dominates(f, fa):
                keep_X.append(xa)
                keep_F.append(fa)
        if not dominated_by_new:
            # also avoid duplicates
            if not any(np.allclose(x, xa) for xa in keep_X):
                keep_X.append(x.copy())
                keep_F.append(f.copy())
        # truncate by crowding if oversized
        if len(keep_X) > self.archive_max:
            Fm = np.array(keep_F)
            cd = crowding_distance(Fm)
            order = np.argsort(-cd)[: self.archive_max]
            keep_X = [keep_X[o] for o in order]
            keep_F = [keep_F[o] for o in order]
        return keep_X, keep_F

    def _pick_leaders(self, X, F, rng, k=3):
        if len(X) == 0:
            return None
        Fm = np.array(F)
        cd = crowding_distance(Fm)
        # prefer less-crowded (more isolated) leaders to promote spread
        probs = cd.astype(float).copy()
        finite = np.isfinite(probs)
        if finite.any():
            mx = probs[finite].max()
            mx = mx if mx > 0 else 1.0
            probs[~finite] = mx * 2.0  # boundary points get high weight
        else:
            probs[:] = 1.0  # all infinite (e.g. <=2 members): uniform
        # guarantee every member has strictly positive selection probability so that
        # sampling without replacement always has enough non-zero entries.
        eps = probs.max() * 1e-3 if probs.max() > 0 else 1.0
        probs = probs + eps
        probs = probs / probs.sum()
        n = len(X)
        size = min(k, n)
        replace = n < k
        idx = rng.choice(n, size=size, replace=replace, p=probs)
        return [X[i] for i in idx]

    # ----- main loop ------------------------------------------------------- #
    def optimize(self, designs: List[Design], snv: SNVContext, eval_fn: EvalFn,
                 seed: int = 0) -> OptResult:
        rng = np.random.default_rng(seed)
        self._bounds(snv, present_editors=[d.editor for d in designs])
        if not self.editors:
            raise ValueError("No editor chemistry matches this SNV's target base.")
        self.n_evals = 0
        dim = 3

        X = rng.uniform(self.lb, self.ub, size=(self.pop_size, dim))
        F = np.array([self._eval(x, eval_fn) for x in X])

        # init archive with non-dominated members
        nd = non_dominated_mask(F)
        archX = [x.copy() for x in X[nd]]
        archF = [f.copy() for f in F[nd]]

        history: List[float] = []

        for t in range(self.n_gen):
            # GWO coefficient a anneals 2 -> 0 (exploration -> exploitation).
            a = 2.0 * (1.0 - t / self.n_gen)
            # CPO cyclic population-reduction proxy: defense intensity grows over time.
            defense = t / self.n_gen

            leaders = self._pick_leaders(archX, archF, rng, k=3)
            for i in range(self.pop_size):
                x = X[i]
                if leaders is None:
                    new = x + rng.normal(0, 0.5, dim)
                else:
                    # --- GWO position update toward up to 3 archive leaders ---
                    parts = []
                    for L in leaders:
                        A = 2 * a * rng.random(dim) - a
                        C = 2 * rng.random(dim)
                        D = np.abs(C * L - x)
                        parts.append(L - A * D)
                    new = np.mean(parts, axis=0)

                    # --- CPO defense mechanisms (probabilistic) ---
                    r = rng.random()
                    leader0 = leaders[0]
                    if r < 0.25:                         # sight: veer away from a random peer
                        peer = X[rng.integers(0, self.pop_size)]
                        new = new + (1 - defense) * rng.random() * (new - peer)
                    elif r < 0.50:                       # sound: oscillatory perturbation
                        new = new + np.sin(2 * np.pi * rng.random(dim)) * (1 - defense) * (self.ub - self.lb) * 0.1
                    elif r < 0.75:                       # odor: diffusion toward leader + noise
                        new = new + rng.random() * (leader0 - new) + rng.normal(0, 0.15 * (1 - defense), dim)
                    else:                                # physical: contraction (exploitation)
                        new = new + defense * rng.random() * (leader0 - new)

                new = np.clip(new, self.lb, self.ub)
                fnew = self._eval(new, eval_fn)

                # greedy replacement w.r.t. Pareto dominance (keep non-dominated movement)
                if dominates(fnew, F[i]) or not dominates(F[i], fnew):
                    X[i], F[i] = new, fnew
                archX, archF = self._archive_add(archX, archF, new, fnew)

            history.append(len(archX))

        Fm = np.array(archF)
        return OptResult(
            designs=[self._decode(x) for x in archX],
            objectives=Fm,
            history=history,
            n_evals=self.n_evals,
            algo=self.name,
        )
