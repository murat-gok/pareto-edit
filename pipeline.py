"""
pipeline.py
===========
End-to-end PARETO-EDIT pipeline: SNV -> enumerate designs -> multi-objective optimize ->
Pareto front of (editor, PAM, placement) designs with interpretable objective values.

Usage:
    from pipeline import ParetoEditPipeline
    pipe = ParetoEditPipeline(oracle=SurrogateOracle(), optimizer=NSGA2())
    result = pipe.run(snv)
    pipe.report(result, snv)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import numpy as np

from design_space import Design, SNVContext, enumerate_designs
from objectives import evaluate, to_human
from oracles.base import Oracle
from optimizers.base import Optimizer, OptResult


@dataclass
class ParetoEditPipeline:
    oracle: Oracle
    optimizer: Optimizer
    exclude_editors: tuple = ()

@dataclass
class ParetoEditPipeline:
    oracle: Oracle
    optimizer: Optimizer
    exclude_editors: tuple = ()
    objective_mask: tuple = (0, 1, 2)   # which objectives the optimizer sees (0=eff,1=purity,2=offtgt)
    collapse_pam: bool = False          # see design_space.enumerate_designs

    def run(self, snv: SNVContext, seed: int = 0) -> OptResult:
        designs = enumerate_designs(snv, exclude_editors=self.exclude_editors,
                                    collapse_pam=self.collapse_pam)
        if not designs:
            raise ValueError("No feasible designs (editor chemistry mismatch with SNV).")

        # cached evaluation: the oracle is the cost bottleneck once real models are plugged in
        cache: dict = {}
        mask = list(self.objective_mask)

        def eval_fn(d: Design) -> np.ndarray:
            key = d.as_tuple()
            if key not in cache:
                try:
                    cache[key] = evaluate(d, snv, self.oracle)
                except Exception as e:
                    if type(e).__name__ == "UnsupportedDesign":
                        cache[key] = np.array([1.0, 1.0, 1e6], dtype=float)
                    else:
                        raise
            full = cache[key]
            # the optimizer only sees the masked objectives (for objective-dropping ablation);
            # the full 3-objective vector is retained in `cache` for post-hoc evaluation.
            return full[mask] if len(mask) < 3 else full

        result = self.optimizer.optimize(designs, snv, eval_fn, seed=seed)
        result.n_evals = len(cache)  # report UNIQUE oracle calls (caching matters)
        result = self._dedup(result)
        return result

    @staticmethod
    def _dedup(result: OptResult) -> OptResult:
        """Collapse distinct genotypes that decode to the same Design (common after
        continuous relaxation in CPO-GWO) so the front reflects unique designs."""
        seen = {}
        for d, f in zip(result.designs, result.objectives):
            seen.setdefault(d.as_tuple(), (d, f))
        result.designs = [v[0] for v in seen.values()]
        result.objectives = np.array([v[1] for v in seen.values()])
        return result

    def report(self, result: OptResult, snv: SNVContext, top: int = 10) -> List[dict]:
        rows = []
        for d, f in zip(result.designs, result.objectives):
            h = to_human(f)
            rows.append({
                "editor": d.editor,
                "pam_variant": d.pam_variant,
                "target_pos": d.target_pos,
                **{k: round(v, 4) for k, v in h.items()},
            })
        # sort the front for readable display: efficiency desc, then purity desc
        rows.sort(key=lambda r: (-r["efficiency"], -r["bystander_purity"], r["offtarget_risk"]))
        return rows[:top]
