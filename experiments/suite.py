"""
experiments/suite.py
====================
Benchmark problem suite for PARETO-EDIT Phase-4.

Provides reproducible SNV problem instances and two search-space regimes:

  * "small"  : the default catalogue (a few editors/PAMs). The Pareto front is cheaply
               enumerable, so we can measure EXACT recovery (% of true front found, HV ratio).
               Use this regime to prove optimality/correctness.

  * "scaled" : an enlarged design space (expanded editor/PAM catalogues + finer placement grid)
               under a CAPPED evaluation budget. Exhaustive enumeration becomes expensive, so
               this regime measures search EFFICIENCY -- the situation that matters once real,
               costly oracles (BE-Hive/DeepBE) replace the surrogate. This is where NSGA-II,
               CPO-GWO, weighted-sum and random search actually diverge.

The scaled regime monkey-patches the catalogue sizes at runtime via a context manager so the
rest of the codebase (design_space, oracles, optimizers) is untouched.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import design_space as ds
from design_space import BaseEditor, PAMVariant, SNVContext


# --------------------------------------------------------------------------- #
# Problem instances
# --------------------------------------------------------------------------- #

@dataclass
class Problem:
    name: str
    snv: SNVContext


def make_problems(n_per_chem: int = 6, seed: int = 0) -> List[Problem]:
    """Generate a reproducible suite of SNV instances for both CBE (target C) and ABE
    (target A) chemistries. Coordinates are synthetic but deterministic; Phase-2 swaps in
    real ClinVar pathogenic SNVs with GRCh38 context."""
    rng = np.random.default_rng(seed)
    problems: List[Problem] = []
    for chem, tb, alt in [("CBE", "C", "T"), ("ABE", "A", "G")]:
        for i in range(n_per_chem):
            chrom = f"chr{int(rng.integers(1, 23))}"
            pos = int(rng.integers(1_000_000, 200_000_000))
            problems.append(Problem(
                name=f"{chem}-{i+1:02d}",
                snv=SNVContext(chrom=chrom, pos=pos, ref="X", alt=alt, target_base=tb),
            ))
    return problems


# --------------------------------------------------------------------------- #
# Scaled search-space regime (context manager)
# --------------------------------------------------------------------------- #

def _expanded_editors() -> dict:
    """Build a larger, biologically-plausible editor catalogue by parametrically varying
    window width, peak and sharpness around the known archetypes. Keeps chemistry labels
    (C->T, A->G, C->G) so feasibility filtering still works."""
    base = dict(ds.EDITORS)  # keep the originals
    extra = {}
    # CBE family variants (different windows / sharpness => different purity-efficiency tradeoffs)
    cbe_specs = [
        ("CBE_w3_7_s0.6", ("C", "T"), (3, 7), 5.0, 0.60, 0.70),
        ("CBE_w4_9_s0.30", ("C", "T"), (4, 9), 6.5, 0.30, 0.68),
        ("CBE_w5_6_s1.0", ("C", "T"), (5, 6), 5.5, 1.00, 0.58),
        ("CBE_w2_8_s0.22", ("C", "T"), (2, 8), 5.0, 0.22, 0.62),
        ("CGBE_w4_8_s0.5", ("C", "G"), (4, 8), 6.0, 0.50, 0.57),
    ]
    abe_specs = [
        ("ABE_w4_9_s0.35", ("A", "G"), (4, 9), 6.5, 0.35, 0.78),
        ("ABE_w5_7_s0.7", ("A", "G"), (5, 7), 6.0, 0.70, 0.66),
        ("ABE_w3_8_s0.28", ("A", "G"), (3, 8), 5.5, 0.28, 0.72),
        ("ABE_w4_6_s0.9", ("A", "G"), (4, 6), 5.0, 0.90, 0.63),
    ]
    for name, edit, win, peak, sharp, eff in cbe_specs + abe_specs:
        extra[name] = BaseEditor(name, edit, win, peak, sharp, eff)
    base.update(extra)
    return base


def _expanded_pams() -> dict:
    base = dict(ds.PAM_VARIANTS)
    extra = {
        "SpG-NGH":   PAMVariant("SpG-NGH",   "NGH", 0.80),
        "SpRY-NAN":  PAMVariant("SpRY-NAN",  "NAN", 0.60),
        "xCas9-NG":  PAMVariant("xCas9-NG",  "NG",  0.75),
        "NG-Cas9":   PAMVariant("NG-Cas9",   "NG",  0.72),
    }
    base.update(extra)
    return base


@contextlib.contextmanager
def scaled_space(offset_hi: int = 16):
    """Temporarily enlarge the design space: more editors, more PAMs, finer placement grid.
    Restores the originals on exit so other code is unaffected."""
    saved = (ds.EDITORS, ds.PAM_VARIANTS, ds.OFFSET_RANGE)
    # also patch the names imported into modules that did `from design_space import ...`
    import oracles.surrogate as surro
    import optimizers.nsga2 as n2
    import optimizers.cpo_gwo as cg
    try:
        ds.EDITORS = _expanded_editors()
        ds.PAM_VARIANTS = _expanded_pams()
        ds.OFFSET_RANGE = (1, offset_hi)
        surro.EDITORS = ds.EDITORS
        surro.PAM_VARIANTS = ds.PAM_VARIANTS
        # nsga2/cpo_gwo read EDITORS/PAM_VARIANTS via search_bounds() in design_space, so the
        # ds.* patch above already covers them; OFFSET_RANGE is read through search_bounds too.
        yield
    finally:
        ds.EDITORS, ds.PAM_VARIANTS, ds.OFFSET_RANGE = saved
        surro.EDITORS = ds.EDITORS
        surro.PAM_VARIANTS = ds.PAM_VARIANTS


def space_size(snv: SNVContext) -> int:
    editors, pams, (lo, hi) = ds.search_bounds(snv)
    return len(editors) * len(pams) * (hi - lo + 1)
