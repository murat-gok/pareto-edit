"""
objectives.py
=============
Maps a Design (via an Oracle) to an objective vector to be MINIMIZED.

Convention (critical, used everywhere downstream):
    f0 = -efficiency          (maximize efficiency  -> minimize negative)
    f1 = -bystander_purity    (maximize purity      -> minimize negative)  [NOVEL]
    f2 =  offtarget_risk      (minimize risk)

So the ideal point is [-1, -1, 0] and lower is always better on every axis.
"""

from __future__ import annotations

from typing import List

import numpy as np

from design_space import Design, SNVContext
from oracles.base import Oracle

OBJ_NAMES = ["neg_efficiency", "neg_bystander_purity", "offtarget_risk"]
N_OBJ = 3


def evaluate(design: Design, snv: SNVContext, oracle: Oracle) -> np.ndarray:
    r = oracle.predict(design, snv)
    return np.array([-r.efficiency, -r.bystander_purity, r.offtarget_risk], dtype=float)


def evaluate_many(designs: List[Design], snv: SNVContext, oracle: Oracle) -> np.ndarray:
    readings = oracle.predict_many(designs, snv)
    return np.array(
        [[-r.efficiency, -r.bystander_purity, r.offtarget_risk] for r in readings],
        dtype=float,
    )


def to_human(f: np.ndarray) -> dict:
    """Convert a minimization vector back to interpretable quantities."""
    return {
        "efficiency": float(-f[0]),
        "bystander_purity": float(-f[1]),
        "offtarget_risk": float(f[2]),
    }
