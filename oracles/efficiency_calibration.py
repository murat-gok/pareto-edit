"""
oracles/efficiency_calibration.py
=================================
Calibrate the surrogate's efficiency component against real measured efficiencies from
BE-dataHIVE, so the efficiency objective is grounded in measured data rather than being a purely
analytic stand-in. This is the lightweight route to a measurement-backed efficiency objective: we
keep the surrogate's biophysical *shape* (position/editor/PAM dependence) but rescale its output
to match the empirical efficiency distribution observed in BE-dataHIVE.

How it works
------------
1. From a BE-dataHIVE efficiency table (the local file fetched by fetch_be_datahive.py), we read
   measured efficiencies (`efficiency_full_grna_calculated`). These are on a 0–100 (%) or 0–1
   scale depending on the snapshot; we normalize to 0–1.
2. We fit a monotonic (isotonic) mapping from the surrogate's predicted efficiency to measured
   efficiency. Because BE-dataHIVE guides do not coincide with arbitrary loci (the same overlap
   issue as recovery), we calibrate on the *distribution* level: we align the surrogate's
   efficiency quantiles to BE-dataHIVE's measured-efficiency quantiles (quantile mapping). This
   needs no per-guide correspondence — only the two marginal distributions.
3. The fitted mapping is saved to JSON and applied by CalibratedEfficiency at predict time.

Quantile mapping is robust and assumption-light: it guarantees the calibrated efficiencies have
the same marginal distribution as real measured efficiencies, correcting any systematic over- or
under-confidence in the surrogate's absolute scale while preserving its ranking of designs.

Usage
-----
    # fit once from the local BE-dataHIVE file, produce efficiency_calibration.json
    python -m oracles.efficiency_calibration fit \
        --be-datahive-file be_datahive_efficiency.pkl --out efficiency_calibration.json

    # then in code:
    cal = CalibratedEfficiency.load("efficiency_calibration.json")
    eff = cal.apply(surrogate_efficiency)
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def _normalize_eff(x: np.ndarray) -> np.ndarray:
    """Normalize measured efficiency to 0–1. BE-dataHIVE snapshots may store 0–100 (%) or 0–1."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size and np.nanmax(x) > 1.5:   # looks like a percentage
        x = x / 100.0
    return np.clip(x, 0.0, 1.0)


class CalibratedEfficiency:
    """Quantile-mapping calibrator: maps a surrogate efficiency in [0,1] to a calibrated value
    whose marginal distribution matches BE-dataHIVE measured efficiencies."""

    def __init__(self, src_quantiles: np.ndarray, tgt_quantiles: np.ndarray):
        # src_quantiles: surrogate-efficiency quantile grid (monotone increasing in [0,1])
        # tgt_quantiles: corresponding measured-efficiency quantiles (monotone increasing)
        self.src = np.asarray(src_quantiles, dtype=float)
        self.tgt = np.asarray(tgt_quantiles, dtype=float)

    def apply(self, eff: float) -> float:
        """Map a single surrogate efficiency to its calibrated value via the quantile curve."""
        e = float(min(max(eff, 0.0), 1.0))
        return float(np.interp(e, self.src, self.tgt))

    # ---- persistence ---- #
    def to_json(self, path: str):
        with open(path, "w") as fh:
            json.dump({"src_quantiles": self.src.tolist(),
                       "tgt_quantiles": self.tgt.tolist(),
                       "method": "quantile_mapping"}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibratedEfficiency":
        with open(path) as fh:
            d = json.load(fh)
        return cls(np.array(d["src_quantiles"]), np.array(d["tgt_quantiles"]))

    @classmethod
    def identity(cls) -> "CalibratedEfficiency":
        g = np.linspace(0, 1, 11)
        return cls(g, g)


def fit_calibration(be_datahive_file: str, n_quantiles: int = 50,
                    surrogate_sample: Optional[np.ndarray] = None) -> CalibratedEfficiency:
    """Fit a quantile-mapping calibrator from a local BE-dataHIVE file.

    src distribution: the surrogate efficiency distribution over the design space (sampled by
    enumerating designs for a few representative SNVs and scoring with the surrogate). If
    `surrogate_sample` is provided it is used directly; otherwise we build it here.
    tgt distribution: BE-dataHIVE measured efficiencies (normalized to 0–1).
    """
    import pandas as pd

    # ---- target: measured efficiency distribution ----
    if be_datahive_file.endswith(".parquet"):
        df = pd.read_parquet(be_datahive_file)
    else:
        df = pd.read_pickle(be_datahive_file)
    eff_col = None
    for c in ["efficiency_full_grna_calculated", "efficiency", "editing_efficiency"]:
        if c in df.columns:
            eff_col = c
            break
    if eff_col is None:
        raise ValueError("No efficiency column found in BE-dataHIVE file.")
    measured = _normalize_eff(df[eff_col].values)
    if measured.size == 0:
        raise ValueError("No finite measured efficiencies.")

    # ---- source: surrogate efficiency distribution ----
    if surrogate_sample is None:
        surrogate_sample = _sample_surrogate_efficiencies()
    src = np.clip(np.asarray(surrogate_sample, dtype=float), 0.0, 1.0)

    qs = np.linspace(0, 1, n_quantiles)
    src_q = np.quantile(src, qs)
    tgt_q = np.quantile(measured, qs)
    # enforce strict monotonicity for interp stability
    src_q = np.maximum.accumulate(src_q)
    tgt_q = np.maximum.accumulate(tgt_q)
    return CalibratedEfficiency(src_q, tgt_q)


def _sample_surrogate_efficiencies(n_snv: int = 30) -> np.ndarray:
    """Enumerate designs for a sample of synthetic SNVs and collect surrogate efficiencies, to
    characterize the surrogate's efficiency marginal distribution."""
    from design_space import enumerate_designs
    from oracles.surrogate import SurrogateOracle
    try:
        from data.clinvar import synthetic_clinvar_problems, variant_to_snv
        snvs = [variant_to_snv(v) for v in synthetic_clinvar_problems(n=n_snv, seed=0)]
    except Exception:
        from design_space import SNVContext
        snvs = [SNVContext(chrom="chr1", pos=1000 + i, ref="A", alt="G",
                           target_base="C" if i % 2 else "A") for i in range(n_snv)]
    sur = SurrogateOracle()
    effs = []
    for snv in snvs:
        for d in enumerate_designs(snv):
            try:
                effs.append(sur.predict(d, snv).efficiency)
            except Exception:
                pass
    return np.array(effs) if effs else np.linspace(0, 1, 100)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fit")
    fp.add_argument("--be-datahive-file", required=True)
    fp.add_argument("--out", default="efficiency_calibration.json")
    fp.add_argument("--n-quantiles", type=int, default=50)
    args = ap.parse_args()
    if args.cmd == "fit":
        cal = fit_calibration(args.be_datahive_file, n_quantiles=args.n_quantiles)
        cal.to_json(args.out)
        # report the mapping briefly
        print(f"Fitted quantile-mapping calibration -> {args.out}")
        for e in [0.1, 0.3, 0.5, 0.7, 0.9]:
            print(f"  surrogate {e:.2f} -> calibrated {cal.apply(e):.3f}")
