"""
oracles/base.py
===============
Abstract Oracle interface.

An Oracle predicts the biological outcome quantities PARETO-EDIT optimizes. Keeping this as a
stable abstract contract is the whole point: phase-0 ships surrogate oracles; phase-3 swaps in
BE-Hive / DeepBE / CRISPRon / crispAI adapters WITHOUT touching the optimizer or objectives.

All predictions are returned in an OracleReading. Quantities:
    efficiency        : P(target base is edited at all), in [0, 1]
    bystander_purity  : P(edited reads carry ONLY the intended conversion | edited), in [0, 1]
                        -- this is the novel objective.
    offtarget_risk    : aggregate genome-wide off-target hazard, >= 0 (lower is better).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from design_space import Design, SNVContext


@dataclass
class OracleReading:
    efficiency: float
    bystander_purity: float
    offtarget_risk: float
    # optional diagnostics (not used by the optimizer, useful for analysis)
    meta: dict | None = None


class Oracle(ABC):
    """Black-box predictor of editing outcomes for a single design at a single SNV."""

    name: str = "abstract"

    @abstractmethod
    def predict(self, design: Design, snv: SNVContext) -> OracleReading:
        ...

    # Optional: vectorized predict for speed. Default loops over predict().
    def predict_many(self, designs, snv: SNVContext):
        return [self.predict(d, snv) for d in designs]


class CompositeOracle(Oracle):
    """Compose per-objective oracles into one reading.

    This is the architecturally correct setup for the paper: each objective comes from the
    predictor validated for it. E.g. efficiency + bystander purity from BE-Hive/DeepBE, and
    off-target risk from crispAI/CFD. Each source supplies the fields it is authoritative for;
    others are ignored.

        oracle = CompositeOracle(
            efficiency_source=behive, purity_source=behive, offtarget_source=crispai)

    Any source may be None, in which case that field falls back to `default`.
    """

    name = "composite"

    def __init__(self, efficiency_source: "Oracle", purity_source: "Oracle",
                 offtarget_source: "Oracle", default: float = 0.0,
                 efficiency_calibrator=None):
        self.eff_src = efficiency_source
        self.pur_src = purity_source
        self.off_src = offtarget_source
        self.default = default
        # optional CalibratedEfficiency mapping surrogate efficiency -> measurement-calibrated
        # efficiency (fitted from BE-dataHIVE). If None, efficiency is used as-is.
        self.eff_cal = efficiency_calibrator

    def predict(self, design: Design, snv: SNVContext) -> OracleReading:
        eff = self.eff_src.predict(design, snv).efficiency if self.eff_src else self.default
        if self.eff_cal is not None:
            eff = self.eff_cal.apply(eff)
        pur = self.pur_src.predict(design, snv).bystander_purity if self.pur_src else self.default
        off = self.off_src.predict(design, snv).offtarget_risk if self.off_src else self.default
        return OracleReading(efficiency=eff, bystander_purity=pur, offtarget_risk=off,
                             meta={"composite": True, "eff_calibrated": self.eff_cal is not None})
