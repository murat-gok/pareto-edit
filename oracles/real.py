"""
oracles/real.py
===============
Phase-3 adapters for REAL predictors. These are stubs that document the integration contract.
Each wraps an external, publicly available model and maps its output onto OracleReading.
Implement these when the models/data are downloaded; the optimizer never changes.

Integration notes (verified public sources):
  * BE-Hive (Arbab et al., Cell 2020): efficiency + full genotype/bystander spectra.
      data: figshare 10.6084/m9.figshare.10673816 ; model code: github.com/maxwshen/be_predict_*
  * DeepBE (Kim et al., 2023): 63 editor combinations; SRA PRJNA821929 ;
      code: github.com/NahyeKim/DeepBE
  * CRISPRon (Xiang et al., Nat Commun 2021): on-target activity + ΔG_B binding energy ->
      maps naturally onto the PAM/binding prior used in efficiency.
  * crispAI (Ozden et al., NAR 2024): ZINB off-target with uncertainty -> offtarget_risk
      (use crispAI-aggregate genome-wide score).
  * BE-dataHIVE (BMC Bioinformatics 2024): >460k gRNA combinations; empirical LOOKUP for
      validation rather than prediction.
"""

from __future__ import annotations

from design_space import Design, SNVContext
from oracles.base import Oracle, OracleReading


class BEHiveOracle(Oracle):
    name = "be-hive"

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        raise NotImplementedError(
            "Phase-3: load BE-Hive bystander+efficiency models from model_dir. "
            "Map predicted genotype distribution -> efficiency (P edited) and "
            "bystander_purity (fraction of edited reads with only the intended edit)."
        )

    def predict(self, design: Design, snv: SNVContext) -> OracleReading:  # pragma: no cover
        raise NotImplementedError


class DeepBEOracle(Oracle):
    name = "deepbe"

    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        raise NotImplementedError("Phase-3: wrap DeepBE inference (63 editor combinations).")

    def predict(self, design: Design, snv: SNVContext) -> OracleReading:  # pragma: no cover
        raise NotImplementedError


class CrispAIOffTargetOracle(Oracle):
    name = "crispai"

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        raise NotImplementedError("Phase-3: wrap crispAI-aggregate genome-wide off-target score.")

    def predict(self, design: Design, snv: SNVContext) -> OracleReading:  # pragma: no cover
        raise NotImplementedError
