"""
oracles/surrogate.py
====================
Phase-0 surrogate oracles. These are NOT the real predictors -- they are biologically
*structured* stand-ins so the entire optimization pipeline runs and can be validated today.
Every formula encodes a real, defensible qualitative relationship that the real oracles
(BE-Hive/DeepBE/CRISPRon/crispAI) also exhibit:

  * Efficiency peaks when the target base sits at the editor's activity peak inside the
    window, scaled by the editor's intrinsic efficiency and the PAM/Cas9 binding prior.
  * Bystander purity is HIGH when there are no same-identity bases elsewhere in the window,
    and falls as nearby bystander bases accumulate (weighted by the window activity curve).
    Narrow-window editors (e.g. YE1) therefore achieve higher purity -- a real trade-off.
  * Off-target risk grows with protospacer GC / self-similarity and with permissive PAMs
    (SpRY > SpG > SpCas9-NGG), a documented qualitative trend.

The point: these surrogates produce a genuine, non-trivial 3-way Pareto trade-off so the
optimizer has something real to discover, and they share the SAME interface as the real models.
"""

from __future__ import annotations

import math

from design_space import (
    EDITORS,
    PAM_VARIANTS,
    PROTOSPACER_LEN,
    Design,
    SNVContext,
)
from oracles.base import Oracle, OracleReading


def _window_activity(pos: int, editor) -> float:
    """Gaussian-ish activity as a function of protospacer position, centered on the
    editor's peak, modulated by sharpness. Returns a value in (0, 1]."""
    return math.exp(-editor.sharpness * (pos - editor.peak) ** 2)


def _in_window(pos: int, editor) -> bool:
    lo, hi = editor.window
    return lo <= pos <= hi


class SurrogateOracle(Oracle):
    name = "surrogate"

    def predict(self, design: Design, snv: SNVContext) -> OracleReading:
        editor = EDITORS[design.editor]
        pam = PAM_VARIANTS[design.pam_variant]
        ctx = snv.window_context(design)  # length == PROTOSPACER_LEN

        # ---- 1) Efficiency -------------------------------------------------- #
        act = _window_activity(design.target_pos, editor)
        efficiency = editor.intrinsic_eff * pam.binding_prior * act
        # if the target base is fully outside the window, efficiency collapses
        if not _in_window(design.target_pos, editor):
            efficiency *= 0.15
        efficiency = float(min(max(efficiency, 0.0), 1.0))

        # ---- 2) Bystander purity ------------------------------------------- #
        # Find other bases in the protospacer with the SAME identity as the target base
        # (these are candidate bystanders the editor could also convert). Weight each by
        # the window activity at its position; purity = 1 / (1 + sum of bystander activity).
        target_id = editor.edit[0]
        bystander_load = 0.0
        for i, b in enumerate(ctx, start=1):
            if i == design.target_pos:
                continue
            if b == target_id and _in_window(i, editor):
                bystander_load += _window_activity(i, editor)
        bystander_purity = 1.0 / (1.0 + bystander_load)
        bystander_purity = float(min(max(bystander_purity, 0.0), 1.0))

        # ---- 3) Off-target risk -------------------------------------------- #
        # GC content of the protospacer (high GC -> more genome-wide near-matches),
        # plus a PAM permissiveness term (permissive PAMs -> more potential off-targets).
        gc = sum(1 for b in ctx if b in "GC") / PROTOSPACER_LEN
        pam_permissiveness = {  # number of genomic matches scales inversely with specificity
            "SpCas9-NGG": 1.0, "SpG-NGN": 1.6, "SpRY-NRN": 2.4,
            "SpRY-NYN": 2.4, "VQR-NGA": 1.4,
        }.get(design.pam_variant, 1.5)
        # simple self-similarity proxy: longest homopolymer run
        run, best = 1, 1
        for a, b in zip(ctx, ctx[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        seed_repeat = best / PROTOSPACER_LEN
        offtarget_risk = (0.5 * gc + 0.3 * seed_repeat) * pam_permissiveness
        offtarget_risk = float(max(offtarget_risk, 0.0))

        return OracleReading(
            efficiency=efficiency,
            bystander_purity=bystander_purity,
            offtarget_risk=offtarget_risk,
            meta={"gc": gc, "bystander_load": bystander_load, "activity": act},
        )
