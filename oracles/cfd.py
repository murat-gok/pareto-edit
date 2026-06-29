"""
oracles/cfd.py
==============
CFD (Cutting Frequency Determination) off-target oracle (Phase 3b).

Implements the canonical CFD scoring algorithm (Doench et al., Nat. Biotechnol. 2016) to give
the off-target objective a *published* basis instead of the surrogate's heuristic. CFD scores
an sgRNA against a candidate off-target by multiplying position-specific mismatch penalties:

    CFD = product over positions p of  penalty(p, sgRNA_nt[p], offtarget_nt[p])
          * PAM_penalty(off-target PAM)

A perfect match scores 1.0; each mismatch multiplies in a value in (0,1]. PAM-distal mismatches
are better tolerated (higher penalty value) than PAM-proximal (seed) ones.

TWO MODES
---------
1. REAL MATRIX (recommended for the paper). Doench 2016 published the exact penalty matrix as a
   pickle (`mismatch_score.pkl`, keys like 'rG:dT,3' -> activity, and `pam_scores.pkl`). It ships
   in many public repos (e.g. the Broad GPP sgRNA scorer; crisprVerse/crisprScore; the original
   CRISPOR distribution). Provide its path and we load it verbatim. We do NOT hardcode the matrix
   here because reproducing ~480 float values from memory would risk transcription errors — the
   scientifically honest choice is to load the authors' published file.

2. POSITION-WEIGHTED FALLBACK (works with no external file). When the real matrix is absent we
   use a transparent, monotonic approximation: each mismatch incurs a penalty that is milder at
   the PAM-distal end and harsher in the seed region, following the well-established qualitative
   trend. This is clearly labelled as an approximation and is intended only so the pipeline runs;
   the paper's off-target numbers should use mode 1.

OFF-TARGET RISK as an OBJECTIVE
-------------------------------
PARETO-EDIT minimizes off-target *risk*. Without a genome-wide scan we cannot enumerate all
off-target sites, so we define a tractable, defensible risk proxy from the guide itself:

    risk = self_offtarget_propensity(protospacer)

computed as the CFD score the guide would assign to its most plausible near-cognate sites
(single/double-mismatch neighbours weighted by seed position), aggregated. Higher = more prone
to off-targeting. (Phase-3c: replace with a true genome-wide aggregate, e.g. crispAI-aggregate
or a scan over CHANGE-seq-derived candidate sites.)
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

from design_space import PROTOSPACER_LEN, Design, SNVContext
from oracles.base import Oracle, OracleReading

NTS = "ACGT"


class CFDOffTargetOracle(Oracle):
    name = "cfd-offtarget"

    def __init__(self, mismatch_pkl: Optional[str] = None, pam_pkl: Optional[str] = None,
                 aggregate: bool = False, max_mismatches: int = 2):
        self.mm = None
        self.pam = None
        # aggregate=True uses the genome-aware-style aggregate-CFD specificity score (Hsu 2013);
        # aggregate=False uses the lighter single-mismatch propensity (backward compatible).
        self.aggregate = aggregate
        self.max_mismatches = max_mismatches
        if mismatch_pkl and os.path.exists(mismatch_pkl):
            with open(mismatch_pkl, "rb") as f:
                # Doench-2016 pickles are Python-2 era; latin1 decodes them on Python 3.
                self.mm = pickle.load(f, encoding="latin1")
        if pam_pkl and os.path.exists(pam_pkl):
            with open(pam_pkl, "rb") as f:
                self.pam = pickle.load(f, encoding="latin1")
        self.using_real_matrix = self.mm is not None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _revcom(s: str) -> str:
        bc = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A"}
        return "".join(bc[b] for b in s[::-1])

    def _mismatch_penalty(self, pos1: int, sg_nt: str, ot_nt: str) -> float:
        """Retained-activity multiplier for an sgRNA:off-target mismatch at 1-based position
        pos1, following the canonical CFD encoding (Doench 2016 / CRISPOR calc_cfd):
            key = 'r' + sgRNA_base(RNA) + ':d' + revcom(offtarget_base) + ',' + pos
        with T mapped to U. Match -> 1.0.
        """
        if sg_nt == ot_nt:
            return 1.0
        if self.using_real_matrix:
            sg_rna = "U" if sg_nt == "T" else sg_nt
            key = f"r{sg_rna}:d{self._revcom(ot_nt)},{pos1}"
            return float(self.mm.get(key, 0.0))
        # ---- position-weighted fallback (no external file) ----
        frac_distal = (PROTOSPACER_LEN - pos1) / (PROTOSPACER_LEN - 1)
        return 0.05 + 0.80 * frac_distal

    def _cfd_of_offtarget(self, proto: str, mm_positions: tuple, mm_bases: tuple) -> float:
        """CFD score of a specific near-cognate off-target derived from `proto` by substituting
        the given bases at the given 1-based positions. CFD = product of per-position retained
        activities (Doench 2016). Perfect match => 1.0; each mismatch multiplies in its penalty."""
        score = 1.0
        for pos1, ot_nt in zip(mm_positions, mm_bases):
            sg = proto[pos1 - 1]
            score *= self._mismatch_penalty(pos1, sg, ot_nt)
            if score == 0.0:
                break
        return score

    def _aggregate_specificity_risk(self, proto: str, max_mismatches: int = 2,
                                    cfd_cutoff: float = 0.023, seed_start: int = 11) -> float:
        """Genome-aware-style aggregate off-target risk via the CFD aggregate-specificity
        framework (Hsu et al. 2013; Doench et al. 2016), the same formulation used by
        CFD-spec / crispAI-aggregate / MOFF-aggregate. Rather than scanning the whole genome
        (which requires Cas-OFFinder indexing and is left as future work), we enumerate the
        guide's near-cognate off-target sites and aggregate their CFD scores into a specificity
        score:

            specificity = 1 / (1 + sum_of_offtarget_CFD)
            risk        = 1 - specificity                # in [0,1); higher = less specific

        Enumeration is focused on the **specificity-defining region** (the PAM-proximal seed,
        positions `seed_start`..20), where mismatches actually modulate off-target activity and
        which therefore discriminates between guides; PAM-distal mismatches are tolerated almost
        uniformly and add little specificity signal (Doench 2016; CRISPOR). We enumerate up to
        `max_mismatches` substitutions within the seed, score each near-cognate's CFD against the
        real Doench-2016 matrix, count only those above the CRISPOR activity cutoff (0.023), and
        combine the aggregate specificity with the single most dangerous near-cognate. This yields
        a genome-aware-style specificity score restricted to the seed neighbourhood; a full
        genome-wide Cas-OFFinder scan weighting each site by occurrence is left as future work.
        """
        import itertools
        n = min(len(proto), PROTOSPACER_LEN)
        proto = proto[:n]
        # specificity-defining region: PAM-proximal seed positions
        positions = [p for p in range(seed_start, n + 1)]
        if not positions:
            positions = list(range(1, n + 1))
        offtarget_cfd_sum = 0.0
        worst_cfd = 0.0
        for k in range(1, max_mismatches + 1):
            for combo in itertools.combinations(positions, k):
                alt_choices = []
                for p in combo:
                    sg = proto[p - 1]
                    alt_choices.append([b for b in NTS if b != sg])
                for bases in itertools.product(*alt_choices):
                    cfd = self._cfd_of_offtarget(proto, combo, bases)
                    if cfd >= cfd_cutoff:
                        offtarget_cfd_sum += cfd
                        if cfd > worst_cfd:
                            worst_cfd = cfd
        specificity = 1.0 / (1.0 + offtarget_cfd_sum)
        spec_risk = 1.0 - specificity
        return float(0.5 * spec_risk + 0.5 * worst_cfd)

    def _self_offtarget_propensity(self, proto: str) -> float:
        """Off-target propensity of a guide: summed CFD retained-activity of its single-mismatch
        near-cognates. For each protospacer position and each alternative base, the per-position
        CFD penalty equals the activity an off-target carrying that single substitution would
        retain; summing the best (most tolerated) substitution per position gives a propensity
        score (more high-CFD near-cognates => more off-target prone). Normalized by length.

        With the real matrix this is fully sequence- and position-specific. In fallback mode the
        matrix is base-agnostic, so we additionally modulate by GC/seed/repeat content to keep
        the objective discriminative (documented approximation)."""
        n = min(len(proto), PROTOSPACER_LEN)
        proto = proto[:n]
        total = 0.0
        for p in range(1, n + 1):
            sg = proto[p - 1]
            best = 0.0
            for ot in NTS:
                if ot == sg:
                    continue
                best = max(best, self._mismatch_penalty(p, sg, ot))
            total += best
        base = total / n
        if self.using_real_matrix:
            return base
        # fallback sequence-content modulation
        gc = sum(1 for b in proto if b in "GC") / n
        seed = proto[10:] if n >= 11 else proto
        seed_gc = (sum(1 for b in seed if b in "GC") / len(seed)) if seed else 0.0
        run = best_run = 1
        for a, b in zip(proto, proto[1:]):
            run = run + 1 if a == b else 1
            best_run = max(best_run, run)
        repeat = best_run / n
        content = 0.6 + 0.4 * gc + 0.3 * seed_gc + 0.1 * repeat
        return base * content

    # ------------------------------------------------------------------ #
    def predict(self, design: Design, snv: SNVContext) -> OracleReading:
        proto = snv.window_context(design)  # same 20-nt the other oracles see
        if self.aggregate:
            risk = self._aggregate_specificity_risk(proto, max_mismatches=self.max_mismatches)
        else:
            risk = self._self_offtarget_propensity(proto)
        # efficiency / purity are not modeled by CFD; return neutral values (CompositeOracle
        # ignores them and takes only offtarget_risk from this source).
        return OracleReading(efficiency=0.0, bystander_purity=0.0, offtarget_risk=float(risk),
                             meta={"cfd_real_matrix": self.using_real_matrix,
                                   "cfd_aggregate": self.aggregate})
