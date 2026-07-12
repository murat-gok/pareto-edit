"""
design_space.py
================
Decision-variable encoding for PARETO-EDIT.

A *design* is a choice of:
    - base editor variant      (defines editing window + chemistry: C->T, A->G, ...)
    - PAM / Cas9 variant       (defines which PAMs are usable + a binding prior)
    - protospacer offset       (where the 20-nt protospacer sits relative to the target base)
    - guide sequence           (derived deterministically from offset + genomic context)

The optimizer searches over (editor, pam_variant, offset). The guide sequence and the
positions of bystander bases follow deterministically from those choices + the local
genomic context of the target SNV. This keeps the search space discrete and small enough
for metaheuristics while remaining biologically faithful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ----------------------------------------------------------------------------- #
# Static catalogues (phase-0 surrogate values; phase-1 replaces with fitted data)
# ----------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BaseEditor:
    """A base-editor variant.

    edit: the conversion it performs on the sense strand, e.g. ('C', 'T').
    window: 1-indexed (start, end) editing window in protospacer coordinates
            (position 1 = PAM-distal end, as is conventional for SpCas9 BEs).
    peak: position of maximal activity inside the window.
    sharpness: how steeply activity falls off away from the peak (higher = narrower).
    intrinsic_eff: baseline efficiency multiplier for this editor.
    """
    name: str
    edit: Tuple[str, str]
    window: Tuple[int, int]
    peak: float
    sharpness: float
    intrinsic_eff: float


@dataclass(frozen=True)
class PAMVariant:
    """A Cas9 / PAM variant.

    pattern: PAM motif using IUPAC-ish 'N' wildcards, e.g. 'NGG'.
    binding_prior: relative on-target binding propensity (CRISPRon-style energy proxy).
    """
    name: str
    pattern: str
    binding_prior: float


# A compact but representative catalogue. Values are biologically motivated placeholders
# (phase-1 calibrates them against Arbab 2020 window profiles & PAM efficiency priors).
EDITORS: Dict[str, BaseEditor] = {
    "BE4max":     BaseEditor("BE4max",     ("C", "T"), window=(4, 8),  peak=6.0, sharpness=0.45, intrinsic_eff=0.75),
    "evoAPOBEC":  BaseEditor("evoAPOBEC",  ("C", "T"), window=(2, 10), peak=6.0, sharpness=0.25, intrinsic_eff=0.65),
    "YE1":        BaseEditor("YE1",        ("C", "T"), window=(5, 7),  peak=6.0, sharpness=0.80, intrinsic_eff=0.60),  # narrow window -> high purity
    "ABE8e":      BaseEditor("ABE8e",      ("A", "G"), window=(4, 8),  peak=6.0, sharpness=0.40, intrinsic_eff=0.80),
    "ABE7.10":    BaseEditor("ABE7.10",    ("A", "G"), window=(4, 7),  peak=5.5, sharpness=0.55, intrinsic_eff=0.70),
    "CGBE1":      BaseEditor("CGBE1",      ("C", "G"), window=(5, 7),  peak=6.0, sharpness=0.60, intrinsic_eff=0.55),
}

PAM_VARIANTS: Dict[str, PAMVariant] = {
    "SpCas9-NGG":   PAMVariant("SpCas9-NGG",   "NGG",  binding_prior=1.00),
    "SpG-NGN":      PAMVariant("SpG-NGN",      "NGN",  binding_prior=0.85),
    "SpRY-NRN":     PAMVariant("SpRY-NRN",     "NRN",  binding_prior=0.70),
    "SpRY-NYN":     PAMVariant("SpRY-NYN",     "NYN",  binding_prior=0.55),
    "VQR-NGA":      PAMVariant("VQR-NGA",      "NGA",  binding_prior=0.65),
}

PROTOSPACER_LEN = 20
# Allowed protospacer offsets: how far (in bp) the protospacer 5' end is shifted so that the
# target base occupies a given position in the window. We allow the target base to land at
# protospacer positions 1..10 (i.e. offsets that place it across/near the window).
OFFSET_RANGE: Tuple[int, int] = (1, 10)  # target-base position within the protospacer


@dataclass(frozen=True)
class Design:
    """A fully-specified candidate design (one point in the search space)."""
    editor: str          # key into EDITORS
    pam_variant: str      # key into PAM_VARIANTS
    target_pos: int       # 1..PROTOSPACER_LEN : position of the target base in the protospacer

    def as_tuple(self) -> Tuple[str, str, int]:
        return (self.editor, self.pam_variant, self.target_pos)


@dataclass
class SNVContext:
    """The genomic problem instance.

    chrom/pos/ref/alt: standard variant coordinates (ref->alt is the *correction* we want,
        so the editor must perform the alt-creating conversion on the appropriate strand).
    flank_5 / flank_3: genomic sequence flanking the target base (uppercase ACGT),
        long enough to enumerate protospacers + PAMs on both strands.
    Used by the oracles to locate bystander bases of the same identity as the target base
    within the editing window.
    """
    chrom: str
    pos: int
    ref: str
    alt: str
    flank_5: str = ""
    flank_3: str = ""
    target_base: str = ""       # substrate the editor must DEAMINATE (sense-strand)
    required_edit: Tuple[str, str] = ("", "")  # the full correction conversion, e.g. ("A","G")
                                # for a G>A lesion (revert A->G). Set by the ClinVar loader.
                                # Empty = unset. This is what makes an edit a CORRECTION,
                                # not merely "some edit at the target base".
    # local context window centered on the target base (for bystander detection)
    context: str = field(default="")

    def window_context(self, design: Design) -> str:
        """Return the protospacer-length sequence with the target base placed at
        design.target_pos, drawn from the real local genomic context.

        Biological model: the genomic neighborhood of the SNV is FIXED. As the optimizer
        slides the protospacer (changes target_pos), a different 20-nt genomic window is
        captured, so the number/position of same-identity bystander bases changes -- this is
        the true source of the efficiency-vs-purity trade-off. We therefore index a fixed
        long context string by target_pos rather than regenerating it per design.
        """
        if self.context:
            long_ctx = self.context
        else:
            # Synthesize a fixed long neighborhood deterministically from SNV coords.
            # Phase-2 replaces this with real GRCh38 sequence (flank_5 + base + flank_3).
            import hashlib
            h = hashlib.sha256(f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}".encode()).hexdigest()
            bases = "ACGT"
            # build a long context (>= 2*PROTOSPACER_LEN) so any placement has full flanks
            long_ctx = "".join(bases[int(c, 16) % 4] for c in (h * 3)[: 2 * PROTOSPACER_LEN + 1])
            long_ctx = list(long_ctx)
            long_ctx[PROTOSPACER_LEN] = self.target_base or "C"  # center = substrate (surrogate fallback)
            long_ctx = "".join(long_ctx)

        # Place the protospacer so the target base (at center index PROTOSPACER_LEN) lands at
        # protospacer position design.target_pos (1-indexed).
        center = PROTOSPACER_LEN
        start = center - (design.target_pos - 1)
        window = long_ctx[start:start + PROTOSPACER_LEN]
        # guarantee correct length and target identity
        window = (window + "A" * PROTOSPACER_LEN)[:PROTOSPACER_LEN]
        w = list(window)
        w[design.target_pos - 1] = self.target_base or "C"
        return "".join(w)


def enumerate_designs(snv: SNVContext, exclude_editors=None) -> List[Design]:
    """Enumerate all feasible designs for an SNV.

    Feasibility rule (phase-0): the editor's chemistry must be able to produce the desired
    correction, and the target base must be able to fall within (or near) the editing window
    for at least one offset. We keep this permissive; the objective function penalizes
    out-of-window placements rather than hard-filtering, which gives the optimizer a smooth
    landscape to explore.

    exclude_editors: optional iterable of editor names to omit (e.g. proxy-mapped editors for a
    direct-model-only analysis).
    """
    exclude = set(exclude_editors or ())
    designs: List[Design] = []
    if not snv.target_base or snv.required_edit == ("", ""):
        # Unset substrate/edit (or a transversion the loader could not map to CBE/ABE):
        # return nothing rather than silently enumerating the wrong chemistry.
        return designs
    for ed_name, ed in EDITORS.items():
        if ed_name in exclude:
            continue
        # chemistry check: the editor's FULL conversion must equal the required correction
        # (input base AND product). Matching only ed.edit[0] would wrongly admit CGBE
        # (C->G) for a C->T correction, so we compare the whole pair. This is what makes
        # the admitted edit an actual CORRECTION rather than merely "an edit at the target".
        if tuple(ed.edit) != tuple(snv.required_edit):
            continue
        for pam_name in PAM_VARIANTS:
            for pos in range(OFFSET_RANGE[0], OFFSET_RANGE[1] + 1):
                designs.append(Design(editor=ed_name, pam_variant=pam_name, target_pos=pos))
    return designs


def search_bounds(snv: SNVContext, exclude_editors=None) -> Tuple[List[str], List[str], Tuple[int, int]]:
    """Return the categorical levels + integer bounds the optimizer searches over,
    restricted to editors whose chemistry matches the SNV (and not in exclude_editors)."""
    exclude = set(exclude_editors or ())
    editors = [k for k, v in EDITORS.items()
               if tuple(v.edit) == tuple(snv.required_edit) and k not in exclude]
    pams = list(PAM_VARIANTS.keys())
    return editors, pams, OFFSET_RANGE
